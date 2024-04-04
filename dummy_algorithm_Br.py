# This is a test dummy algorithm to get the opportunity cost curves
from ortools.linear_solver import pywraplp
import offer_utils as ou
import pandas as pd
import numpy as np
import argparse
import json
import datetime
from itertools import accumulate

# Standard battery parameters
socmax = 608
socmin = 128
chmax = 125
dcmax = 125
efficiency = 0.892
duration_minutes = 5

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


class Agent():
    '''
    Agent is re-initialized every time the WEASLE Platform calls market_participant.py
    Input: time_step, market_data, and resource_data are input arguments from the script call
    Additional input data must be saved to disc and reloaded each time Agent is created (e.g., to facilitate Agent persistence)
    Output:
    - make_me_an_offer() reads the market type and saves to disc a JSON file containing offer data
    '''

    def __init__(self, time_step, market_info, resource_info):
        # Data input from WEASLE
        self.step = time_step
        self.market = market_info
        self.resource = resource_info
        self.rid = resource_info['rid']

        self.duration_minutes = duration_minutes

        # Standard battery parameters
        self.socmax = socmax
        self.socmin = socmin
        self.chmax = chmax
        self.dcmax = dcmax
        self.efficiency = efficiency

        # Configurable options
        self.price_ceiling = 999
        self.price_floor = 0

        # Add the offer binner
        self.binner = ou.Binner(output_type='lists')

        self._prev_dam_file = 'prev_day_ahead_market'
        self.save_from_previous()

    def make_me_an_offer(self):
        # Read in information from the market
        market_type = self.market["market_type"]
        if 'DAM' in market_type:
            offer = self._day_ahead_offer()
        elif 'RTM' in market_type:
            offer = self._real_time_offer()
        else:
            raise ValueError(f"Unable to find offer function for market_type={market_type}")

        # Then save the result
        self._save_json(offer, f'offer_{self.step}.json')

    def save_from_previous(self):
        # if the current market type is DAM, then we need to save it in order to run RTM
        if 'DAM' in self.market["market_type"]:
            self._save_json(self.market['previous'], self._prev_dam_file)

    def _day_ahead_offer(self):
        # Make the offer curves and unload into arrays
        type = self.market['market_type']
        # bus = self.resource['bus'] #TODO: switch when this is updated
        bus = 'NEVP'
        prices = self.market["previous"][type]["prices"]["EN"][bus]
        self._calculate_offer_curve(prices)
        self._descretize_offer_curves()
        self._format_offer_curves()

        return self.formatted_offer

    def _format_offer_curves(self):
        # Offer parsing script below:
        required_times = [t for t in self.market['timestamps']]

        # Convert the offer curves to timestamp:offer_value dictionaries
        block_ch_mc = {}
        for i, cost in enumerate(self.charge_mc):
            block_ch_mc[required_times[i]] = float(cost)
        block_ch_mq = {}
        for i, power in enumerate(self.charge_mq):
            block_ch_mq[required_times[i]] = float(power)  # 125MW

        block_dc_mc = {}
        block_soc_mc = {}
        for i, cost in enumerate(self.discharge_mc):
            block_dc_mc[required_times[i]] = float(cost)
            block_soc_mc[required_times[i]] = 0

        block_dc_mq = {}
        block_soc_mq = {}
        for i, power in enumerate(self.discharge_mq):
            block_dc_mq[required_times[i]] = float(power)  # 125MW
            block_soc_mq[required_times[i]] = 0

        # estimate initial SoC for tomorrow's DAM
        t_init = datetime.datetime.strptime(self.market['timestamps'][0],'%Y%m%d%H%M')
        t_now = datetime.datetime.strptime(self.market['current_time'],'%Y%m%d%H%M') # update to new market_data
        #t_now = datetime.datetime.strptime(self.market['current_time'][5:],'%Y%m%d%H%M')
        t_init = t_init.strftime('%Y%m%d%H%M')
        t_now = t_now.strftime('%Y%m%d%H%M')
        if self.resource['schedule'].keys():
            schedule = self.resource['schedule'][self.rid]['EN']
            schedule_to_tomorrow = [q for t,q in schedule if t_now <= t < t_init]   # these may be misordered but that is OK
            schedule_to_tomorrow = self._process_efficiency(schedule_to_tomorrow)
            soc_estimate = self.resource['status'][self.rid]['soc'] - sum(schedule_to_tomorrow) * self.duration_minutes / 60
            dispatch_estimate = self.resource['schedule'][self.rid]['EN'][t_init]
        else:
            soc_estimate = self.resource['status'][self.rid]['soc']
            dispatch_estimate = 0
        soc_estimate = min(self.socmax, max(soc_estimate, self.socmin))


        # Package the dictionaries into an output formatted dictionary
        offer_out_dict = {self.rid: {}}
        offer_out_dict[self.rid] = {"block_ch_mc": block_ch_mc, "block_ch_mq": block_ch_mq, "block_dc_mc": block_dc_mc,
                               "block_dc_mq": block_dc_mq, "block_soc_mc": block_soc_mc, "block_soc_mq": block_soc_mq}
        offer_out_dict[self.rid].update(self._default_reserve_offer())
        offer_out_dict[self.rid].update(self._default_dispatch_capacity())
        offer_out_dict[self.rid].update(self._default_offer_constants(soc_begin=soc_estimate, init_en=dispatch_estimate))

        self.formatted_offer = offer_out_dict

    def _descretize_offer_curves(self):
        charge_offer = list(self.binner.collate(self.charge_mq, self.charge_mc))
        discharge_offer = list(self.binner.collate(self.discharge_mq, self.discharge_mc))
        self.charge_mq = charge_offer[0]
        self.charge_mc = charge_offer[1]
        self.discharge_mq = discharge_offer[0]
        self.discharge_mc = discharge_offer[1]

    def _process_efficiency(self, data:list):
        processed_data = []
        for num in data:
            if num < 0:
                processed_data.append(num * self.efficiency)
            else:
                processed_data.append(num)
        return processed_data

    def _real_time_offer(self):
        initial_soc = self.resource["status"][self.rid]["soc"]
        soc_available = initial_soc - self.socmin
        soc_headroom = self.socmax - initial_soc
        block_dc_mc = {}
        block_dc_mq = {}
        block_ch_mc = {}
        block_ch_mq = {}
        block_soc_mc = {}
        block_soc_mq = {}

        t_end = self.market['timestamps'][-1]
        for t_now in self.market['timestamps']:
            en_ledger = {t:order for t,order in self.resource['ledger'][self.rid]['EN'].items() if t >= t_now}
            block_ch_mq[t_now] = []
            block_ch_mc[t_now] = []
            block_dc_mq[t_now] = []
            block_dc_mc[t_now] = []

            # add blocks for cost of current dispatch:
            if t_now in en_ledger.keys():
                for mq, mc in en_ledger[t_now]:
                    if mq < 0:
                        soc_available += mq * self.efficiency
                        soc_headroom -= mq * self.efficiency
                        block_ch_mq[t_now].append(-mq)
                        block_ch_mc[t_now].append(mc)
                    elif mq > 0:
                        soc_available -= mq
                        soc_headroom += mq
                        block_dc_mq[t_now].append(mq)
                        block_dc_mc[t_now].append(mc)

            # add blocks for soc available/headroom
            ledger_list = [tup for sublist in en_ledger.values() for tup in sublist]
            ledger_decreasing = sorted(ledger_list, key=lambda tup:tup[1], reverse=True)
            ledger_increasing = sorted(ledger_list, key=lambda tup:tup[1], reverse=False)
            # use decreasing price ledger for charging cost curve
            remaining_capacity = soc_headroom
            for mq, mc in ledger_decreasing:
                if 0 > mq >= -remaining_capacity:
                    remaining_capacity += mq * self.efficiency # remaining_capacity decreases because m<0
                    block_ch_mq[t_now].append(-mq)
                    block_ch_mc[t_now].append(mc)
                else:
                    remaining_capacity -= remaining_capacity
                    block_ch_mq[t_now].append(remaining_capacity)
                    block_ch_mc[t_now].append(mc)
                    break
            if remaining_capacity:
                block_ch_mq[t_now].append(remaining_capacity)
                block_ch_mc[t_now].append(self.price_floor)
            # use increasing price ledger for discharging cost curve
            remaining_capacity = soc_available
            for mq, mc in ledger_increasing:
                if 0 < mq <= remaining_capacity:
                    remaining_capacity -= mq
                    block_dc_mq[t_now].append(mq)
                    block_dc_mc[t_now].append(mc)
                else:
                    remaining_capacity -= remaining_capacity
                    block_dc_mq[t_now].append(remaining_capacity)
                    block_dc_mc[t_now].append(mc)
                    break
            if remaining_capacity:
                block_dc_mq[t_now].append(remaining_capacity)
                block_dc_mc[t_now].append(self.price_ceiling)

        # valuation of post-horizon SoC
        post_market_ledger = {t: order for t, order in self.resource['ledger'][self.rid]['EN'].items() if t > t_end}
        post_market_list = [tup for sublist in post_market_ledger.values() for tup in sublist]
        post_market_sorted = sorted(post_market_list, key=lambda tup:tup[1], reverse=True)
        block_soc_mq[t_end] = []
        block_soc_mc[t_end] = []
        remaining_capacity = soc_available
        for mq, mc in post_market_sorted:
            if 0 < mq <= remaining_capacity:
                remaining_capacity -= mq
                block_soc_mq[t_end].append(mq)
                block_soc_mc[t_end].append(mc)
            else:
                remaining_capacity -= remaining_capacity
                block_soc_mq[t_end].append(remaining_capacity)
                block_soc_mc[t_end].append(mc)
        if remaining_capacity:
            block_soc_mq[t_end].append(remaining_capacity)
            block_soc_mc[t_end].append(self.price_ceiling)
        block_soc_mq[t_end].append(soc_headroom)
        block_soc_mc[t_end].append(self.price_floor)


        # Package the dictionaries into an output formatted dictionary
        offer_out_dict = {self.rid: {}}
        offer_out_dict[self.rid] = {"block_ch_mc": block_ch_mc, "block_ch_mq": block_ch_mq, "block_dc_mc": block_dc_mc,
                               "block_dc_mq": block_dc_mq, "block_soc_mc": block_soc_mc, "block_soc_mq": block_soc_mq}
        offer_out_dict[self.rid].update(self._default_reserve_offer())
        offer_out_dict[self.rid].update(self._default_dispatch_capacity())
        offer_out_dict[self.rid].update(self._default_offer_constants(bid_soc=True))

        return offer_out_dict

    def _default_reserve_offer(self):
        reg = ['cost_rgu', 'cost_rgd', 'cost_spr', 'cost_nsp']
        res_dict = {}
        for r in reg:
            res_dict[r] = {t: 0 for t in self.market['timestamps']}
        return res_dict

    def _default_dispatch_capacity(self):
        max_dict = {}
        max_dict['chmax'] = {t: self.chmax for t in self.market['timestamps']}
        max_dict['dcmax'] = {t: self.dcmax for t in self.market['timestamps']}
        return max_dict

    def _default_offer_constants(self, **options):
        constants = {}
        constants['soc_begin'] = self.resource['status'][self.rid]['soc']
        constants['init_en'] = self.resource['status'][self.rid]['dispatch']
        constants['init_status'] = 1
        constants['ramp_dn'] = 9999
        constants['ramp_up'] = 9999
        constants['socmax'] = self.socmax
        constants['socmin'] = self.socmin
        constants['eff_ch'] = self.efficiency
        constants['eff_dc'] = 1.0
        constants['soc_end'] = self.socmin
        constants['bid_soc'] = False

        constants.update(options)

        return constants

    def _load_dam_prices_times(self):
        now = self.market['timestamps'][0]
        hour_beginning = now[:10] + '00'
        type = self.market['market_type']
        bus = self.resource['bus']
        if hour_beginning in self.market['previous'][type]['timestamp']:
            prices = self.market['previous'][type]['EN'][bus]
            times = self.market['previous'][type]['timestamp']
        else:
            with open(self._prev_dam_file, "r") as file:
                prices = json.load(file)
                times = [key for key in prices.keys()]
                prices = [value for value in prices.values()]
        return prices, times

    def _save_json(self, save_dict, filename=None):
        # Save as json file in the current directory with name offer_{time_step}.json
        if filename is None:
            filename =f'offer_{self.step}.json'
        with open(filename, 'w') as f:
            json.dump(save_dict, f, indent=4, cls=NpEncoder)

    def _calculate_opportunity_costs(self, prices, charge_mq, discharge_mq):

        # combine the charge/discharge list
        combined_list = [dis - ch for ch, dis in zip(charge_mq, discharge_mq)]
        time = self.market['timestamps']
        schedule = dict(zip(time, combined_list))

        # finding the index for first charge and last discharge
        t1_ch = next((index for index, key in enumerate(schedule) if schedule[key] < 0), None)
        t_last_dis = next((i for i in range(len(combined_list) - 1, -1, -1) if combined_list[i] > 0), None)
        assert isinstance(t1_ch, int), "t1_ch is not an int"
        assert isinstance(t_last_dis, int), "t_last_dis is not an int"

        # create two list for charging/discharging opportunity costs
        charge_list = []
        discharge_list = []

        # opportunity_costs = pd.DataFrame(None, index=range(len(prices)), columns=['Time', 'charge cost', 'disch cost'])
        # soc = pd.DataFrame(None, index=range(len(prices) + 1), columns=['Time', 'SOC'])

        for index, key in enumerate(schedule):
            i = index
            value = schedule[key]

            # charging
            if value < 0:
                oc_ch, oc_dis = self._calc_oc_charge(combined_list, prices, i)
            # discharging
            elif value > 0:
                oc_ch, oc_dis = self._calc_oc_discharge(combined_list, prices, i)
            else:
                # before first charge
                if i < t1_ch:
                    oc_ch, oc_dis = self._calc_oc_before_first_charge(prices, t1_ch, i)
                # after the last discharge
                elif i > t_last_dis:
                    oc_ch, oc_dis = self._calc_oc_after_last_discharge(prices, t_last_dis, i)
                # between cycles
                else:
                    oc_ch, oc_dis = self._calc_oc_between_cycles(combined_list, prices, i)

            # save to list
            charge_list.append(oc_ch)
            discharge_list.append(oc_dis)

        assert sum(abs(c) for c in charge_list) > 0, "calc_oc: charge list has no values"
        assert sum(abs(d) for d in discharge_list) > 0, "calc_oc: discharge list has no values"

        return charge_list, discharge_list

    def _calculate_offer_curve(self, prices):

        # marginal cost comes from opportunity cost calculation
        charge_mq, discharge_mq = self._scheduler(prices)
        charge_mc, discharge_mc = self._calculate_opportunity_costs(prices, charge_mq, discharge_mq)
        # self.charge_mc = oc['charge cost'].values
        # self.discharge_mc = oc['disch cost'].values
        self.charge_mc = charge_mc
        self.discharge_mc = discharge_mc

        # marginal quantities from scheduler values
        self.charge_mq = charge_mq
        self.discharge_mq = discharge_mq

    def _calc_oc_charge(self, combined_list, prices, idx):
        # opportunity cost during scheduled charge
        j = idx + 1 + next((index for index, value in enumerate(combined_list[idx + 1:]) if value > 0), None)

        if j == idx + 1:
            arr2 = 0
        else:
            if j == idx + 2:
                arr2 = prices[idx + 1]
            else:
                arr2 = min(prices[(idx + 1):j])
        if idx == 0:
            oc_ch = min(min(prices[1:j]), self.efficiency * prices[j])
            arr1 = prices[0]
            oc_dis = oc_ch + 0.01
        else:
            oc_ch = min(np.delete(np.array(prices[0:j]), idx).min(), self.efficiency * prices[j])
            arr1 = min(prices[0:idx])
            oc_dis = (-prices[idx] + arr1 + arr2) / self.efficiency

        return oc_ch, oc_dis

    def _calc_oc_discharge(self, combined_list, prices, idx):
        # opportunity cost during scheduled discharge
        j = max((index for index, value in enumerate(combined_list[:idx]) if value < 0), default=None)
        arr1 = 0 if idx == len(prices)-1 else prices[idx + 1] if idx == len(prices) - 2 else max(prices[(idx + 1):])
        arr2 = 0 if j == idx - 1 else prices[j + 1] if j == idx - 2 else max(prices[(j + 1):idx])
        #print("discharge, idx and len(prices)", j, idx, len(prices))
        oc_ch = (-prices[idx] + arr1 + arr2) * self.efficiency
        oc_dis = max(prices[j] / self.efficiency, max(prices[(j + 1):]))

        return oc_ch, oc_dis

    def _calc_oc_before_first_charge(self, prices, t1_idx:int, idx:int):
        # opportunity cost before first charge
        max_ch = 0 if idx == t1_idx - 1 else prices[idx + 1] if idx == t1_idx - 2 else max(prices[(idx + 1):t1_idx])
        oc_ch = max(max_ch * self.efficiency, prices[t1_idx])
        oc_dis = oc_ch + 0.01 if idx == 0 else prices[0] / self.efficiency if idx == 1 else min(prices[0:idx]) / self.efficiency

        return oc_ch, oc_dis

    def _calc_oc_after_last_discharge(self, prices, t_last, idx):
        # opportunity cost after last discharge
        oc_ch = max(prices[(idx + 1):]) * self.efficiency if idx < len(prices) - 2 else prices[idx + 1] if idx == len(
            prices) - 2 else min(prices)
        arr = prices[idx - 1] if idx == t_last + 2 else min(prices[(t_last + 1):idx]) if idx > t_last + 2 else max(prices)
        oc_dis = min(prices[t_last], arr / self.efficiency)

        return oc_ch, oc_dis

    def _calc_oc_between_cycles(self, combined_list, prices, idx):
        j_next = idx + 1 + next((index for index, value in enumerate(combined_list[idx + 1:]) if value > 0),None)
        j_prev = max((index for index, value in enumerate(combined_list[:idx]) if value < 0), default=None)
        oc_ch = 0 if idx < j_prev + 2 else prices[idx - 1] if idx == j_prev + 2 else max(
            max(prices[(j_prev + 1):idx]) * self.efficiency, prices[j_prev])
        oc_dis = 0 if idx > j_next - 2 else min(prices[j_next],
                                              prices[idx + 1] / self.efficiency) if idx == j_next - 2 else min(
            prices[j_next], min(prices[(idx + 1):j_next]) / self.efficiency)

        return oc_ch, oc_dis

    def _scheduler(self, prices):

        number_step =len(prices)
        # [START solver]
        # Create the linear solver with the GLOP backend.
        solver = pywraplp.Solver.CreateSolver("GLOP")
        if not solver:
            raise ImportError("Scheduler unable to load solver.")
        # [END solver]

        #Variables: all are continous
        charge = [solver.NumVar(0.0, self.chmax, "c"+str(i)) for i in range(number_step)]
        discharge = [solver.NumVar(0, self.dcmax,  "d"+str(i)) for i in range(number_step)]
        dasoc = [solver.NumVar(0.0, self.socmax, "b"+str(i)) for i in range(number_step+1)]
        dasoc[0]=0

        #Objective function
        solver.Minimize(
            sum(prices[i]*(charge[i]-discharge[i]) for i in range(number_step)))
        for i in range(number_step):
            solver.Add(dasoc[i] + self.efficiency*charge[i] - discharge[i] == dasoc[i+1])
        solver.Solve()
        #print("Solution:")
        #print("The Storage's profit =", solver.Objective().Value())
        charge_list = []
        discharge_list = []
        dasoc_list=[]
        for i in range(number_step):
            charge_list.append(charge[i].solution_value())
            discharge_list.append(discharge[i].solution_value())
            #dasoc_list.append(dasoc[i].solution_value())

        assert sum(abs(c) for c in charge_list) > 0, "scheduler: charge list has no values"
        assert sum(abs(d) for d in discharge_list) > 0, "scheduler: discharge list has no values"

        return charge_list, discharge_list


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('time_step', type=int, help='Integer time step tracking the progress of the\
                        simulated market.')
    parser.add_argument('market_file', help='json formatted dictionary with market information.')
    parser.add_argument('resource_file', help='json formatted dictionary with resource information.')

    args = parser.parse_args()

    # Parse inputs
    time_step = args.time_step
    with open(args.market_file, 'r') as f:
        market_info = json.load(f)
    with open(args.resource_file, 'r') as f:
        resource_info = json.load(f)