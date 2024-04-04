'''
Utility functions to support the make_offer.py code
'''

import numpy as np
import os
import json

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def load_json(filename, filedir='./'):
    '''Find the status of your resources and of the next market configuration'''
    with open(os.path.join(filedir,f'{filename}.json'), 'r') as f:
        json_dict = json.load(f)
    return json_dict

def split_mktid(mktid):
    """Splits the market_id string into the market type and the time string (YYYYmmddHHMM)"""

    split_idx = [i for i, char in enumerate(mktid) if char == '2'][0]
    mkt_type = mktid[:split_idx]
    start_time = mktid[split_idx:]
    return mkt_type, start_time

def compute_offers(resources, times, demand, renewables):
    """Takes the status and forecast and makes an offer dictionary"""
    # We will loop through keys to fill our offer out, making updates as needed
    status = resources['status']
    klist = list(status.keys())
    my_soc = status[klist[0]]['soc']
    offer_keys = ['cost_rgu', 'cost_rgd', 'cost_spr', 'cost_nsp', 'block_ch_mc', 'block_dc_mc',
                  'block_soc_mc', 'block_ch_mq', 'block_dc_mq', 'block_soc_mq', 'soc_end',
                  'bid_soc', 'init_en', 'init_status', 'ramp_up', 'ramp_dn', 'socmax', 'socmin',
                  'soc_begin', 'eff_ch', 'eff_dc', 'chmax', 'dcmax']
    offer_vals = [3, 3, 0, 0, 0, 0, [-20, -10, 0, 10], 125, 125, [250, 50, 208, 100], 128,
                  False, 0, 0, 9999, 9999, 608, 128, my_soc, 0.9, 1, 125, 125]
    use_time = [True, True, True, True, True, True, True, True, True, True, False, False, False,
                False, False, False, False, False, False, False, False, True, True]
    offer_out = {}
    for rid in status.keys():
        resource_offer = {}
        for i, key in enumerate(offer_keys):
            if use_time[i]:
                time_dict = {}
                for t in times:
                    time_dict[t] = offer_vals[i]
            else:
                time_dict = offer_vals[i]
            resource_offer[key] = time_dict
        offer_out[rid] = resource_offer
    return offer_out

def save_offer(offer, time_step):
    """Saves the offer in json format to the correct resource directory"""
    # Write the data dictionary to a JSON file
    if time_step != 4:
        json_file = f'offer_{time_step}.json'
        with open(json_file, "w") as f:
            json.dump(offer, f, cls=NpEncoder, indent=4)

class Binner():
    '''
    Binner class is used to reduce bid and offer curves into a specified number of segments
    Attributes:
        n (int): The maximum number of segments (bins) to create.
        qmin (float): The minimum allowed quantity for each bin.
        reverse (bool): Determines if the binning order should be reversed.
        output_type (list or tuple): Specifies the output format of the collate method.
            Accepted values: 'lists' (default) or 'tuples'.
    Methods:
    - collate()
    '''

    def __init__(self, n=10, qmin=0.1, output_type='lists'):
        self._n = n
        self._qmin = qmin
        self._output_type = output_type

    @property
    def n(self):
        return self._n

    @n.setter
    def n(self, value):
        if not isinstance(value, int) or value < 1:
            raise ValueError("n must be a positive integer")
        self._n = value

    @property
    def qmin(self):
        return self._qmin

    @qmin.setter
    def qmin(self, value):
        if not isinstance(value, (int, float)) or value <= 0:
            raise ValueError("qmin must be a positive number")
        self._qmin = value

    @property
    def output_type(self):
        return self._output_type

    @output_type.setter
    def output_type(self, value):
        if value not in ['lists', 'tuples']:
            raise ValueError("output_type must be either 'lists' or 'tuples'")
        self._output_type = value

    def collate(self, *args, n: int = None, qmin: float = None, reverse: bool = False):

        if len(args) == 1 and all(isinstance(arg, tuple) for arg in args[0]):
            tup_list = args[0]
        elif len(args) == 2 and all(isinstance(arg, list) for arg in args):
            input1, input2 = args
            if len(input1) != len(input2):
                raise ValueError("Input lists must have same length")
            tup_list = list(zip(input1, input2))
        else:
            raise TypeError("Input must be a list of tuples or two lists")

        return self._collate(tup_list, n, qmin, reverse)

    def _collate(self, tup_list, n=None, qmin=None, reverse=False):

        if n is None:
            n = self.n
        if qmin is None:
            qmin = self.qmin

        sorted_tups = sorted(tup_list, key=lambda x: x[0], reverse=reverse)
        collated_list = []
        total_qty = sum(x[1] for x in sorted_tups)
        remaining_qty = total_qty

        for i, (qty, price) in enumerate(sorted_tups):
            if len(collated_list) == n - 1:
                combined_quantity = sum(x[0] for x in collated_list) + qty
                combined_price = remaining_qty / combined_quantity
                collated_list.append((combined_quantity, combined_price))
                break

            # Find the largest bin size that satisfies the min_bin constraint
            max_bin_size = remaining_qty / (n - len(collated_list))
            next_bin_start = i + 1
            while next_bin_start < len(sorted_tups) and sum(
                    x[0] for x in sorted_tups[i:next_bin_start]) <= max_bin_size:
                next_bin_start += 1

            # Calculate the bin size and price
            bin_size = sum(x[0] for x in sorted_tups[i:next_bin_start])
            
            if bin_size != 0:
                bin_price = sum(x[0] * x[1] for x in sorted_tups[i:next_bin_start]) / bin_size
            else:
                bin_price = 0

            # Ensure the bin size is greater than or equal to min_bin
            if bin_size >= qmin:
                collated_list.append((bin_size, bin_price))
                remaining_qty -= sum(x[1] for x in sorted_tups[i:next_bin_start])

        if self.output_type == 'lists':
            collate_out = zip(*collated_list)
        elif self.output_type == 'tuples':
            collate_out = collated_list
        else:
            raise ValueError(f"output_type {self.output_type} not supported.")

        return collate_out
