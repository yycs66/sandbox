# This is a test dummy algorithm to get the opportunity cost curves
from ortools.linear_solver import pywraplp
import pandas as pd
import numpy as np
import argparse
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

#calculate the opportunity cost for charge/discharge
def cost_curves(prices):
    # battery parameters
    capacity=18
    ch_limit=20
    dis_limit=20
    effcy=0.9
    
    def scheduler(prices):
    
        number_step =len(prices)
        # [START solver]
        # Create the linear solver with the GLOP backend.
        solver = pywraplp.Solver.CreateSolver("GLOP")
        if not solver:
            return
        # [END solver]

    #Variables: all are continous
        charge = [solver.NumVar(0.0, ch_limit, "c"+str(i)) for i in range(number_step)]
        discharge = [solver.NumVar(0, dis_limit,  "d"+str(i)) for i in range(number_step)]
        soc = [solver.NumVar(0.0, capacity, "b"+str(i)) for i in range(number_step+1)]
        soc[0]=0

    #Objective function
        solver.Minimize(
            sum(prices[i]*(charge[i]-discharge[i]) for i in range(number_step)))
        for i in range(number_step):
            solver.Add(soc[i] +effcy*charge[i] -discharge[i]==soc[i+1])
    # create the constaints on soc
        solver.Solve()
        #print("Solution:")
        #print("The Storage's profit =", solver.Objective().Value())
        charge_list=[]
        discharge_list=[]
        for i in range(number_step):
            charge_list.append(charge[i].solution_value())
            discharge_list.append(discharge[i].solution_value())
        return charge_list,discharge_list
        
    [charge,discharge]=scheduler(prices)
    #find the index of time period for charging/discharging
    index_ch=np.asarray(np.where(np.array(charge)>0)).flatten()
    ch_index =np.where(np.array(charge)>0)[0]
    index_dis=np.asarray(np.where(np.array(discharge)>0)).flatten()
    dis_index =np.where(np.array(discharge)>0)[0]

    idx0 =np.append(dis_index,ch_index)
    idx = np.sort(idx0)

    # create two list for charging/discharging opportunity costs
    oc_dis_list=[]
    oc_ch_list=[]
    
    offer = pd.DataFrame(None,index=range(len(prices)), columns=['Time','charge cost','disch cost'])

#offer =offer.astype('Float64')
     
    for index, row in offer.iterrows():
        i =index
        row['Time'] =index
        if i in ch_index:
            # for hours in charge
            indx =np.where(ch_index==i)[0]
            j =indx.item()
            if i==0:
                oc_ch = min(prices[1:index_dis[j]], effcy*prices[index_dis[j]])
                oc_ch_list.append(oc_ch)  
                oc_dis = oc_ch+0.01
                print("oc_ch type is", type(oc_ch))
                row['charge cost'] =oc_ch
                row['disch cost'] =oc_dis
                oc_dis_list.append(oc_dis)  
            else:
                #oc_ch updates
                arr =np.delete(prices[0:index_dis[j]], index_ch[j])
                min_oc_temp = arr.min()
                oc_ch = min(min_oc_temp, effcy*prices[index_dis[j]])
                oc_ch_list.append(oc_ch)
                #oc_dis updates
                arr1 =prices[0:index_ch[j]].min()
                arr2 =prices[(index_ch[j]+1):index_dis[j]].min()
                oc_dis =(-prices[index_ch[j]]+arr1+arr2)/effcy
                oc_dis_list.append(oc_dis)
                row['charge cost'] =oc_ch
                row['disch cost'] =oc_dis
        elif i in dis_index:
                # for scheduled discharge
                indx =np.where(dis_index==i)[0]
                j =indx.item()
                arr1 = prices[(index_ch[j]+1):index_dis[j]].max()
                arr2 = prices[(index_dis[j]+1):24].max()
                oc_ch =(-prices[index_dis[j]] +arr1 +arr2)*effcy
                oc_ch_list.append(oc_ch)
                oc_dis = max(prices[index_ch[j]]/effcy, prices[(index_dis[j]+1):24].max())
                oc_dis_list.append(oc_dis)
                row['charge cost'] =oc_ch
                row['disch cost'] =oc_dis
        elif i< ch_index[0]:
                # opportunity cost for charging
                max_ch_temp =prices[(i+1):ch_index[0]].max()
                oc_ch =max(max_ch_temp*effcy, prices[index_ch[0]])
                oc_ch_list.append(oc_ch)
                row['charge cost'] =oc_ch
                
                #opportunity cosy for discharging
                if i==0:
                    # Hour 0, the oc_dis = to oc_ch+0.01
                    oc_dis = oc_ch+0.01
                    oc_dis_list.append(oc_dis)
                    row['disch cost'] =oc_dis
                elif i==1:  
                    oc_dis =prices[0]/effcy
                    oc_dis_list.append(oc_dis)
                    row['disch cost'] =oc_dis
                elif i>1 :
                    oc_dis= prices[0:i].min()/effcy
                    oc_dis_list.append(oc_dis)  
                    row['disch cost'] =oc_dis
        elif i>dis_index[-1]:
            
                if i != len(prices)-1:
                    oc_ch =prices[(i+1):len(prices)].max()*effcy
                    oc_ch_list.append(oc_ch) 
                    oc_dis = min(prices[index_ch[-1]], prices[(index_dis[-1]+1):len(prices)].min()/effcy)
                    oc_dis_list.append(oc_dis)
                    row['charge cost'] =oc_ch
                    row['disch cost'] =oc_dis  
                else:
                    oc_ch=0
                    oc_ch_list.append(oc_ch) 
                    oc_dis =max(prices[index_ch[-1]]/effcy, prices[i])
                    oc_dis_list.append(oc_dis) 
                    row['charge cost'] =oc_ch
                    row['disch cost'] =oc_dis
        
            # for hours between
            #print("orignial i is ", i)
        elif i>index_dis[0] and i< index_ch[1]:
                # hours between last discharge and next charge period
                max_ch_temp =prices[i:ch_index[1]].max()
                oc_ch =max(max_ch_temp*effcy, prices[index_ch[1]])
                oc_ch_list.append(oc_ch)
                row['charge cost'] =oc_ch
                oc_dis= prices[0:i].min()/effcy
                oc_dis_list.append(oc_dis)  
                row['disch cost'] =oc_dis
        else:
            for k in range(len(ch_index)):
                if i== index_ch[k]+1:
                    #print("i1 is between", i)
                    oc_ch = max(prices[index_ch[k]], prices[i]*effcy)
                    oc_ch_list.append(oc_ch)
                    row['charge cost'] =oc_ch
                    if i!= index_dis[k]-1:
                        oc_dis =min(prices[index_dis[k]], prices[(i+1):index_dis[k]].min()/effcy)
                        oc_dis_list.append(oc_dis)
                        row['disch cost'] =oc_dis
                    else:
                        oc_dis = prices[index_dis[k]]
                        oc_dis_list.append(oc_dis)
                        row['disch cost'] =oc_dis
                elif i>index_ch[k]+1 and i< index_dis[k]:
                    max_ch_temp = prices[(index_ch[k]+1):i]
                    max_ch_temp =max_ch_temp.max()
                    oc_ch = max(max_ch_temp*effcy, prices[index_ch[k]])
                    oc_ch_list.append(oc_ch)
                    row['charge cost'] =oc_ch
                    if i< index_dis[k]-2:
                        oc_dis = min(prices[index_dis[k]], prices[(i+1):index_dis[k]].min()/effcy)
                        oc_dis_list.append(oc_dis)
                        row['disch cost'] =oc_dis
                    elif i== index_dis[k]-2:
                        oc_dis = min(prices[index_dis[k]], prices[i+1]/effcy)
                        oc_dis_list.append(oc_dis)
                        row['disch cost'] =oc_dis
                    elif i== index_dis[k]-1:
                        oc_dis = min(prices[index_dis[k]],prices[index_dis[k]]/effcy)
                        oc_dis_list.append(oc_dis)
                        row['disch cost'] =oc_dis                 
    quantity = pd.DataFrame(charge)
    #quantity = pd.DataFrame(charge)
    #ch_cost =pd.DataFrame(oc_ch_list,columns=['cost'])
    offer_ch =pd.concat([offer['Time'], offer['charge cost'], quantity], axis=1,ignore_index=True)
    offer_ch.columns =['Time','COST','MW']

    quan2 =pd.DataFrame(discharge)
    #dis_cost = pd.DataFrame(oc_dis_list)
    offer_dis=pd.concat([offer['Time'], offer['disch cost'], quantity], axis=1,ignore_index=True)
    offer_dis.columns= ['Time','COST','MW']
          

    return offer_ch, offer_dis

if __name__ == '__main__':
    # Add argument parser for three required input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('time_step', type=int, help='Integer time step tracking the progress of the\
                        simulated market.')
    parser.add_argument('market_info', help='json formatted dictionary with market information.')
    parser.add_argument('resource_info', help='json formatted dictionary with resource information.')

    args = parser.parse_args()
    
    # Parse json inputs into python dictionaries
    time_step = args.time_step
    market_info = json.loads(args.market_info)
    resource_info = json.loads(args.resource_info)
    
    # Read in information from the market
    prices = market_info["prev_cleared"]["lw_lmp"]
    required_times = [t for t in market_info['intervals'].keys()]
    
    # A set of prices included for reference (the same as those sent in market_info)
    # prices = [11.589574176737466, 14.818897513178682, 19.999999999999993, 20.000000000000046, 20.0, 20.0, 20.000000000000007, 19.999999999999993, 10.592726268568267, 2.168250942593657, 10.008099093287708, 10.425226940104036, 10.420123233959057, 10.41519610348975, 10.403200027853545, 10.432381551298858, 21.056725586230904, 200.00000000000006, 200.00000000000003, 200.00000000000006, 200.00000000000006, 199.99999999999997, 200.00000000000006, 200.0, 200.00000000000003, 200.00000000000006, 200.0, 199.99999999999994, 200.00000000000003, 200.0, 200.00000000000003, 200.00000000000003, 20.575591538122225, 18.540571670322418, 18.735067390805465, 18.829070031890733]
    prices = np.array(prices)
    
    # Make the offer curves and unload into arrays
    offer_ch, offer_dis = cost_curves(prices)
    charge_mc = offer_ch['COST'].values
    charge_mq = offer_ch['MW'].values
    discharge_mc = offer_dis['COST'].values
    discharge_mq = offer_dis['MW'].values
    
    # Offer parsing script below:
    
    # Convert the offer curves to timestamp:offer_value dictionaries
    block_ch_mc = {}
    for i, cost in enumerate(charge_mc):
        block_ch_mc[required_times[i]] = float(cost)
        
    block_ch_mq = {}
    for i, power in enumerate(charge_mq):
        block_ch_mq[required_times[i]] = float(power) # 125MW
        
    block_dc_mc = {}
    block_soc_mc = {}
    for i, cost in enumerate(discharge_mc):
        block_dc_mc[required_times[i]] = float(cost)
        block_soc_mc[required_times[i]] = 0
        
    block_dc_mq = {}
    block_soc_mq = {}
    for i, power in enumerate(discharge_mq):
        block_dc_mq[required_times[i]] = float(power) # 125MW
        block_soc_mq[required_times[i]] = 0
        
    reg = ['cost_rgu', 'cost_rgd', 'cost_spr', 'cost_nsp']
    zero_arr = np.zeros(len(required_times))
    rgu_dict = {}
    for r in reg:
        rgu_dict[r] = {}
        for t in required_times:
            rgu_dict[r][t] = 0
    
    max_dict = {}
    for mx in ['chmax', 'dcmax']:
        max_dict[mx] = {}
        for t in required_times:
            max_dict[mx][t] = 125
    
    constants = {}
    constants['soc_begin'] = 133.33
    constants['init_en'] = 0
    constants['init_status'] = 0
    constants['ramp_dn'] = 9999
    constants['ramp_up'] = 9999
    constants['socmax'] = 633.33
    constants['socmin'] = 133.33
    constants['eff_ch'] = 0.892
    constants['eff_dc'] = 1.0
    constants['soc_end'] = 133.33
    constants['bid_soc'] = False
        
    # Pacakge the dictionaries into an output formatted dictionary
    rid = 'r000001'
    offer_out_dict = {rid:{}}
    offer_out_dict[rid] = {"block_ch_mc":block_ch_mc, "block_ch_mq":block_ch_mq, "block_dc_mc":block_dc_mc, "block_dc_mq":block_dc_mq, "block_soc_mc":block_soc_mc, "block_soc_mq":block_soc_mq}
    offer_out_dict[rid].update(rgu_dict)
    offer_out_dict[rid].update(max_dict)
    offer_out_dict[rid].update(constants)
    
    # Save as json file in the current directory with name offer_{time_step}.json
    with open(f'offer_{time_step}.json', 'w') as f:
        json.dump(offer_out_dict, f, indent=4, cls=NpEncoder)
    

    