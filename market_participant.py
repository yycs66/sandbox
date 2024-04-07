'''
Scaled offer generation
This code will take the same inputs and provide the same outputs as a competitor algorithm.
'''

import sys
import offer_utils as ou
import dummy_algorithm_Br as da
import json
import argparse
import numpy as np
import random
import csv
import os

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
    
class Scaled_agent():
    def __init__(self,time_step,market_info,resource_info):
        self.step = time_step
        self.market = market_info
        self.resource = resource_info
        self.market_type = market_info['market_type']
        self.rid = resource_info['rid']
        #self.da = da.Agent(time_step, market_info, resource_info)
    def scaling(self, da,scaling_factor):
        '''
        This function is to generate a scaled_cost_offer based on the dummy_algorithm_Br.py
        Args:
            da: the dummy_algorithm_Br object
            market_type: the market type
            scaling_factor: the scaling factor
        Returns:
            scaled_cost_offer: the scaled cost offer
        '''
        # Parse inputs

        dummy_agent = da.Agent(time_step, market_info, resource_info)
        dummy_offer = dummy_agent.make_me_an_offer()
        timestamp = list(dummy_offer[self.rid]['block_ch_mc'].keys())
        #print("lenth of timestamp is: ", len(timestamp))  
        #print("lenth of scaling factor is: ", len(scaling_factor))
        factor_action = np.hstack((timestamp, scaling_factor))
        print("factor_action is: ", factor_action)
        factor_action_dict = dict(zip(timestamp, factor_action))
        if 'DAM' in self.market_type:
                for timestamp, value in dummy_offer[self.rid]['block_ch_mc'].items():
                    if isinstance(value, (int, float, np.int64, np.float64)):
                        dummy_offer[self.rid]['block_ch_mc'][timestamp] = value * factor_action[timestamp]
                    elif isinstance(value, (list, tuple, np.ndarray)):
                        dummy_offer[self.rid]['block_ch_mc'][timestamp] = [v * factor_action[timestamp] for v in value]
                for timestamp, value in dummy_offer[self.rid]['block_dc_mc'].items():
                    if isinstance(value, (int, float, np.int64, np.float64)):
                        dummy_offer[self.rid]['block_dc_mc'][timestamp] = value * factor_action[timestamp]
                    elif isinstance(value, (list, tuple, np.ndarray)):
                        dummy_offer[self.rid]['block_dc_mc'][timestamp] = [v * factor_action[timestamp] for  v in value]
                with open(f'offer_{self.step}.json', 'w') as f: 
                    json.dump(dummy_offer, f, cls=NpEncoder)
        elif 'RTM' in self.market_type:
            for timestamp, value in dummy_offer[self.rid]['block_soc_mc'].items():
                if isinstance(value, (list, tuple, np.ndarray)):
                    scaled_value = [v * scaling_factor for v in value]
                    dummy_offer[self.rid]['block_soc_mc'][timestamp] = scaled_value
                elif isinstance(value, (int, float, np.int64, np.float64)):
                    dummy_offer[self.rid]['block_soc_mc'][timestamp] = value * scaling_factor
            with open(f'offer_{self.step}.json', 'w') as f: 
                json.dump(dummy_offer, f, cls=NpEncoder)

        return dummy_offer #updated offer from scaling factor

if __name__ == "__main__":

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
    """ with open(f'market_0.json', 'w') as f:
        json.dump(market_info, f, cls=NpEncoder) """
    with open(args.resource_file, 'r') as f:
        resource_info = json.load(f)
    """ with open(f'resource_0.json', 'w') as f:
        json.dump(resource_info, f, cls=NpEncoder) """

    #factor = random.uniform(0.1, 3)
    factors = np.random.normal(loc=1, scale=0.5, size=36)
    output_file = 'time_step_factor.csv'

    # Write the header row if the file doesn't exist
    if not os.path.exists(output_file):
        with open(output_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            #writer
            for time_step, factor in enumerate(factors):
                writer.writerow([time_step, factor])

        # Write the (time_step, factor) pairs
    with open(output_file, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        #writer.writerow([time_step, factor])
        for time_step, factor in enumerate(factors):
            writer.writerow([time_step, factor])
    
    scaled_agent = Scaled_agent(time_step,market_info,resource_info)
    scaled_agent.scaling(da,factor)

    # Write the updated market and resource information to file
    with open(args.market_file, 'w') as f:
        json.dump(market_info, f, cls=NpEncoder)
    with open(f'market_{time_step}.json', 'w') as f:
        json.dump(market_info, f, cls=NpEncoder)
    with open(args.resource_file, 'w') as f:
        json.dump(resource_info, f, cls=NpEncoder)
    with open(f'resource_{time_step}.json', 'w') as f:
        json.dump(resource_info, f, cls=NpEncoder)
    

    
