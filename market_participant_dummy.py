'''
Simulated competitor code for the ESPA-Comp.
This code will take the same inputs and provide the same outputs as a competitor algorithm.
'''

import sys
import offer_utils as ou
import dummy_algorithm_Br as da
import json
import argparse
import numpy as np

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
    with open(args.resource_file, 'r') as f:
        resource_info = json.load(f)

    agent = da.Agent(time_step, market_info, resource_info)
    agent.make_me_an_offer()
