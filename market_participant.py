import sys
import offer_utils as ou
import dummy_algorithm_Br as da
import json
import argparse
import numpy as np
import pandas as pd
import random
import csv
import os
import torch
import torch.nn as nn

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
    def __init__(self, time_step, market_info, resource_info):
        self.step = time_step
        self.market = market_info
        self.resource = resource_info
        self.market_type = market_info['market_type']
        self.rid = resource_info['rid']

    def scaling(self, da, scaling_factor):
        # Parse inputs
        dummy_agent = da.Agent(time_step, market_info, resource_info)
        dummy_offer = dummy_agent.make_me_an_offer()
        timestamp = list(dummy_offer[self.rid]['block_ch_mc'].keys())
        #print("lenth of timestamp is: ", len(timestamp))  
        #print("lenth of scaling factor is: ", len(scaling_factor))
        #factor_action = np.hstack((timestamp, scaling_factor))
        factor_action_dict = dict(zip(timestamp, scaling_factor))
        if 'DAM' in self.market_type:
                for timestamp, value in dummy_offer[self.rid]['block_ch_mc'].items():
                    if isinstance(value, (int, float, np.int64, np.float64)):
                        dummy_offer[self.rid]['block_ch_mc'][timestamp] = value * factor_action_dict[timestamp]
                    elif isinstance(value, (list, tuple, np.ndarray)):
                        dummy_offer[self.rid]['block_ch_mc'][timestamp] = [v * factor_action_dict[timestamp] for v in value]
                for timestamp, value in dummy_offer[self.rid]['block_dc_mc'].items():
                    if isinstance(value, (int, float, np.int64, np.float64)):
                        dummy_offer[self.rid]['block_dc_mc'][timestamp] = value * factor_action_dict[timestamp]
                    elif isinstance(value, (list, tuple, np.ndarray)):
                        dummy_offer[self.rid]['block_dc_mc'][timestamp] = [v * factor_action_dict[timestamp] for  v in value]
                with open(f'offer_{self.step}.json', 'w') as f: 
                    json.dump(dummy_offer, f, cls=NpEncoder)
        elif 'RTM' in self.market_type:
            for timestamp, value in dummy_offer[self.rid]['block_soc_mc'].items():
                if isinstance(value, (list, tuple, np.ndarray)):
                    scaled_value = [v * factor_action_dict[timestamp] for v in value]
                    dummy_offer[self.rid]['block_soc_mc'][timestamp] = scaled_value
                elif isinstance(value, (int, float, np.int64, np.float64)):
                    dummy_offer[self.rid]['block_soc_mc'][timestamp] = value * factor_action_dict[timestamp]
            with open(f'offer_{self.step}.json', 'w') as f: 
                json.dump(dummy_offer, f, cls=NpEncoder)
        return dummy_offer #updated offer from scaling factor

# Define Actor Network
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, action_bound, hidden_dim=(256, 128)):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim[0])
        self.fc2 = nn.Linear(hidden_dim[0], hidden_dim[1])
        self.fc3 = nn.Linear(hidden_dim[1], action_dim)
        self.action_bound = torch.tensor(action_bound)
        
    def forward(self, state):
        x = state.float()
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x)) * self.action_bound
        return x

# Define DDPG Agent
class DDPGAgent:
    def __init__(self, state_dim, action_dim, action_bound, hidden_dim=(256, 128)):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bound = action_bound
        self.actor = Actor(state_dim, action_dim, action_bound, hidden_dim)
        self.actor.load_state_dict(torch.load('actor_model.pth'))
        self.actor.eval()
    
    def choose_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        action = self.actor(state).detach().cpu().numpy().squeeze(0)
        action = np.clip(action, 0, self.action_bound)
        return action

# Define the Energy Environment
class EnergyEnvironment:
    def __init__(self, episode):
        self.episode = episode
        self.current_step = 0
        self.max_steps = None
        self.load_data()
        
    def load_data(self):
        market_filename = f'market_{self.episode}.json'
        with open(market_filename, 'r') as file:
            market_data = json.load(file)
        
        self.price_forecast = pd.DataFrame.from_dict(market_data['price_forecast'])
        self.solar_data = pd.DataFrame.from_dict(market_data['solar_forecast'])
        self.wind_data = pd.DataFrame.from_dict(market_data['wind_forecast'])
        self.load_data = pd.DataFrame.from_dict(market_data['load_forecast'])
        
        self.max_steps = self.price_forecast.shape[0]
        
    def reset(self):
        self.current_step = 0
        state = self.get_state()
        return state
    
    def step(self, action):
        done = self.current_step + 1 >= self.max_steps
        
        if not done:
            self.current_step += 1
            next_state = self.get_state()
            reward = self.get_reward()
        else:
            next_state = np.zeros_like(state)
            reward = 0
        
        return next_state, reward, done
    
    def get_state(self):
        resource_filename = f'resource_{self.current_step}.json'
        with open(resource_filename, 'r') as file:
            resource_data = json.load(file)
        
        soc = resource_data['soc']
        
        state = [self.price_forecast.iloc[self.current_step, 0],
                 self.solar_data.iloc[self.current_step, 0],
                 self.wind_data.iloc[self.current_step, 0],
                 self.load_data.iloc[self.current_step, 0],
                 soc]
        
        state = pd.to_numeric(state)
        return state
    
    def get_reward(self):
        resource_filename = f'resource_{self.current_step}.json'
        with open(resource_filename, 'r') as file:
            resource_data = json.load(file)
        
        reward = resource_data['score']['net_revenue']
        return reward

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
    with open(f'market_{time_step-1}.json', 'w') as f:
        json.dump(market_info, f, cls=NpEncoder) 
    with open(args.resource_file, 'r') as f:
        resource_info = json.load(f)
    with open(f'resource_{time_step-1}.json', 'w') as f:
        json.dump(resource_info, f, cls=NpEncoder)

    # Define state and action dimensions
    state_dim = 5
    action_dim = 36
    action_bound = 3.0

    # Initialize DDPG agent with the best hyperparameters
    hidden_dim = (256, 128)  
    agent = DDPGAgent(state_dim=state_dim, action_dim=action_dim, action_bound=action_bound, hidden_dim=hidden_dim)

    # Initialize variables
    num_episodes = 289
    total_reward = 0

    # Testing loop
    for episode in range(num_episodes):
        env = EnergyEnvironment(episode)
        state = env.reset()
        done = False
        episode_reward = 0
        actions_taken = []

        while not done:
            if episode == 0:
                factors = np.random.uniform(0.1, 3, 36)
            else:
                action = agent.choose_action(state)
                factors = action

            scaled_agent = Scaled_agent(episode, market_info, resource_info)
            scaled_agent.scaling(da, factors)

            next_state, reward, done = env.step(factors)
            episode_reward += reward
            state = next_state

        total_reward += episode_reward
        print(f"Episode {episode}: Total Reward = {episode_reward}")

        # Save actions taken in the episode
        action_filename = f'action_{episode}.json'
        action_data = {'actions': factors.tolist()}
        with open(action_filename, 'w') as file:
            json.dump(action_data, file)

    print(f"Total Reward over {num_episodes} episodes: {total_reward}")