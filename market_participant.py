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
        self.bus = resource_info['bus']

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
    def __init__(self, state_dim, action_dim, action_bound, hidden_dim):
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
    def __init__(self, state_dim, action_dim, action_bound,hidden_dim, buffer_size, 
                 learning_rate,batch_size, gamma, tau):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bound = action_bound
        self.batch_size = batch_size
        self.gamma = gamma
        self.hidden_dim = hidden_dim
        self.lr = learning_rate  
        self.tau = tau
        self.buffersize = buffer_size
        self.actor = Actor(state_dim, action_dim, action_bound, hidden_dim)
        self.actor.load_state_dict(torch.load('actor_model.pth'))
        self.actor.eval()
    def choose_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        action = self.actor(state).detach().cpu().numpy().squeeze(0)
        #action = np.clip(action, 0, self.action_bound)
        if np.isnan(action).any() or np.isinf(action).any():
            # Find the nearest action from the training trajectory
            action = np.nan_to_num(action, nan=1.0)

        action = np.clip(action, 0, self.action_bound)
        return action
    

""" a new DDPGAgent which can do train-tuning
class DDPGAgent:
    def __init__(self, state_dim, action_dim, action_bound, hidden_dim=(256, 128), mode='train'):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bound = action_bound
        self.actor = Actor(state_dim, action_dim, action_bound, hidden_dim)
        
        if mode == 'fine-tune':
            self.actor.load_state_dict(torch.load('actor_model.pth'))
        
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters())
        self.mode = mode

    def train(self, state, action):
        state = torch.FloatTensor(state).unsqueeze(0)
        action = torch.FloatTensor(action).unsqueeze(0)
        
        actor_output = self.actor(state)
        actor_loss = torch.mean((actor_output - action) ** 2)
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

    def choose_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        action = self.actor(state).detach().cpu().numpy().squeeze(0)
        action = np.clip(action, 0, self.action_bound)
        return action

    def save_model(self, filename):
        torch.save(self.actor.state_dict(), filename) """
# Define the Energy Environment
class EnergyEnvironment:
    def __init__(self, episode):
        self.episode = episode
        self.current_step = 0
        self.max_steps = None
        self.loaddata()
        
    def loaddata(self):
        market_filename = f'market_{self.episode-1}.json'
        with open(market_filename, 'r') as file:
            market_data = json.load(file)
        self.market_type = market_data['market_type']
        resource_filename = f'resource_{self.episode-1}.json'
        with open(resource_filename, 'r') as file:
            resource_data = json.load(file)
        self.rid = resource_info['rid']
        self.bus = resource_info['bus']
        
        if 'EN' in market_data['previous'][self.market_type]['prices']:
            self.price_forecast = pd.DataFrame(market_data['previous'][self.market_type]['prices']['EN'][self.bus])
        else:
            self.price_forecast =  pd.DataFrame([0] * 36)
        
        if 'solar' in market_data['forecast']:
            self.solar_data = pd.DataFrame(market_data['forecast']['solar'])
        else:
            self.solar_data = pd.DataFrame([0] * 36)  # Default values of all zeros with 36 elements
        
        if 'wind' in market_data['forecast']:
            self.wind_data = pd.DataFrame(market_data['forecast']['wind'])
        else:
            self.wind_data = pd.DataFrame([0] * 36)  # Default values of all zeros with 36 elements
        
        if 'load' in market_data['forecast']:
            self.load_data = pd.DataFrame(market_data['forecast']['load'])
        else:
            self.load_data = pd.DataFrame([0] * 36) 
    
        self.max_steps = self.price_forecast.shape[0]
        
    def reset(self):
        self.current_step = 0
        state = self.get_state()
        return state
    
    def step(self, action):
        # Check if the episode is done
        done = self.current_step + 1 >= self.max_steps
        
        if not done:
            self.current_step += 1
            if action is None:
                state = self.get_state()    
                action = agent.choose_action(state)
            
            # Load data for the next state
            market_filename = f'market_{self.episode-1}.json'
            resource_filename = f'resource_{self.episode-1}.json'
            
            
            with open(market_filename, 'r') as file:
                market_data = json.load(file)
            with open(resource_filename, 'r') as file:
                resource_data = json.load(file)
            if self.current_step < len(self.price_forecast):
                self.bus = resource_data['bus']
                self.rid = resource_data['rid']
                
                if 'EN' in market_data['previous'][self.market_type]['prices']:
                    next_price_forecast = pd.to_numeric(market_data['previous'][self.market_type]['prices']['EN'][self.bus][self.current_step])
                else:
                    next_price_forecast = 0.0
                
                if 'solar' in market_data['forecast']:
                    next_solar_data = pd.to_numeric(market_data['forecast']['solar'][self.current_step])
                else:
                    next_solar_data = 0.0
                
                if 'wind' in market_data['forecast']:
                    next_wind_data = pd.to_numeric(market_data['forecast']['wind'][self.current_step])
                else:
                    next_wind_data = 0.0
                
                if 'load' in market_data['forecast']:
                    next_load_data = pd.to_numeric(market_data['forecast']['load'][self.current_step])
                else:
                    next_load_data = 0.0
                
                next_soc_data = self.get_soc() * (0.4 * action + 0.6 * (1 - action))
                print("next_price_forecast is",next_price_forecast )
                print("next_solar is ", next_solar_data)
                print("next_wind is ", next_wind_data)
                print("next_load is ", next_load_data)
                print("next_soc is ", next_soc_data)
                
                next_state = np.array([next_price_forecast, next_solar_data, next_wind_data, next_load_data, next_soc_data])
            else:
                next_state = np.zeros(5)
                done = True  # Set done to True if self.current_step exceeds the valid range
        
                
        reward = self.calculate_reward(action)
        
        return next_state, reward, done
    
    def get_state(self):
        resource_filename = f'resource_{self.episode-1}.json'
        market_filename = f'market_{self.episode-1}.json'
        with open(resource_filename, 'r') as file:
            resource_data = json.load(file)
        with open(market_filename, 'r') as file:
            market_data = json.load(file)
        soc = resource_data['status'][self.rid]['soc']
        state = [self.price_forecast.iloc[self.current_step, 0],
                 self.solar_data.iloc[self.current_step, 0],
                 self.wind_data.iloc[self.current_step, 0],
                 self.load_data.iloc[self.current_step, 0],
                 soc]
        state = pd.to_numeric(state)
        return state
    def get_soc(self):
        resource_filename = f'resource_{self.episode-1}.json'
        with open(resource_filename, 'r') as file:
            resource_data = json.load(file)
            soc = resource_data['status'][self.rid]['soc']
        return soc

    def calculate_reward(self, action):
        dummy_offer = da.Agent(self.episode, market_info, resource_info).make_me_an_offer()
        def get_average(value):
            if isinstance(value, list):
                return np.mean(value)
            else:
                return value
        
        
        if 'DAM' in self.market_type:
            reward = 0
            timestamps = list(dummy_offer[self.rid]['block_ch_mc'].keys())
            for i in range(len(timestamps)):
                current_timestamp = timestamps[i]
                if i >= self.current_step and i < self.current_step + 24:
                    action_step = action[i - self.current_step]  # Use the action value for the current step
                    price_forecast = self.price_forecast.iloc[i, 0]
                    ch_mc = get_average(dummy_offer[self.rid]['block_ch_mc'][current_timestamp])
                    ch_mq = get_average(dummy_offer[self.rid]['block_ch_mq'][current_timestamp])
                    dc_mc = get_average(dummy_offer[self.rid]['block_dc_mc'][current_timestamp])
                    dc_mq = get_average(dummy_offer[self.rid]['block_dc_mq'][current_timestamp])
                    soc_mc = get_average(dummy_offer[self.rid]['block_soc_mc'][current_timestamp])
                    soc_mq = get_average(dummy_offer[self.rid]['block_soc_mq'][current_timestamp])
                    ch_reward = np.array(ch_mq * (price_forecast - abs((action_step-1) * ch_mc)), dtype=np.float32)
                    dc_reward = np.array(dc_mq * (price_forecast - abs((1-action_step * dc_mc))), dtype=np.float32)
                    soc_reward = np.array(soc_mq * (price_forecast - abs((1-action_step) * soc_mc)), dtype=np.float32)
                    alpha=0.6
                    reward = alpha*(dc_reward - ch_reward) +(1-alpha)
        elif 'RTM' in self.market_type:
            timestamps = list(dummy_offer[self.rid]['block_soc_mc'].keys())
            current_timestamp = timestamps[self.current_step]
            price_forecast = self.price_forecast.iloc[self.current_step, 0]
            soc_mc = get_average(dummy_offer[self.rid]['block_soc_mc'][current_timestamp]) if current_timestamp in dummy_offer[self.rid]['block_soc_mc'] else 0
            soc_mq = get_average(dummy_offer[self.rid]['block_soc_mq'][current_timestamp]) if current_timestamp in dummy_offer[self.rid]['block_soc_mq'] else 0
            action_step = action[0]  # Use the action value for the current step
            soc_reward = np.array(soc_mq * (price_forecast - abs((1-action_step) * soc_mc)), dtype=np.float32)
            reward = soc_reward
        return reward
    def set_episode(self, episode):
        self.episode = episode

""" a new main file for train and tuning mode
if __name__ == "__main__":
    # ... (existing code) ...

    # Initialize DDPG agent with the best hyperparameters
    hidden_dim = (256, 128)  # Replace with the best hyperparameters found during tuning
    agent = DDPGAgent(state_dim=state_dim, action_dim=action_dim, action_bound=action_bound, hidden_dim=hidden_dim, mode=args.mode)

    # Training/fine-tuning loop
    for episode in range(time_step, time_step + args.num_episodes):
        env = EnergyEnvironment(episode)
        state = env.reset()
        done = False

        while not done:
            action = agent.choose_action(state)
            factors = action

            scaled_agent = Scaled_agent(episode, market_info, resource_info)
            scaled_agent.scaling(da, factors)

            next_state, reward, done = env.step(factors)

            # Train or fine-tune the agent
            if args.mode == 'train':
                agent.train(state, action)
            elif args.mode == 'fine-tune':
                agent.train(state, action)

            state = next_state

        # Save the updated model after each episode
        agent.save_model(f'actor_model_{episode}.pth')

    # Save the final model
    agent.save_model('actor_model.pth') """    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('time_step', type=int, help='Integer time step tracking the progress of the\
                        simulated market.')
    parser.add_argument('market_file', help='json formatted dictionary with market information.')
    parser.add_argument('resource_file', help='json formatted dictionary with resource information.')
    """ parser.add_argument('--mode', type=str, choices=['train', 'fine-tune'], default='train',
                    help='Specify the mode: "train" to train a new model, "fine-tune" to fine-tune a pre-trained model.') """
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
    learning_rate = 0.001
    hidden_dim=(256,128)
    buffer_size=int(1e5) 
    batch_size=64 
    gamma=0.99
    tau=0.001
    agent = DDPGAgent(state_dim=state_dim, learning_rate=learning_rate,hidden_dim=hidden_dim,
                  buffer_size=buffer_size,batch_size=batch_size, gamma=gamma,
                   tau=tau, action_dim=action_dim, action_bound=action_bound)


    # Initialize variables
    total_reward = 0

    # Testing loop
    while True:
        env = EnergyEnvironment(time_step)
        state = env.reset()
        done = False
        episode_reward = 0
        actions_taken = []

        while not done:
            action = agent.choose_action(state)
            factors = action

            scaled_agent = Scaled_agent(time_step, market_info, resource_info)
            scaled_agent.scaling(da, factors)

            next_state, reward, done = env.step(action)
            
            # Estimate the reward based on the action taken
            reward = env.calculate_reward(action)  # Calculate the reward
            episode_reward += reward
            
            state = next_state

        total_reward += episode_reward
        print(f"Episode {time_step}: Estimated Reward = {episode_reward}")

        # Save actions taken in the episode
        action_filename = f'action_{time_step}.json'
        action_data = {'actions': factors.tolist()}
        with open(action_filename, 'w') as file:
            json.dump(action_data, file)
        
        time_step += 1
        
        # Check if the simulation has reached the end
        if time_step >= 289 * 1: # test 1 day
            break

    print(f"Total Estimated Reward over {time_step} episodes: {total_reward}")