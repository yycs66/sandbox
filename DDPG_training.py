""" update rewards """
# Import necessary libraries
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import namedtuple, deque

batch_size = 64

# Define Actor Network
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim,action_bound, hidden_dim =(256,128)):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim[0])
        self.fc2 = nn.Linear(hidden_dim[0], hidden_dim[1])
        self.fc3 = nn.Linear(hidden_dim[1], action_dim)
        self.action_bound = torch.tensor(action_bound)
        
    def forward(self, state):
        x = state.float()  # Cast input to float data type
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x)) * self.action_bound
        return x

# Define Critic Network
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)
        
    def forward(self, state, action):
        # Cast input tensors to float data type
        state = state.float()
        action = action.float()
        if action.dim() == 1:
            action = action.unsqueeze(1) 
        x = torch.cat([state, action], dim=1)  # Concatenate along the second dimension
        x = x.float()  # Cast input to float data type
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    

# Define DDPG Agent
class DDPGAgent:
    #def __init__(self, state_dim, action_dim, action_bound, buffer_size=int(1e5), batch_size=64, gamma=0.99, tau=0.001):
    def __init__(self, state_dim, action_dim, action_bound,action_min, action_max, buffer_size=int(1e5), batch_size=64, gamma=0.99, tau=0.001):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bound = action_bound
        self.action_min = action_min
        self.action_max = action_max
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.actor = Actor(state_dim, action_dim, action_bound)
        self.target_actor = Actor(state_dim, action_dim, action_bound)
        self.critic = Critic(state_dim, action_dim)
        self.target_critic = Critic(state_dim, action_dim)
        buffer_size = int(1e5)  # Define the buffer size
        self.memory = ReplayBuffer(buffer_size)
        
            
    def choose_action(self, state, noise=0.1):
        state = torch.FloatTensor(state).unsqueeze(0)
        action = self.actor(state).detach().cpu().numpy().squeeze(0)
        # Add noise to the action
            
        action = np.clip(action, -1, 1)  # Clip actions to the range [-1, 1]
        action = (action + 1) / 2  # Scale actions to the range [0, 1]
        return action[0] # return a scalar value
    
    def learn(self):
        if len(self.memory) < self.batch_size:
            return
        
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards).unsqueeze(-1)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones).unsqueeze(-1)
        
        target_actions = self.target_actor(next_states).view(-1, self.action_dim)
        target_actions = target_actions
        target_values = self.target_critic(next_states, target_actions)
        target_returns = rewards + self.gamma * target_values * (1 - dones)
        
        current_values = self.critic(states, actions.unsqueeze(-1))
        #critic_loss = nn.MSELoss()(current_values, target_returns.detach())
        critic_loss = nn.MSELoss()(current_values, target_returns)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        policy_loss = -self.critic(states, self.actor(states)).mean()
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()
        
        self.soft_update(self.actor, self.target_actor)
        self.soft_update(self.critic, self.target_critic)
    
    def soft_update(self, local_model, target_model):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)

# Define Replay Buffer
class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.memory = deque(maxlen=buffer_size)
        self.Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))
    
    def add(self, state, action, reward, next_state, done):
        state = np.array(state, dtype=np.float32)
        action = np.array(action, dtype=np.float32)
        next_state = np.array(next_state, dtype=np.float32)
        done = np.array([done], dtype=np.float32)
        reward = np.array([reward], dtype=np.float32).flatten()[0]
        
        self.memory.append(self.Transition(state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = np.array(states)
        actions = np.array(actions)
        #rewards = np.array(list(rewards)).reshape((-1, 1))
        rewards = np.array(rewards)
        next_states = np.array(next_states)
        dones = np.array(dones)
        """ print("States shape:", np.shape(states))
        print("Actions shape:", np.shape(actions))
        print("Rewards shape:", np.shape(rewards))
        print("Next_states shape:", np.shape(next_states))
        print("Dones shape:", np.shape(dones)) """
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.memory)

 # Define the Energy Environment
class EnergyEnvironment:
    def __init__(self, price_forecast, solar_data, wind_data, load_data, soc_data, action_data, rewards_data):
        self.price_forecast = price_forecast
        self.solar_data = solar_data
        self.wind_data = wind_data
        self.load_data = load_data
        self.soc_data = soc_data
        self.action_data = action_data
        self.rewards_data = rewards_data
        self.current_step = 0
        self.episode = 0
        self.max_episodes = len(action_data)-1
        self.max_steps = price_forecast.shape[1]
        
    def reset(self):
        self.current_step = 0
        state = [self.price_forecast.iloc[self.episode,self.current_step],
                 self.solar_data.iloc[self.episode,self.current_step],
                 self.wind_data.iloc[self.episode,self.current_step],
                 self.load_data.iloc[self.episode,self.current_step],
                 self.soc_data.iloc[self.episode,self.current_step]]
        
        state = pd.to_numeric(state)
        return state
    
    def step(self, action):
        # Update the state based on the action taken
       
        # Check if the episode is done
        done = self.current_step +1 >= self.max_steps
        
        if not done:
            self.current_step += 1
            action_value = self.action_data.iloc[self.episode, self.current_step]
        
            next_state = [self.price_forecast.iloc[self.episode,self.current_step ],
                          self.solar_data.iloc[self.episode,self.current_step],
                          self.wind_data.iloc[self.episode,self.current_step],
                          self.load_data.iloc[self.episode,self.current_step],
                          self.soc_data.iloc[self.episode,self.current_step]]
            next_state = pd.to_numeric(next_state)
            
        else:
            next_state = np.zeros_like(state)  # Placeholder for terminal state
        
        # Get the reward from the dataset
        reward = self.calculate_reward(action)
        
        return next_state, reward, done
    def calculate_reward(self, action):
        combined_ch_mc_data = pd.read_csv("train_data/combined_df_ch_mc.csv")
        combined_ch_mq_data = pd.read_csv("train_data/combined_df_ch_mq.csv")
        combined_dc_mc_data = pd.read_csv("train_data/combined_df_dc_ch.csv")
        combined_dc_mq_data = pd.read_csv("train_data/combined_df_dc_mq.csv")
        combined_soc_mc_data = pd.read_csv("train_data/combined_df_soc_mc.csv")
        combined_soc_mq_data = pd.read_csv("train_data/combined_df_soc_mq.csv")
        #action = self.action_data.iloc[self.episode, self.current_step]
        
        price_forecast = torch.tensor(self.price_forecast.iloc[self.episode, self.current_step], dtype=torch.float32)
        ch_mc = torch.tensor(combined_ch_mc_data.iloc[self.episode, self.current_step], dtype=torch.float32)
        ch_mq = torch.tensor(combined_ch_mq_data.iloc[self.episode, self.current_step], dtype=torch.float32)
        dc_mc = torch.tensor(combined_dc_mc_data.iloc[self.episode, self.current_step], dtype=torch.float32)
        dc_mq = torch.tensor(combined_dc_mq_data.iloc[self.episode, self.current_step], dtype=torch.float32)
        soc_mc = torch.tensor(combined_soc_mc_data.iloc[self.episode, self.current_step], dtype=torch.float32)
        soc_mq = torch.tensor(combined_soc_mq_data.iloc[self.episode, self.current_step], dtype=torch.float32)


        done = self.current_step + 1 >= self.max_steps
        ch_reward = np.array(ch_mq*(price_forecast - abs(action * ch_mc)), dtype=np.float32)
        dc_reward = np.array(dc_mq*(price_forecast - abs(action * dc_mc)), dtype=np.float32)
        soc_reward = np.array(soc_mq*(price_forecast - abs(action * soc_mc)), dtype=np.float32)
        reward = ch_reward + dc_reward + soc_reward
        """ if done:
            reward =np.array(self.rewards_data.iloc[self.episode, 1],dtype=np.float32)
        else:
            ch_reward = np.array(ch_mq*(price_forecast - abs(action * ch_mc)), dtype=np.float32)
            dc_reward = np.array(dc_mq*(price_forecast - abs(action * dc_mc)), dtype=np.float32)
            soc_reward = np.array(soc_mq*(price_forecast - abs(action * soc_mc)), dtype=np.float32)
            reward = ch_reward + dc_reward + soc_reward """

        
        return reward   
    def set_episode(self, episode):
        self.episode = episode

# Load and preprocess data
price_forecast = pd.read_csv('train_data/price_forecast2_all.csv', index_col=False, usecols=lambda x: x != 'Unnamed: 0')
#print(price_forecast)
solar_data = pd.read_csv('train_data/solar_forecast_all.csv',index_col=False, usecols=lambda x: x != 'Unnamed: 0')
wind_data = pd.read_csv('train_data/wind_forecast_all.csv',index_col=False, usecols=lambda x: x != 'Unnamed: 0')
load_data =pd.read_csv('train_data/load_forecast_all.csv',index_col=False, usecols=lambda x: x != 'Unnamed: 0')
soc_data = pd.read_csv('train_data/soc_all.csv',index_col=False, usecols=lambda x: x != 'Unnamed: 0')
action_data = pd.read_csv('train_data/factor_all.csv',index_col=False, usecols=lambda x: x != 'Unnamed: 0')
rewards_data = pd.read_csv('train_data/score_all.csv',index_col=False, usecols=lambda x: x != 'Unnamed: 0')


# Preprocess data and define state and action dimensions
#state_dim = len(price_forecast.columns) + len(solar_data.columns) + len(wind_data.columns) + len(load_data.columns) + len(soc_data.columns)
# Define state and action dimensions
price_dim = price_forecast.shape[1]
solar_dim = solar_data.shape[1]
wind_dim = wind_data.shape[1]
load_dim = load_data.shape[1]
soc_dim = soc_data.shape[1]


# Define state dimension as the sum of individual dimensions
state_dim = 5

# Define action dimension
action_dim = action_data.shape[1]
print('action dim is ',action_dim)
action_max = action_data.max().values
action_min = action_data.min().values
action_bound = action_max - action_min

# Initialize DDPG agent
agent = DDPGAgent(state_dim=state_dim, action_dim=action_dim, action_min=action_min, action_max=action_max,action_bound=action_bound)

# Create an instance of the EnergyEnvironment
env = EnergyEnvironment(price_forecast, solar_data, wind_data, load_data, soc_data, action_data, rewards_data)

# Initialize environment
initial_state = np.zeros(state_dim)  # Placeholder initial state
#print("max_steps",price_forecast.shape[1]) # Placeholder max steps per episode

#num_episodes = len(action_data)
num_episodes =1000

# Training loop
for episode in range(num_episodes):
    state = env.reset()
    env.set_episode(episode)
    done = False
    episode_reward = 0  # Track the total reward for each episode
    while not done:
        action = agent.choose_action(state)
        next_state, reward, done = env.step(action)
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        episode_reward += reward  # Accumulate the reward for the episode
        agent.learn()
    print(f"Episode {episode}: Total Reward = {reward}")
    
    # Train the agent after every 10 episodes
    if (episode + 1) % 10 == 0:
        agent.learn()
# Save the trained model
torch.save(agent.actor.state_dict(), 'actor_model.pth')
torch.save(agent.critic.state_dict(), 'critic_model.pth')