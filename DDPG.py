""" Best paramters for DDPG """
# Import necessary libraries
import numpy as np
import pandas as pd
print(pd.__version__)
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import namedtuple, deque

batch_size = 64

# Define Actor Network
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim,action_bound, hidden_dim):
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
            action = action.unsqueeze(-1)
        x = torch.cat([state, action], dim=-1)  # Concatenate along the second dimension
        #print("x", x.size())
        x = x.float()  # Cast input to float data type
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    

# Define DDPG Agent
class DDPGAgent:

    def __init__(self, state_dim, action_dim, action_bound,hidden_dim, buffer_size, 
                 learning_rate,batch_size, gamma, tau):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bound = action_bound
        self.buffer_size = buffer_size  
        self.batch_size = batch_size
        self.gamma = gamma
        self.hidden_dim = hidden_dim
        self.lr = learning_rate  
        self.tau = tau
        self.actor = Actor(state_dim, action_dim, action_bound,hidden_dim)
        self.target_actor = Actor(state_dim, action_dim, action_bound, hidden_dim)
        self.critic = Critic(state_dim, action_dim)
        self.target_critic = Critic(state_dim, action_dim)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), self.lr)#lr=0.002
        self.critic_optimizer = optim.Adam(self.critic.parameters(), self.lr)#lr=0.002
        self.memory = ReplayBuffer(buffer_size)
    
    def choose_action(self, state, noise=0.1):
        state = torch.FloatTensor(state).unsqueeze(0)
        action = self.actor(state).detach().cpu().numpy().squeeze(0)
        # Add noise to the action
        noise = np.random.normal(0, noise, size=self.action_dim)
        action += noise

        """ action = np.clip(action, -self.action_bound, self.action_bound)
        action = np.clip(action, -self.action_bound, self.action_bound) """
        action = np.clip(action, action_min, action_max)
        return action
    
    def learn(self):
        if len(self.memory) < self.batch_size:
            return
        
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards).unsqueeze(-1)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones).unsqueeze(-1)
        
        target_actions = self.target_actor(next_states)
        target_actions = target_actions
        target_values = self.target_critic(next_states, target_actions)
        target_returns = rewards + self.gamma * target_values * (1 - dones)
        
        current_values = self.critic(states, actions)
        critic_loss = nn.MSELoss()(current_values, target_returns.detach())
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
        self.memory.append(self.Transition(state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)
    
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
            action = self.action_data.iloc[self.episode, self.current_step]
        
            next_state = [self.price_forecast.iloc[self.episode,self.current_step ],
                          self.solar_data.iloc[self.episode,self.current_step],
                          self.wind_data.iloc[self.episode,self.current_step],
                          self.load_data.iloc[self.episode,self.current_step],
                          self.soc_data.iloc[self.episode,self.current_step]]
            next_state = pd.to_numeric(next_state)
            
        else:
            next_state = np.zeros_like(state)  # Placeholder for terminal state
        reward = self.calculate_reward(action)
        
        return next_state, reward, done
    def calculate_reward(self, action):   
        combined_ch_mc_data = pd.read_csv("train_data/combined_df_ch_mc.csv", index_col=False, usecols=lambda x: x != 'Unnamed: 0', header=None)
        combined_ch_mc_data.index = pd.to_numeric(combined_ch_mc_data.index, errors='coerce')
        combined_ch_mc_data = combined_ch_mc_data.fillna(0).replace([np.inf, -np.inf], 0)
        combined_ch_mc_data.columns = combined_ch_mc_data.columns.astype('float64')
        combined_ch_mq_data = pd.read_csv("train_data/combined_df_ch_mq.csv",index_col=False, usecols=lambda x: x != 'Unnamed: 0', header=None)
        combined_dc_mc_data = pd.read_csv("train_data/combined_df_dc_ch.csv",index_col=False, usecols=lambda x: x != 'Unnamed: 0', header=None)
        combined_dc_mq_data = pd.read_csv("train_data/combined_df_dc_mq.csv",index_col=False, usecols=lambda x: x != 'Unnamed: 0', header=None)
        combined_soc_mc_data = pd.read_csv("train_data/combined_df_soc_mc.csv",index_col=False, usecols=lambda x: x != 'Unnamed: 0', header=None)
        combined_soc_mq_data = pd.read_csv("train_data/combined_df_soc_mq.csv",index_col=False, usecols=lambda x: x != 'Unnamed: 0', header=None)

        # Check if the current state-action pair exists in the predefined dataset
        if self.episode in combined_ch_mc_data.index and self.current_step in combined_ch_mc_data.columns:
            ch_mc = torch.tensor(pd.to_numeric(combined_ch_mc_data.iloc[self.episode, self.current_step], errors='coerce'), dtype=torch.float32)
            ch_mq = torch.tensor(pd.to_numeric(combined_ch_mq_data.iloc[self.episode, self.current_step], errors='coerce'), dtype=torch.float32)
            dc_mc = torch.tensor(pd.to_numeric(combined_dc_mc_data.iloc[self.episode, self.current_step], errors='coerce'), dtype=torch.float32)
            dc_mq = torch.tensor(pd.to_numeric(combined_dc_mq_data.iloc[self.episode, self.current_step], errors='coerce'), dtype=torch.float32)
            soc_mc = torch.tensor(pd.to_numeric(combined_soc_mc_data.iloc[self.episode, self.current_step], errors='coerce'), dtype=torch.float32)
            soc_mq = torch.tensor(pd.to_numeric(combined_soc_mq_data.iloc[self.episode, self.current_step], errors='coerce'), dtype=torch.float32)
        else:
            # Find the nearest matching state-action pair from the predefined dataset
            #nearest_episode = np.argmin(np.abs(combined_ch_mc_data.index - self.episode))
            #nearest_step = np.argmin(np.abs(combined_ch_mc_data.columns.astype(int) - self.current_step))
            nearest_episode = np.argmin(np.abs(combined_ch_mc_data.index - self.episode))
            nearest_step_index = np.abs(combined_ch_mc_data.columns.astype(int) - self.current_step).idxmin()
            nearest_step = combined_ch_mc_data.columns.get_loc(nearest_step_index)
            ch_mc = torch.tensor(combined_ch_mc_data.iloc[nearest_episode, nearest_step], dtype=torch.float32)
            ch_mq = torch.tensor(combined_ch_mq_data.iloc[nearest_episode, nearest_step], dtype=torch.float32)
            dc_mc = torch.tensor(combined_dc_mc_data.iloc[nearest_episode, nearest_step], dtype=torch.float32)
            dc_mq = torch.tensor(combined_dc_mq_data.iloc[nearest_episode, nearest_step], dtype=torch.float32)
            soc_mc = torch.tensor(combined_soc_mc_data.iloc[nearest_episode, nearest_step], dtype=torch.float32)
            soc_mq = torch.tensor(combined_soc_mq_data.iloc[nearest_episode, nearest_step], dtype=torch.float32)
        
        action_step = torch.tensor(action_data.iloc[self.episode, self.current_step], dtype=torch.float32)
        price_forecast = torch.tensor(self.price_forecast.iloc[self.episode, self.current_step], dtype=torch.float32)
        
        done = self.current_step + 1 >= self.max_steps
        if done:
            reward = self.rewards_data.iloc[self.episode, 1]
        else:
            ch_reward = np.array(ch_mq * (price_forecast - abs((action_step-1) * ch_mc)), dtype=np.float32)
            dc_reward = np.array(dc_mq * (price_forecast - abs((1-action_step * dc_mc))), dtype=np.float32)
            soc_reward = np.array(soc_mq * (price_forecast - abs((1-action_step) * soc_mc)), dtype=np.float32)
            alpha = 0.6  # You can adjust the value of alpha as needed
            reward = alpha * (dc_reward - ch_reward) + (1 - alpha) * soc_reward
        
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
learning_rate = 0.001
hidden_dim=(256,128)
buffer_size=int(1e5) 
batch_size=64 
gamma=0.99
tau=0.001

action_max = action_data.max().values
action_min = action_data.min().values
action_bound = action_max - action_min

# Initialize DDPG agent
agent = DDPGAgent(state_dim=state_dim, learning_rate=learning_rate,hidden_dim=hidden_dim,
                  buffer_size=buffer_size,batch_size=batch_size, gamma=gamma,
                   tau=tau, action_dim=action_dim, action_bound=action_bound)

# Create an instance of the EnergyEnvironment
env = EnergyEnvironment(price_forecast, solar_data, wind_data, load_data, soc_data, action_data, rewards_data)

# Initialize environment
initial_state = np.zeros(state_dim)  # Placeholder initial state
#print("max_steps",price_forecast.shape[1]) # Placeholder max steps per episode

num_episodes = len(action_data)

# Training loop
for episode in range(num_episodes):
    state = env.reset()
    env.set_episode(episode)
    done = False
    while not done:
        action = agent.choose_action(state)
        next_state, reward, done = env.step(action)
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        agent.learn()
    print(f"Episode {episode}: Total Reward = {reward}")
    
    # Train the agent after every 10 episodes
    if (episode + 1) % 10 == 0:
        agent.learn()
# Save the trained model
torch.save(agent.actor.state_dict(), 'actor_model.pth')
torch.save(agent.critic.state_dict(), 'critic_model.pth')