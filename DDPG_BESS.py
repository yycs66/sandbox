""" tuning parameters """
# Import necessary libraries
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import namedtuple, deque
from itertools import product
from sklearn.model_selection import train_test_split

#batch_size = 64

# Define Actor Network
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim,action_bound, hidden_dim=(256,128)):
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
    def __init__(self, state_dim, action_dim, action_min, action_max,action_bound, buffer_size=int(1e5), batch_size=64, gamma=0.99, tau=0.001):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_min = action_min
        self.action_max = action_max
        self.action_bound = action_bound
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.actor = Actor(state_dim, action_dim, action_bound)
        self.target_actor = Actor(state_dim, action_dim, action_bound)
        self.critic = Critic(state_dim, action_dim)
        self.target_critic = Critic(state_dim, action_dim)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=0.001)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=0.002)
        self.memory = ReplayBuffer(buffer_size)
    
    def choose_action(self, state, noise=0.1):
        state = torch.FloatTensor(state).unsqueeze(0)
        action = self.actor(state).detach().cpu().numpy().squeeze(0)
        # Add noise to the action
        noise = np.random.normal(0, noise, size=self.action_dim)
        action += noise

        action = np.clip(action, -1, 1)  #add action to actor network's output before applying bounds
        # Scale the action to the original range
        action = self.action_min + (action + 1) * (self.action_max - self.action_min) / 2
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
        if done:
            reward = self.rewards_data.iloc[self.episode, 1]
        else:
            reward = 0  # Placeholder reward for non-terminal steps
        
        return next_state, reward, done
    
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

# Normalize the data to [0, 1]
price_min = price_forecast.min().min()
price_max = price_forecast.max().max()
price_forecast = (price_forecast - price_min) / (price_max - price_min)

solar_min = solar_data.min().min()
solar_max = solar_data.max().max()
solar_data = (solar_data - solar_min) / (solar_max - solar_min)

wind_min = wind_data.min().min()
wind_max = wind_data.max().max()
wind_data = (wind_data - wind_min) / (wind_max - wind_min)

load_min = load_data.min().min()
load_max = load_data.max().max()
load_data = (load_data - load_min) / (load_max - load_min)

soc_min = soc_data.min().min()
soc_max = soc_data.max().max()
soc_data = (soc_data - soc_min) / (soc_max - soc_min)


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
action_max = action_data.max().values
action_min = action_data.min().values
action_bound = action_max - action_min

# Initialize DDPG agent
agent = DDPGAgent(state_dim=state_dim, action_dim=action_dim,action_min=action_min, action_max=action_max, action_bound=action_bound)

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
    #print(f"Episode {episode}: Total Reward = {reward}")
    
    # Train the agent after every 10 episodes
    if (episode + 1) % 10 == 0:
        agent.learn()
# Save the trained model
torch.save(agent.actor.state_dict(), 'actor_model.pth')
torch.save(agent.critic.state_dict(), 'critic_model.pth')

# Define the range of hyperparameters to tune
hyperparams = {
    'hidden_dim': [(128, 64), (256, 128), (512, 256)],
    'batch_size': [32, 64, 128],
    'gamma': [0.95, 0.99],
    'tau': [0.001, 0.005, 0.01],
    'noise_scale': [0.05, 0.1, 0.2]
}

# Generate all combinations of hyperparameters
hyperparam_combinations = list(product(*hyperparams.values()))

# Split the data into training and validation sets
train_indices, val_indices = train_test_split(range(len(action_data)), test_size=0.2, random_state=42)

best_reward = -np.inf
best_hyperparams = None

# Iterate over each combination of hyperparameters
for hidden_dim, batch_size, gamma, tau, noise_scale in hyperparam_combinations:
    # Initialize DDPG agent with the current hyperparameters
    agent = DDPGAgent(state_dim=state_dim, action_min=action_min, action_max=action_max, action_dim=action_dim, action_bound=action_bound,
                      buffer_size=int(1e5), batch_size=batch_size, gamma=gamma, tau=tau)

    # Create an instance of the EnergyEnvironment
    env = EnergyEnvironment(price_forecast, solar_data, wind_data, load_data, soc_data, action_data, rewards_data)

    num_episodes = len(train_indices)
    total_reward = 0

    # Training loop
    for episode_idx in range(num_episodes):
        episode = train_indices[episode_idx]
        state = env.reset()
        env.set_episode(episode)
        done = False
        episode_reward = 0

        while not done:
            action = agent.choose_action(state, noise=noise_scale)
            next_state, reward, done = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward
            agent.learn()

        total_reward += episode_reward
        #print(f"Episode {episode_idx}: Total Reward = {episode_reward}")

        # Train the agent after every 10 episodes
        if (episode_idx + 1) % 10 == 0:
            agent.learn()

    # Evaluate the agent on the validation set
    val_reward = 0
    for episode_idx in range(len(val_indices)):
        episode = val_indices[episode_idx]
        state = env.reset()
        env.set_episode(episode)
        done = False
        episode_reward = 0

        while not done:
            action = agent.choose_action(state, noise=0)  # No noise during evaluation
            next_state, reward, done = env.step(action)
            state = next_state
            episode_reward += reward

        val_reward += episode_reward

    avg_val_reward = val_reward / len(val_indices)
    #print(f"Hyperparameters: hidden_dim={hidden_dim}, batch_size={batch_size}, gamma={gamma}, tau={tau}, noise_scale={noise_scale}")
    #print(f"Average Validation Reward: {avg_val_reward}")

    if avg_val_reward > best_reward:
        best_reward = avg_val_reward
        best_hyperparams = (hidden_dim, batch_size, gamma, tau, noise_scale)

#print(f"Best Hyperparameters: hidden_dim={best_hyperparams[0]}, batch_size={best_hyperparams[1]}, gamma={best_hyperparams[2]}, tau={best_hyperparams[3]}, noise_scale={best_hyperparams[4]}")
#print(f"Best Average Validation Reward: {best_reward}")