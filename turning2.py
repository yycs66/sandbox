"tuning parameters"
import numpy as np
import pandas as pd
import torch
from DDPG import DDPGAgent, EnergyEnvironment

# Load and preprocess data
price_forecast = pd.read_csv('train_data/price_forecast2_all.csv', index_col=False, usecols=lambda x: x != 'Unnamed: 0')
solar_data = pd.read_csv('train_data/solar_forecast_all.csv', index_col=False, usecols=lambda x: x != 'Unnamed: 0')
wind_data = pd.read_csv('train_data/wind_forecast_all.csv', index_col=False, usecols=lambda x: x != 'Unnamed: 0')
load_data = pd.read_csv('train_data/load_forecast_all.csv', index_col=False, usecols=lambda x: x != 'Unnamed: 0')
soc_data = pd.read_csv('train_data/soc_all.csv', index_col=False, usecols=lambda x: x != 'Unnamed: 0')
action_data = pd.read_csv('train_data/factor_all.csv', index_col=False, usecols=lambda x: x != 'Unnamed: 0')
rewards_data = pd.read_csv('train_data/score_all.csv', index_col=False, usecols=lambda x: x != 'Unnamed: 0')

# Define state and action dimensions
state_dim = 5
action_dim = action_data.shape[1]
action_max = action_data.max().values
action_min = action_data.min().values
action_bound = action_max - action_min

# Define hyperparameter search space
hidden_dim_options = [(128, 64), (256, 128), (512, 256)]
learning_rate_options = [0.001, 0.0005]
batch_size_options = [32, 64, 128]
gamma_options = [0.95, 0.99]
tau_options = [0.001, 0.01]

# Create an instance of the EnergyEnvironment
env = EnergyEnvironment(price_forecast, solar_data, wind_data, load_data, soc_data, action_data, rewards_data)

best_reward = -np.inf
best_params = None
num_episodes = 289
action_bound= 2

# Iterate over hyperparameter combinations
for hidden_dim in hidden_dim_options:
    for learning_rate in learning_rate_options:
        for batch_size in batch_size_options:
            for gamma in gamma_options:
                for tau in tau_options:
                    # Initialize DDPG agent with current hyperparameters
                    agent = DDPGAgent(state_dim=state_dim, action_dim=action_dim, action_bound=action_bound, hidden_dim=hidden_dim,
                                      learning_rate=learning_rate, batch_size=batch_size, gamma=gamma, tau=tau)
                    
                    # Training loop
                    
                    total_reward = 0
                    for episode in range(min(num_episodes, len(action_data))):
                        state = env.reset()
                        env.set_episode(episode)
                        done = False
                        while not done:
                            action = agent.choose_action(state)
                            next_state, reward, done = env.step(action)
                            agent.remember(state, action, reward, next_state, done)
                            state = next_state
                            agent.learn()
                        total_reward += reward
                    
                    # Check if current hyperparameters yield better performance
                    if total_reward > best_reward:
                        best_reward = total_reward
                        best_params = (hidden_dim, learning_rate, batch_size, gamma, tau)

print("Best hyperparameters:")
print("Hidden dimensions:", best_params[0])
print("Learning rate:", best_params[1])
print("Batch size:", best_params[2])
print("Gamma:", best_params[3])
print("Tau:", best_params[4])
print("Best total reward:", best_reward)