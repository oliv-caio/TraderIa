import gym
import gym_anytrading
from stable_baselines.common.vec_env import DummyVecEnv
from gym_anytrading.datasets import FOREX_EURUSD_1H_ASK
from stable_baselines import DQN
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

env = gym.make('forex-v0', 
    df = FOREX_EURUSD_1H_ASK, 
    frame_bound=(250,1500), 
    window_size=250)


model = DQN('MlpPolicy', env, 
    prioritized_replay=True, 
    gamma=0.99, 
    buffer_size=10000)

model.learn(total_timesteps=10000)

env = gym.make('forex-v0', 
	df = FOREX_EURUSD_1H_ASK, 
	frame_bound=(250,1500), 
	window_size=250)
	
obs = env.reset()

while True: 
    obs = obs[np.newaxis, ...]
    action, _states = model.predict(obs)
    
    
    obs, rewards, done, info = env.step(action)
    if done:
        print(info)
        break
        
plt.cla()
env.render_all()
plt.show()
