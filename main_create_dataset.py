"""
Nikos Kaparinos 119 - Julio Hamiti xxx
2023
"""
import gymnasium as gym
import numpy as np
import pandas as pd
from stable_baselines3 import A2C, PPO
import wandb
from wandb.integration.sb3 import WandbCallback
from gym import wrappers
import time
from os import makedirs
from utilities import *
from tqdm import tqdm
import pickle


def main():
    """ Create expert datasets using trained agent """
    # Environment
    start = time.perf_counter()
    env_id = 'CartPole-v1'
    saved_model = './logs/CartPole/PPO_Apr_07_2023_22_26_18/model.zip'
    dataset_size = 10_000
    env = gym.make(env_id)

    # Load agent
    model = PPO.load(saved_model)

    # Create dataset
    obs_shape = env.observation_space.shape[0]
    dataset = np.empty((dataset_size, obs_shape + 1))
    obs = env.reset()
    for step in tqdm(range(dataset_size)):
        last_obs = obs
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        dataset[step, 0:-1] = last_obs
        dataset[step, -1] = action
        if done:
            obs = env.reset()

    # Save datasets
    datasets_dir = f'logs/{env_id[:-3]}'
    makedirs(datasets_dir, exist_ok=True)
    with open(f'{datasets_dir}/{env_id[:-3]}_dataset_{dataset_size}.pickle', 'wb') as f:
        pickle.dump(dataset, f)

    # Execution Time
    end = time.perf_counter()
    print(f"\nExecution time = {end - start:.2f} second(s)")


if __name__ == '__main__':
    main()
