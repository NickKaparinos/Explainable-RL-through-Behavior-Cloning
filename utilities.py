"""
Nikos Kaparinos 119 - Julio Hamiti 106
2023
"""
import cv2
import gym
import numpy as np
import pandas as pd
from gym import wrappers
from os import makedirs


# def add_feature_names(dataset: np.array, env_id: str) -> pd.DataFrame:
#     """ Converts the dataset from np.array to pd.DataFrame by adding column names """
#     if env_id == 'CartPole-v1':
#         feature_names = ["Cart Position", "Cart Velocity", "Pole Angle", "Pole Angular Velocity", "Action"]
#     # TODO
#
#     dataset = pd.DataFrame(data=dataset, columns=feature_names)
#     return dataset


def evaluate_agent(agent, env, predict_fn, num_episodes: int = 100) -> list:
    """ Evaluates agent on RL environment"""
    reward_list = []
    episode_reward = 0
    i = 0
    obs = env.reset()
    while i < num_episodes:
        action = predict_fn(agent, obs)
        obs, reward, done, info = env.step(action)
        episode_reward += reward
        if done:
            obs = env.reset()
            i += 1
            reward_list.append(episode_reward)
            episode_reward = 0
    return reward_list


def record_videos(agent, env, predict_fn, log_dir: str, num_episodes: int = 5, prefix: str = ''):
    """ Records video of trained agent"""
    video_frames_list = []
    video_dir = f'{log_dir}videos/'
    makedirs(video_dir, exist_ok=True)

    i = 0
    obs = env.reset()
    while i < num_episodes:
        action = predict_fn(agent, obs)
        obs, reward, done, info = env.step(action)
        frame = env.render(mode='rgb_array')
        video_frames_list.append(frame)
        if done:
            obs = env.reset()
            i += 1
            encode_video(video_frames_list, video_dir, i, prefix=prefix)
            video_frames_list.clear()


def encode_video(video_frames_list: list, video_dir: str, video_num: int, fps: float = 60.0, prefix: str = '') -> None:
    """ Encodes video frames list to .mp4 video """
    width, height = video_frames_list[0].shape[0], video_frames_list[0].shape[1]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(f'{video_dir}{prefix}video{video_num}.mp4', fourcc, fps, (height, width))

    for frame in video_frames_list:
        video.write(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    cv2.destroyAllWindows()
    video.release()


def sklearn_predict_fn(agent, obs: np.array):
    """ Predict function for the decision tree and logistic regression agents """
    action = agent.predict(obs.reshape(1, -1))
    return int(action[0])


def ppo_predict_fn(model, obs):
    """ Predict function for the PPO agent """
    action, _state = model.predict(obs, deterministic=False)
    return action


def compare_agents_performance(agent1_mean_reward, agent2_mean_reward):
    """ Compares agent 1 and agent 2 performance"""
    if np.sign(agent1_mean_reward) == 1:
        if np.sign(agent2_mean_reward) == 1:
            return agent2_mean_reward / agent1_mean_reward
        else:
            return 0
    else:
        return agent1_mean_reward / agent2_mean_reward
