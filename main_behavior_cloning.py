"""
Nikos Kaparinos 119 - Julio Hamiti 106
2023
"""

import os
import pickle
import time
from itertools import product
from statistics import mean
from sklearn import tree
import gymnasium as gym
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from stable_baselines3 import PPO
from tqdm import tqdm
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPClassifier
import wandb
from utilities import *


def main():
    """ Train behavior cloning models """
    start = time.perf_counter()
    env_id = 'CartPole-v1'
    saved_model = './logs/CartPole/PPO_Apr_07_2023_22_26_18/model.zip'
    log_dir = f'logs/{env_id[:-3]}/'
    datasets_dir = f'logs/{env_id[:-3]}'
    n_dataset = 10000
    n_videos = 6
    video_dir = f'{log_dir}Dataset-{n_dataset}/'
    os.makedirs(log_dir, exist_ok=True)

    # Load dataset
    dataset_name = f'dataset_{n_dataset}.pickle'
    project_name = f'AML-BC-{env_id[:-3]}-{n_dataset}'
    with open(f'{datasets_dir}/{dataset_name}', 'rb') as f:
        dataset = pickle.load(f)
    dataset = add_feature_names(dataset, env_id)
    X, y = dataset.iloc[:, :-1], dataset.iloc[:, -1]

    # Env
    env = gym.make(env_id)

    # Agent evaluation
    model = PPO.load(saved_model)
    rl_agent_reward_list = evaluate_agent(model, env, ppo_predict_fn, num_episodes=10)
    rl_agent_mean = mean(rl_agent_reward_list)

    # # SVM classifier
    # best_performance, best_model = -9999, None
    # kernels = ['linear', 'poly', 'rbf']
    # gamma_list = [0.01, 0.1, 0.25, 0.5, 1.0, 2.0, 10.0, 100.0]
    # C_list = [0.01, 0.1, 0.25, 0.5, 1.0, 2.0, 10.0, 100.0]
    # # gamma_list = [1]
    # # C_list = [1]

    # for kernel, C in tqdm(product(kernels, C_list)):
    #     for idx, gamma in enumerate(gamma_list):
    #         if idx >= 1 and kernel != 'rbf':
    #             continue
    #         cls = SVC(C=C, kernel=kernel, gamma=gamma, random_state=0)
    #         model = Pipeline([
    #             ('scaler', MinMaxScaler()),
    #             ('clf', cls)])
    #
    #         train_start = time.perf_counter()
    #         model.fit(X, y)
    #         train_end = time.perf_counter()
    #         train_duration = train_end - train_start
    #
    #         # Evaluation
    #         y_pred = model.predict(X)
    #         acc = accuracy_score(y, y_pred)
    #
    #         # Wandb logging
    #         model_name = f'SVM_{time.strftime("%d_%b_%Y_%H_%M_%S", time.localtime())}'
    #         config = {'kernel': kernel, 'C': C, 'gamma': gamma, 'model': 'SVM'}
    #         wandb.init(project=project_name, name=model_name, config=config)
    #         agent_reward_list = evaluate_agent(model, env, sklearn_predict_action)
    #         wandb.log({'training_time': train_duration, 'train_acc': acc, 'test_mean_reward': mean(agent_reward_list),
    #                    'compared_to_rl': compare_agents_performance(rl_agent_mean, mean(agent_reward_list))})
    #         wandb.finish()
    #         if mean(agent_reward_list) > best_performance:
    #             best_performance = mean(agent_reward_list)
    #             best_model = model
    # # Save video
    # record_videos(best_model, env, sklearn_predict_action, log_dir=video_dir, num_episodes=n_videos, prefix='SVM-')

    # Decision Tree
    cls = DecisionTreeClassifier(max_depth=4, criterion='gini', random_state=0)
    model = Pipeline([
        ('scaler', MinMaxScaler()),
        ('clf', cls)])
    model.fit(X.values, y.values)
    y_pred = model.predict(X.values)
    acc = accuracy_score(y, y_pred)
    tree.plot_tree(cls)

    # Evaluate Behavior Cloning agent on the RL environment
    reward_list = evaluate_agent(model, env, sklearn_predict_fn, num_episodes=10)
    average_reward = mean(reward_list)
    performance_compared_to_rl = compare_agents_performance(average_reward, rl_agent_mean)
    print(f"Decision Tree Agent:\nTraining accuracy = {acc}\nAverage reward = {average_reward}")
    print(f"Performance compared to RL: {performance_compared_to_rl:.2f}")

    # Feature Importances
    dt_feature_importances = cls.feature_importances_

    # Logistic regression
    cls = LogisticRegression(random_state=0)
    model = Pipeline([
        ('scaler', MinMaxScaler()),
        ('clf', cls)])
    model.fit(X.values, y.values)
    y_pred = model.predict(X.values)
    acc = accuracy_score(y, y_pred)

    # Evaluate Behavior Cloning agent on the RL environment
    reward_list = evaluate_agent(model, env, sklearn_predict_fn, num_episodes=10)
    average_reward = mean(reward_list)
    performance_compared_to_rl = compare_agents_performance(average_reward, rl_agent_mean)
    print(f"\nLogistic Regression Agent:\nTraining accuracy = {acc}\nAverage reward = {average_reward}")
    print(f"Performance compared to RL: {performance_compared_to_rl:.2f}")

    # Feature Importances
    logistic_feature_importances = cls.coef_

    # TODO plot decision tree and logistic importances

    # Execution Time
    end = time.perf_counter()
    print(f"\nExecution time = {end - start:.2f} second(s)")


if __name__ == '__main__':
    main()
