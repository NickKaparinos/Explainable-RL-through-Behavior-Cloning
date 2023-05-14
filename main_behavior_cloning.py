"""
Nikos Kaparinos 119 - Julio Hamiti 106
2023
"""

import os
import pickle
import time
from statistics import mean

import gymnasium as gym
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from stable_baselines3 import PPO

from utilities import *


def main():
    """ Train behavior cloning models """
    start = time.perf_counter()
    env_id = 'LunarLander-v2'
    saved_model = './logs/LunarLander/PPO_09_Nov_2022_08_40_24/model.zip'
    log_dir = f'logs/{env_id[:-3]}/'
    os.makedirs(log_dir, exist_ok=True)
    n_dataset, n_videos, n_evaluation_episodes = 10_000, 1, 100
    sns.set()

    # Load dataset
    dataset_name = f'dataset_{n_dataset}.pickle'
    with open(f'{log_dir}/{dataset_name}', 'rb') as f:
        dataset = pickle.load(f)
    dataset, action_names = add_feature_names(dataset, env_id)
    X, y = dataset.iloc[:, :-1], dataset.iloc[:, -1]

    # RL Agent evaluation
    env = gym.make(env_id)
    model = PPO.load(saved_model)
    rl_agent_reward_list = evaluate_agent(model, env, ppo_predict_fn, num_episodes=n_evaluation_episodes)
    rl_agent_mean = mean(rl_agent_reward_list)

    # Save video
    record_videos(model, env, ppo_predict_fn, log_dir=log_dir, num_episodes=n_videos, prefix='RL-')

    # Decision Tree Behavior Cloning
    results_dict = pd.DataFrame(columns=[f'Decision Tree {i}' for i in range(1, 5)] + ['Logistic Regression'],
                                index=['Performance compared to RL'])
    for max_depth in range(1, 5):
        cls = DecisionTreeClassifier(max_depth=max_depth, criterion='gini', random_state=0)
        model = Pipeline([
            ('scaler', MinMaxScaler()),
            ('clf', cls)])
        model.fit(X.values, y.values)
        y_pred = model.predict(X.values)
        acc = accuracy_score(y, y_pred)

        # Save video
        record_videos(model, env, sklearn_predict_fn, log_dir=log_dir, num_episodes=n_videos,
                      prefix=f'DT-{max_depth}-')

        # Plot and save decision tree
        tree.plot_tree(cls, feature_names=dataset.columns[:-1], filled=True, class_names=action_names)
        plt.savefig(f'{log_dir}dt-{max_depth}-tree.png', dpi=125 * max_depth)

        # Evaluate Behavior Cloning agent on the RL environment
        reward_list = evaluate_agent(model, env, sklearn_predict_fn, num_episodes=n_evaluation_episodes)
        average_reward = mean(reward_list)
        performance_compared_to_rl = compare_agents_performance(rl_agent_mean, average_reward)
        print(f"Decision Tree Agent:\nTraining accuracy = {acc}\nAverage reward = {average_reward}")
        print(f"Performance compared to RL: {performance_compared_to_rl:.2f}")
        results_dict.iloc[0, max_depth - 1] = performance_compared_to_rl

        # Feature Importance Barplot
        dt_feature_importances = cls.feature_importances_
        plt.figure(max_depth, figsize=(10, 10))
        plt.clf()
        sns.barplot(x=dataset.columns[:-1], y=dt_feature_importances)
        # plt.xticks(rotation=45)
        plt.title(f'Decision Tree (max depth = {max_depth}) feature importances', fontdict={'size': 14})
        plt.xlabel('Features')
        plt.ylabel('Feature importance')
        plt.savefig(f'{log_dir}dt-importances-{max_depth}.png', dpi=150)

    # Logistic Regression Behavior Cloning
    cls = LogisticRegression(random_state=0)
    model = Pipeline([
        ('scaler', MinMaxScaler()),
        ('clf', cls)])
    model.fit(X.values, y.values)
    y_pred = model.predict(X.values)
    acc = accuracy_score(y, y_pred)

    # Save video
    record_videos(model, env, sklearn_predict_fn, log_dir=log_dir, num_episodes=n_videos,
                  prefix='logistic-')

    # Evaluate Behavior Cloning agent on the RL environment
    reward_list = evaluate_agent(model, env, sklearn_predict_fn, num_episodes=n_evaluation_episodes)
    average_reward = mean(reward_list)
    performance_compared_to_rl = compare_agents_performance(rl_agent_mean, average_reward)
    print(f"\nLogistic Regression Agent:\nTraining accuracy = {acc}\nAverage reward = {average_reward}")
    print(f"Performance compared to RL: {performance_compared_to_rl:.2f}")
    results_dict.iloc[0, -1] = performance_compared_to_rl

    # Feature Importance Barplot
    # logistic_feature_importances = np.absolute(cls.coef_) / np.sum(np.absolute(cls.coef_))
    # plt.figure(5, figsize=(10, 10))
    # sns.barplot(x=dataset.columns[:-1], y=logistic_feature_importances.reshape(-1, ))
    # # plt.xticks(rotation=45)
    # plt.title(f'Logistic regression feature importances', fontdict={'size': 14})
    # plt.xlabel('Features')
    # plt.ylabel('Feature importance')
    # plt.savefig(f'{log_dir}logistic-importances.png', dpi=150)

    # Plot Agent performance results
    plt.figure(6, figsize=(10, 10))
    sns.barplot(results_dict)
    plt.title(f'Behavior cloning agent performance compared to RL', fontdict={'size': 16})
    plt.xlabel('Agent')
    plt.ylabel('Performance compared to RL')
    plt.savefig(f'{log_dir}performance-comparison.png', dpi=150)

    # Execution Time
    end = time.perf_counter()
    print(f"\nExecution time = {end - start:.2f} second(s)")


if __name__ == '__main__':
    main()
