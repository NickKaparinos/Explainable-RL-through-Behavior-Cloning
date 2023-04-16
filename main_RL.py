"""
Nikos Kaparinos 119 - Julio Hamiti 106
2023
"""
import time
from stable_baselines3 import PPO, DQN
from wandb.integration.sb3 import WandbCallback

import wandb
from utilities import *


def main():
    """ Train Proximal Policy Agent """
    # Environment       'CartPole-v1' 'MountainCar-v0' 'Acrobot-v1' 'LunarLander-v2'
    start = time.perf_counter()
    env_id = 'CartPole-v1'
    env = gym.make(env_id)

    if 'MountainCar' in env_id:
        env._max_episode_steps = 500

    # Logging
    config = {'policy_type': 'MlpPolicy', 'total_timesteps': 5_000, 'env_id': env_id}
    model_name = f'PPO_{time.strftime("%b_%d_%Y_%H_%M_%S", time.localtime())}'
    log_dir = f'logs/{env_id[:-3].replace("/", "-")}/{model_name}/'
    makedirs(log_dir, exist_ok=True)
    run = wandb.init(project=f'AML-RL-{env_id[:-3]}'.replace('/', '-'), entity="nickkaparinos", name=model_name,
                     config=config, sync_tensorboard=True)

    # Agent training
    model = PPO(config["policy_type"], env, verbose=0, tensorboard_log=f"logs/runs/{run.id}")
    model.learn(total_timesteps=config['total_timesteps'], callback=WandbCallback(verbose=2))
    model.save(f'{log_dir}/model.zip')
    run.finish()

    # Record video
    env = gym.make(env_id)
    record_videos(model, env, ppo_predict_fn, log_dir=log_dir, num_episodes=2)

    # Execution Time
    end = time.perf_counter()
    print(f"\nExecution time = {end - start:.2f} second(s)")


if __name__ == '__main__':
    main()
