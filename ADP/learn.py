import os
import datetime
import sys
from enum import Enum
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from torch import nn
from my_cartpole_real import CartPoleReal
from my_cartpole_gs import CartPoleGS
from my_pendulum import Pendulum
from my_pendulum_gs import PendulumGS
from my_cartpole_double import CartpoleDouble
import gymnasium as gym
from gym import error, spaces, utils
from gym.utils import seeding
from utils.callbacks import SaveModelAndEvalCallback
from stable_baselines3 import PPO, TD3, SAC, DDPG, A2C
from sb3_contrib import TRPO
from stable_baselines3.td3.policies import TD3Policy
from stable_baselines3.sac.policies import SACPolicy
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common import results_plotter
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from torch import nn


def setup_env(env_name):
    if env_name == "cartpole_real":
        env = CartPoleReal()
    elif env_name == "cartpole_gs":
        env = CartPoleGS()
    elif env_name == "cartpole_double":
        env = CartpoleDouble()
    elif env_name == "pendulum":
        env = Pendulum()
    elif env_name == "pendulumgs":
        env = PendulumGS()
    elif env_name == "cheetah":
        env = gym.make("HalfCheetah-v4")
    env = Monitor(env)
    return env

def setup_env2(env_name):
    if env_name == "cartpole_real":
        env = CartPoleReal()
    elif env_name == "cartpole_gs":
        env = CartPoleGS()
    elif env_name == "cartpole_double":
        env = CartpoleDouble()
    elif env_name == "pendulum":
        env = Pendulum()
    elif env_name == "pendulumgs":
        env = PendulumGS()
    elif env_name == "cheetah":
        env = gym.make("HalfCheetah-v4")
    env = Monitor(env)
    return env

def load_algo(algo_name, env):
    policy_kwargs = dict(activation_fn=nn.ReLU,
                     net_arch=dict(pi=[256, 256], vf=[256, 256]))
    if algo_name == 'PPO':
        agent = PPO('MlpPolicy', env, policy_kwargs = policy_kwargs, batch_size=512, verbose=1)
    elif algo_name == 'TRPO':
        agent = TRPO("MlpPolicy", env, policy_kwargs = policy_kwargs, batch_size=512, verbose=1)
    elif algo_name == 'DDPG':
        dim_actions = env.action_space.shape[-1]
        action_noise = NormalActionNoise(mean=np.zeros(dim_actions), sigma=0.1 * np.ones(dim_actions))
        agent = DDPG("MlpPolicy",  env, action_noise = action_noise, verbose=1)
    elif algo_name == 'SAC':
        agent = SAC(SACPolicy, env, verbose=1)
    elif algo_name == 'TD3':
        dim_actions = env.action_space.shape[-1]
        action_noise = NormalActionNoise(mean=np.zeros(dim_actions), sigma=0.1 * np.ones(dim_actions))
        agent = TD3("MlpPolicy", env, action_noise=action_noise, verbose=1)
    elif algo_name == 'A2C':
        agent = A2C("MlpPolicy", env, policy_kwargs = policy_kwargs, verbose=1, n_steps=2048, device="cpu")
    return agent


def main(ARGS):
    #### Define and parse (optional) arguments for the script ##########################################
    parser = argparse.ArgumentParser(description='Learning')
    # parser.add_argument('--Control', default='PID', type=str, help='Controller name (default: PID)')
    parser.add_argument('--run_name', default='untitled', type=str, help='Name for video files, frames, log files, etc.')
    parser.add_argument('--eval_episodes', default=10, type=int, help='Number of test simulations (default: 10)')
    parser.add_argument('--env', default="pendulumgs", type=str, help='Simulation environment', metavar='')
    parser.add_argument('--algo', default="PPO", type=str, help='Training Algorithm', metavar='')
    parser.add_argument('--total_timesteps', default=5000000, type=int, help='How many timesteps to perform during training (default: 5e6)')
    parser.add_argument('--eval_freq', default=50000, type=int, help='Number of timesteps between evaluations and model saves (default: 50000)')
    ARGS = parser.parse_args()

    learn_env = setup_env(ARGS.env)
    eval_env = setup_env2(ARGS.env)
    agent = load_algo(ARGS.algo, learn_env)

    log_path = os.path.join("..","logs", ARGS.run_name + "-" + datetime.datetime.now().strftime('%Y_%m_%d_%H_%M'))

    os.makedirs(log_path)

    callback = SaveModelAndEvalCallback(eval_freq=ARGS.eval_freq, eval_env=eval_env, log_path=log_path, n_eval_episodes=ARGS.eval_episodes,
                             algo_name=ARGS.algo)

    agent.learn(total_timesteps=ARGS.total_timesteps, callback=callback)

if __name__ == "__main__":
    main(sys.argv)