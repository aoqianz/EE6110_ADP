
from my_cartpole_real import CartPoleReal
from my_cartpole_gs import CartPoleGS
import os
import gym
import numpy as np
import matplotlib.pyplot as plt
import sys
import argparse

from stable_baselines3 import PPO, TD3, SAC, DDPG, A2C
from sb3_contrib import TRPO
from scipy.stats import sem, t
from numpy import mean


def main(ARGS):
    num_trials = 4
    num_timesteps = 2000
    theta = np.zeros((num_trials, num_timesteps))
    theta_dot = np.zeros((num_trials, num_timesteps))
    x = np.zeros((num_trials, num_timesteps))
    x_dot = np.zeros((num_trials, num_timesteps))
    control = np.zeros((num_trials, num_timesteps))

    env = CartPoleReal()

    obs, info = env.reset()
    controller = TRPO.load(os.path.join("cartpole", "TRPO"))
    for i in range(num_timesteps):
        action, _ = controller.predict(obs, state=None, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        x[0, i] = obs[0]
        x_dot[0, i] = obs[1]
        theta[0, i] = obs[2]
        theta_dot[0, i] = obs[3]
        control[0, i] = action

        if terminated or truncated:
            obs, info = env.reset()
            if terminated:
                print("stopped by termination")
            if truncated:
                print("stopped by truncation")
            break
    obs, info = env.reset()
    controller = PPO.load(os.path.join("cartpole", "PPO"))
    for i in range(num_timesteps):
        action, _ = controller.predict(obs, state=None, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        x[1, i] = obs[0]
        x_dot[1, i] = obs[1]
        theta[1, i] = obs[2]
        theta_dot[1, i] = obs[3]
        control[1, i] = action

        if terminated or truncated:
            obs, info = env.reset()
            if terminated:
                print("stopped by termination")
            if truncated:
                print("stopped by truncation")
            break
    obs, info = env.reset()
    controller = DDPG.load(os.path.join("cartpole", "DDPG"))
    for i in range(num_timesteps):
        action, _ = controller.predict(obs, state=None, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        x[2, i] = obs[0]
        x_dot[2, i] = obs[1]
        theta[2, i] = obs[2]
        theta_dot[2, i] = obs[3]
        control[2, i] = action

        if terminated or truncated:
            obs, info = env.reset()
            if terminated:
                print("stopped by termination")
            if truncated:
                print("stopped by truncation")
            break
    obs, info = env.reset()
    controller = SAC.load(os.path.join("cartpole", "SAC"))
    for i in range(num_timesteps):
        action, _ = controller.predict(obs, state=None, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        x[3, i] = obs[0]
        x_dot[3, i] = obs[1]
        theta[3, i] = obs[2]
        theta_dot[3, i] = obs[3]
        control[3, i] = action

        if terminated or truncated:
            obs, info = env.reset()
            if terminated:
                print("stopped by termination")
            if truncated:
                print("stopped by truncation")
            break

    # Plot the mean and confidence interval
    plt.subplot(2, 2, 1)
    plt.plot(x[0], 'b', label='TRPO')
    plt.plot(x[1], 'g', label='PPO')
    plt.plot(x[2], 'r', label='DDPG')
    plt.plot(x[3], 'y', label='SAC')
    plt.title('x')
    plt.xlabel('Time step')
    plt.ylabel('m')
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(x_dot[0], 'b', label='TRPO')
    plt.plot(x_dot[1], 'g', label='PPO')
    plt.plot(x_dot[2], 'r', label='DDPG')
    plt.plot(x_dot[3], 'y', label='SAC')

    plt.title('x_dot')
    plt.xlabel('Time step')
    plt.ylabel('m/s')
    plt.legend()


    plt.subplot(2, 2, 3)
    plt.plot(theta[0], 'b', label='TRPO')
    plt.plot(theta[1], 'g', label='PPO')
    plt.plot(theta[2], 'r', label='DDPG')
    plt.plot(theta[3], 'y', label='SAC')

    plt.title('theta')
    plt.xlabel('Time step')
    plt.ylabel('rad')
    plt.legend()


    plt.subplot(2, 2, 4)
    plt.plot(theta_dot[0], 'b', label='TRPO')
    plt.plot(theta_dot[1], 'g', label='PPO')
    plt.plot(theta_dot[2], 'r', label='DDPG')
    plt.plot(theta_dot[3], 'y', label='SAC')

    plt.title('theta_dot')
    plt.xlabel('Time step')
    plt.ylabel('rad/s')
    plt.legend()

    plt.suptitle('RL controllers on Cartpole', fontsize=13)
    plt.show()



if __name__ == "__main__":
    main(sys.argv)