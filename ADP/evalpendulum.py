
from my_pendulum import Pendulum
from my_pendulum_gs import PendulumGS
import os
import gym
import numpy as np
import matplotlib.pyplot as plt
import sys

from stable_baselines3 import PPO, TD3, SAC, DDPG, A2C
from sb3_contrib import TRPO
from scipy.stats import sem, t
from numpy import mean


def main(ARGS):

    num_trials = 1
    num_timesteps = 500
    theta = np.zeros((num_trials, num_timesteps))
    theta_dot = np.zeros((num_trials, num_timesteps))
    control = np.zeros((num_trials, num_timesteps))

    env = Pendulum()
    controller = SAC.load(os.path.join("pendulum", "SAC"), env=env, custom_objects={'observation_space': env.observation_space, 'action_space': env.action_space})
    for episode in range(num_trials):
        obs, info = env.reset()
        for i in range(num_timesteps):
            action, _ = controller.predict(obs, state=None, deterministic=True)
            # action = controller.cal_gain2(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            theta[episode, i] = obs[0]
            theta_dot[episode, i] = obs[1]
            control[episode, i] = env.last_u
            # print(episode_score)
            if terminated or truncated:
                obs, info = env.reset()
                if terminated:
                    print("stopped by termination")
                if truncated:
                    print("stopped by truncation")
                break
    theta_mean = mean(theta, axis=0)
    # Standard error of mean
    theta_errors = sem(theta, axis=0)
    # Calculate the t-multiplier for 95% confidence interval
    confidence = 0.95
    degrees_freedom = num_trials - 1
    t_multiplier = t.ppf((1 + confidence) / 2, degrees_freedom)
    # Calculate the confidence intervals
    theta_confidence_intervals = theta_errors * t_multiplier

    theta_dot_mean = mean(theta_dot, axis=0)
    # Standard error of mean
    theta_dot_errors = sem(theta_dot, axis=0)
    # Calculate the t-multiplier for 95% confidence interval
    confidence = 0.95
    degrees_freedom = num_trials - 1
    t_multiplier = t.ppf((1 + confidence) / 2, degrees_freedom)
    # Calculate the confidence intervals
    theta_dot_confidence_intervals = theta_dot_errors * t_multiplier

    control_mean = mean(control, axis=0)
    # Standard error of mean
    control_errors = sem(control, axis=0)
    # Calculate the t-multiplier for 95% confidence interval
    confidence = 0.95
    degrees_freedom = num_trials - 1
    t_multiplier = t.ppf((1 + confidence) / 2, degrees_freedom)
    # Calculate the confidence intervals
    control_confidence_intervals = control_errors * t_multiplier

    # Plot the mean and confidence interval
    plt.subplot(1, 3, 1)
    plt.fill_between(range(num_timesteps), theta_mean - theta_confidence_intervals, theta_mean + theta_confidence_intervals, color='b',
                     alpha=0.2)
    plt.plot(theta_mean, 'b', label='theta')
    plt.title('theta')
    plt.xlabel('Time step')
    plt.ylabel('rad')
    # plt.legend()

    plt.subplot(1, 3, 2)
    plt.fill_between(range(num_timesteps), theta_dot_mean - theta_dot_confidence_intervals, theta_dot_mean + theta_dot_confidence_intervals,
                     color='r',
                     alpha=0.2)
    plt.plot(theta_dot_mean, 'r', label='theta_d')
    plt.title('theta_dot')
    plt.xlabel('Time step')
    plt.ylabel('rad/s')

    plt.subplot(1, 3, 3)
    plt.fill_between(range(num_timesteps), control_mean - control_confidence_intervals,
                     control_mean + control_confidence_intervals,
                     color='g',
                     alpha=0.2)
    plt.plot(control_mean, 'g', label='control')
    plt.title('control input')
    plt.xlabel('Time step')
    plt.ylabel('Nm')
    plt.show()



if __name__ == "__main__":
    main(sys.argv)