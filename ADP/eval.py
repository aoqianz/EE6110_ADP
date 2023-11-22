from my_cartpole_real import CartPoleReal
from my_cartpole_gs import CartPoleGS
from my_pendulum import Pendulum
from my_cartpole_double import CartpoleDouble
from my_pendulum_gs import PendulumGS
import os
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import sys
import argparse
from pid import PID, PID2
from stable_baselines3 import PPO, TD3, SAC, DDPG, A2C
from sb3_contrib import TRPO

def run_evaluation(env, agent, num_episode, steps):
    for episode in range(num_episode):
        obs, info = env.reset()
        episode_score = 0
        for i in range(steps):
            action = agent.cal_gain2(obs)

            obs, reward, terminated, truncated, info = env.step(action)

            episode_score += reward
            if terminated or truncated:
                obs, info = env.reset()
                if terminated:
                    print("stopped by termination")
                if truncated:
                    print("stopped by truncation")
                break


def run_evaluation_rl(env, agent, num_episode, steps):
    for episode in range(num_episode):
        obs, info = env.reset()
        episode_score = 0
        for i in range(steps):
            action, _ = agent.predict(obs, state=None, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_score += reward
            # print(episode_score)
            if terminated or truncated:
                obs, info = env.reset()
                if terminated:
                    print("stopped by termination")
                if truncated:
                    print("stopped by truncation")
                break

def main(ARGS):
    #### Define and parse (optional) arguments for the script ##########################################
    parser = argparse.ArgumentParser(description='Testing of RL algorithms')
    parser.add_argument('--Control', default='SAC', type=str, help='Controller name (default: PID)')
    parser.add_argument('--run_name', default='untitled', type=str, help='Name for video files, frames, log files, etc.')
    parser.add_argument('--eval_episodes', default=1, type=int, help='Number of test simulations (default: 1)')
    parser.add_argument('--agent_name', default="PID", type=str, help='Name of the agent to be loaded (default: None')
    parser.add_argument('--env', default="cheetah", type=str, help='Simulation environment', metavar='')
    parser.add_argument('--episode_len', default=2000, type=int, help='Duration of the simulation in steps', metavar='')

    ARGS = parser.parse_args()


    if ARGS.env == "cartpole_real":
        env = CartPoleReal(render_mode="human")
    elif ARGS.env == "cartpole_double":
        env = CartpoleDouble(render_mode="human")
    elif ARGS.env == "cartpole_gs":
        env = CartPoleGS(render_mode="human")
    elif ARGS.env == "pendulum":
        env = Pendulum(render_mode="human")
    elif ARGS.env == "pendulumgs":
        env = PendulumGS(render_mode="human")
    elif ARGS.env == "cheetah":
        env = gym.make("HalfCheetah-v4",render_mode = "human")

    if ARGS.Control == "PID" and ARGS.env == "pendulum":
        controller = PID2(50, 2, 10)
    elif ARGS.Control == "PID":
        controller = PID(80, 0.5, 150, 3.5, 0.001, 0.15)
    elif ARGS.Control == "PPO":
        controller = PPO.load(os.path.join("..", "model", "ppo", "PPO"))
    elif ARGS.Control == "TRPO":
        controller = TRPO.load(os.path.join("..", "model", "trpo", "TRPO"))
    elif ARGS.Control == "DDPG":
        controller = DDPG.load(os.path.join("..", "model", "ddpg", "DDPG"))
    elif ARGS.Control == "SAC":
        controller = SAC.load(os.path.join("..", "model", "sac", "SAC"))
    elif ARGS.Control == "A2C":
        controller = A2C.load(os.path.join("..", "model", "a2c", "A2C"))
    elif ARGS.Control == "TD3":
        controller = TD3.load(os.path.join("..", "model", "td3", "TD3"))
    else:
        controller = DDPG.load("ddpg_pendulum")

    if ARGS.Control == "PID":
        run_evaluation(env,controller,ARGS.eval_episodes,ARGS.episode_len)
    else:
        run_evaluation_rl(env, controller, ARGS.eval_episodes, ARGS.episode_len)


if __name__ == "__main__":
    main(sys.argv)