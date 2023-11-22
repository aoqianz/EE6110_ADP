import os
import time
from typing import Union, Optional
import numpy as np
import pandas as pd
import gym
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm


class SaveModelAndEvalCallback(BaseCallback):
    def __init__(
            self,
            eval_env: Union[gym.Env, VecEnv],
            label: str = 'eval',
            log_path: str = None,
            n_eval_episodes: int = 5,
            eval_freq: int = 10000,
            algo_name: str = None,
            *args, **kwargs
    ):
        super(SaveModelAndEvalCallback, self).__init__(*args, **kwargs)
        self.eval_env = eval_env
        self.n_calls = 0
        self.eval_freq = eval_freq
        self.log_path = log_path
        self.algo_name = algo_name
        self.n_eval_episodes = n_eval_episodes
        self.start_time = 0
        self.time_usage = 0
        self.timesteps_list, self.time_usage_list, self.reward_mean_list, self.reward_std_list = [], [], [], []


    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        self.start_time = time.time()


    def _on_step(self):
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            batch_end_time = time.time()
            batch_time_usage = batch_end_time - self.start_time
            self.time_usage += batch_time_usage
            self.time_usage_list.append(self.time_usage)
            self.start_time = time.time()

            self.model.save(os.path.join(self.log_path, 'model_' + str(self.n_calls)))
            if isinstance(self.model, OffPolicyAlgorithm): self.model.save_replay_buffer(
                os.path.join(self.log_path, 'experience'))
            self.eval_env.reset()
            reward_mean, reward_std = evaluate_policy(self.model, self.eval_env, n_eval_episodes= self.n_eval_episodes)
            self.reward_mean_list.append(reward_mean)
            self.reward_std_list.append(reward_std)
            self.timesteps_list.append(self.n_calls)

            # save evaluate metric and model
            result_df = pd.DataFrame(columns=['algo', 'timesteps', 'time_usage', 'reward_mean', 'reward_std'])
            result_df['algo'] = [self.algo_name] * int(self.n_calls/self.eval_freq)
            result_df['timesteps'] = self.timesteps_list
            result_df['time_usage'] = self.time_usage_list
            result_df['reward_mean'] = self.reward_mean_list
            result_df['reward_std'] = self.reward_std_list
            resulf_filename = os.path.join(self.log_path,
                                           str(self.algo_name) + '.csv')
            result_df.to_csv(resulf_filename, index=True)
        return True