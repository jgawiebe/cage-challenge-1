import inspect
import numpy as np
import datetime
import os
import tensorflow as tf

from stable_baselines3 import PPO, A2C, DQN, HER, DDPG, SAC, TD3
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common import results_plotter
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common.noise import NormalActionNoise

from CybORG import CybORG
from CybORG.Agents.BaseAgent import BaseAgent

class PPOAgent(BaseAgent):

    def __init__(self, env, model_name=None):
        if model_name is None:
            self.model = PPO('MlpPolicy', env, tensorboard_log= "./tb_logs/", verbose=1)
        else:
            print('Loading saved model.')
            self.load(model_name)

    def train(self, timesteps,log_name):
      self.model.learn(timesteps, tb_log_name=log_name)

    def get_action(self, observation, action_space):
        action, _states = self.model.predict(observation)
        print("Returning action...", action)
        return action

    def save(self, name = None):
        if name is None:
            name = "{}-{}".format(datetime.datetime.now(), self.__str__)
        self.model.save(name)

    def load(self, model_name):
        self.model = PPO.load(model_name, force_reset=True)

    def __str__(self):
        return self.__class__.__name__