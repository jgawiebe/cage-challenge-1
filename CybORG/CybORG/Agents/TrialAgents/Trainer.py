import inspect
import matplotlib.pyplot as plt
import numpy as np
import os

from stable_baselines3 import PPO

from CybORG import CybORG
from CybORG.Agents import B_lineAgent, SleepAgent
from CybORG.Agents.BaseAgent import BaseAgent
from CybORG.Agents.SimpleAgents.RLAgent import RLAgent
from CybORG.Agents.SimpleAgents.TestAgent import TestAgent
from CybORG.Agents.TrialAgents.PPOAgent import PPOAgent

from CybORG.Agents.Wrappers.EnumActionWrapper import EnumActionWrapper
from CybORG.Agents.Wrappers.FixedFlatWrapper import FixedFlatWrapper
from CybORG.Agents.Wrappers.OpenAIGymWrapper import OpenAIGymWrapper
from CybORG.Agents.Wrappers.ReduceActionSpaceWrapper import ReduceActionSpaceWrapper
from CybORG.Agents.Wrappers import ChallengeWrapper

cyborg_version = '1.2'
scenario = 'Scenario1b'

def challenge_wrap(env):
    return ChallengeWrapper('Blue', env)

def wrap( env):
    return ChallengeWrapper('Blue', env)

def train_model(scenario):
    cyborg_version = '1.2'
    scenario = 'Scenario1b'
    agent_name = 'Blue'
    timesteps = 10000
    #steps = round(timesteps/1000000, 2)

    lines = inspect.getsource(wrap)
    wrap_line = lines.split('\n')[1].split('return ')[1]

    path = str(inspect.getfile(CybORG))
    path = path[:-10] + '/Shared/Scenarios/Scenario1b.yaml'

    red_agent = B_lineAgent

    #cyborg = OpenAIGymWrapper(
    #    agent_name,
    #    EnumActionWrapper(
    #        FixedFlatWrapper(
    #            ReduceActionSpaceWrapper(
    #                CybORG(path, 'sim', agents={'Red': red_agent})
    #            )
    #        )
    #    )
    #)

    # if evaluating for CAGE challenge
    cyborg = ChallengeWrapper(
        agent_name,
        CybORG(path, 'sim', agents={'Red': red_agent})
    )

    model = TestAgent()
    #model = PPOAgent(env=cyborg)
    #model = RLAgent(env=cyborg, agent_type="PPO")

    model.train(timesteps=int(timesteps), log_name = f"{model.__class__.__name__}")

    model.save(f"{model.__class__.__name__} against {red_agent.__name__}")

if __name__ == "__main__":
    train_model(scenario)