from os.path import dirname, abspath
from os import environ
# Needed because libiomp5md error gets thrown 
# https://stackoverflow.com/questions/64209238/error-15-initializing-libiomp5md-dll-but-found-libiomp5md-dll-already-initial
environ['KMP_DUPLICATE_LIB_OK']='True'
import sys
sys.path.append(dirname(dirname(abspath(__file__))))
import inspect

from CybORG import CybORG
from CybORG.Agents import B_lineAgent
from CybORG.Agents.Wrappers import ChallengeWrapper

import ray
from ray import tune
import ray.rllib.agents.ppo as ppo
from ray.tune.logger import pretty_print
from ray.tune.registry import register_env
from ray.rllib.agents.ppo import PPOTrainer


def cyborg(env_config):
    path = str(inspect.getfile(CybORG))
    path = path[:-10] + '/Shared/Scenarios/Scenario1b.yaml'
    cyborg = CybORG(path, 'sim', agents={'Red': B_lineAgent})
    return ChallengeWrapper(env=cyborg, agent_name='Blue')

register_env("cyborg", cyborg)

# Create our RLlib Trainer.
trainer = PPOTrainer(config={
    "env": "cyborg",
    "num_workers": 2,
    "framework": "torch",
    "timesteps_per_iteration": 100
})

for i in range(1000):
   # Perform one iteration of training the policy with PPO
   result = trainer.train()
   print(pretty_print(result))

   if i % 200 == 0:
       checkpoint = trainer.save()
       print("checkpoint saved at", checkpoint)

# ray.init()
# tune.run(
#     "PPO",
#     stop={"episode_reward_mean": 200},
#     config={
#         "env": "cyborg",
#         "num_gpus": 1,
#         "num_workers": 1,
#         "lr": tune.grid_search([0.01, 0.001, 0.0001]),
#     },
# )