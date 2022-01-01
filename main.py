import sys

import gym
import ray
from ray.rllib.agents import ppo, a3c, cql, ddpg

gym.logger.set_level(40)

sys.path.append("..")

from source.envs.environment import WhitedBasicModel
from source.solvers.ray_solver import RaySolver

# A3C_Trainer = a3c.A3CTrainer
PPO_Trainer = ppo.PPOTrainer
# CQL_Trainer = cql.CQLTrainer
DDPG_Trainer = ddpg.DDPGTrainer

if __name__ == "__main__":
    ray.init()
    env = WhitedBasicModel(env_config={"structural_params": {}, "env_params": {}})
    solver = RaySolver(env=env,
                       trainer=PPO_Trainer,
                       solver_params={"verbose": True, "episodes": 30,
                                      "trainer_config": {
                                          "num_workers": 3,
                                          "gamma": env.structural_params.get("gamma", 0.99),
                                      }
                                      })
    solver.train()
    ray.shutdown()
    # run until episode ends
    episode_reward = 0
    done = False
    obs = env.reset()
    while not done:
        action = solver.act(obs)
        obs, reward, done, info = env.step(action)
        episode_reward += reward
        print(obs, action, reward, done, info)
