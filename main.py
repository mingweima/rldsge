import sys
import ray
import gym

gym.logger.set_level(40)

sys.path.append("..")

from source.envs.environment import WhitedBasicModel
from source.solvers.rl import RaySolver

if __name__ == "__main__":
    ray.init()
    env = WhitedBasicModel(env_config={"structural_params": {}, "env_params": {}})
    solver = RaySolver(env=env, solver_params={"verbose": True, "episodes": 30})
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
        print(obs, reward, done, info)
