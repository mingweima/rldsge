import sys
from typing import Type, Dict

import gym
import numpy as np

gym.logger.set_level(40)

from ray.rllib.agents import ppo
from ray.tune.registry import register_env
from ray.tune.logger import pretty_print
from ray.rllib.agents.trainer import Trainer

sys.path.append("..")

from source.solvers.solver import Solver
from source.envs.env import StructuralModel


class RaySolver(Solver):
    """Generic solver using RLlib/Ray algorithms"""

    def __init__(self, env: StructuralModel = None, solver_params: dict = None,
                 trainer: Type[Trainer] = ppo.PPOTrainer):
        super().__init__(env, solver_params)

        self.trainer_config = solver_params.get("trainer_config", {})

        def env_creator(env_config):
            env.__init__(env_config=env_config)
            return env

        register_env("my-env", env_creator)
        config = {
            # config to pass to env class
            "env_config": {"structural_params": env.structural_params, "env_params": env.env_params,
                           "is_mutable": env.is_mutable},
            "framework": "torch",
        }
        config.update(self.trainer_config)
        trainer_instance = trainer(env="my-env", config=config)  # create an instance of Trainer
        self.trainer = trainer_instance

    def act(self, obs: np.ndarray, evaluation: bool = True) -> np.ndarray:
        return self.trainer.compute_single_action(obs)

    def sample(self, env: StructuralModel = None, param_dict: Dict[str, float] = None) -> np.ndarray:
        """Sample a single observation paxth
        :return: obs of size (N, T), where N is the dim of obs, T is eps length """
        env = self.env if not env else env
        if param_dict is not None:
            env.set_structural_params(param_dict)
        done = False
        obs_step = env.reset()
        obs = None
        while not done:
            if obs is None:
                obs = obs_step.reshape(-1, 1)
            action = self.act(obs_step)
            obs_step, _, done, _ = env.step(action, resample_param=False)
            obs = np.concatenate((obs, obs_step.reshape(-1, 1)), axis=1)
        return obs

    def train(self, training_params: dict = None) -> None:
        episodes = self.solver_params.get("episodes", 50)
        save_freq = self.solver_params.get("save_freq", episodes - 1)
        print_freq = self.solver_params.get("print_freq", episodes // 10)
        verbose = self.solver_params.get("verbose", False)
        for i in range(episodes):
            # Perform one iteration of training the policy with the method
            result = self.trainer.train()
            if verbose and episodes % print_freq == 0:
                print(pretty_print(result))

            if i > 0 and i % save_freq == 0:
                checkpoint = self.trainer.save()
                print("checkpoint saved at", checkpoint)
