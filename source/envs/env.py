from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Dict

import gym
import numpy as np

from source.utils.math_utils import log_ar1

gym.logger.set_level(40)


class StructuralModel(gym.Env, ABC):
    @abstractmethod
    def __init__(self, env_config: dict = None):
        if env_config is None:
            env_config = {}
        self.env_config = env_config
        assert "structural_params" in env_config  # structural params is a grid {param: [min, max]} if mutable,
        # and is {param: value if not mutable}
        assert "env_params" in env_config
        assert "is_mutable" in env_config

        structural_params, env_params = env_config['structural_params'], env_config['env_params']
        self.structural_params = structural_params  # structural params to be estimated

        self.env_params = env_params  # other params related to the environment; must contain num of structural params
        assert "num_structural_params" in self.env_params
        assert "action_size" in self.env_params
        assert "obs_size" in self.env_params

        assert isinstance(env_config["is_mutable"], bool)
        self.is_mutable = env_config['is_mutable']
        self.allow_resample_param = True
        # set current_structural_params. if mutable we need to use OrderedDict to make sure observations are in order
        self.current_structural_params = OrderedDict() if self.is_mutable else self.structural_params

    @abstractmethod
    def reset(self) -> np.ndarray:
        """Reset environment like in gym"""

    @abstractmethod
    def step(self, action: np.ndarray, resample_param: bool = False) -> (np.ndarray, float, bool, dict):
        assert not (not self.is_mutable) and resample_param
        """Step using action like in gym"""

    def sample_structural_params(self) -> None:
        """generate uniform samples from grid"""
        assert self.is_mutable
        assert self.allow_resample_param
        for struct_param, grid in self.structural_params.items():
            self.current_structural_params[struct_param] = np.random.uniform(low=grid[0], high=grid[1], size=None)

    def set_structural_params(self, param_dict: Dict[str, float]):
        """fix param dict for a mutable env for testing"""
        assert self.is_mutable
        self.current_structural_params = param_dict
        self.allow_resample_param = False

    def structural_params_to_array(self) -> np.ndarray:
        """convert OrderedDict into np array"""
        assert self.is_mutable
        return np.array(list(self.current_structural_params.values()))


class WhitedBasicModel(StructuralModel, ABC):
    """Basic Model in Strebulaev and Whited (2012), *Dynamic Models and Structural Estimation*"""

    def __init__(self, env_config=None):
        if env_config is None:
            env_config = {}
        assert "structural_params" in env_config
        assert "env_params" in env_config
        assert "is_mutable" in env_config
        structural_params, env_params, is_mutable = env_config['structural_params'], env_config['env_params'], \
                                                    env_config['is_mutable']
        # # Set default value of structural params
        # structural_params.setdefault('gamma', 0.98)  # discount rate
        # structural_params.setdefault('delta', 0.15)  # depreciation of capital
        # structural_params.setdefault('theta', 0.7)  # pi = z * k**theta
        # structural_params.setdefault('rho', 0.7)  # persistence of the shock process
        # structural_params.setdefault('sigma', 0.15)  # standard deviation of the shock process

        # Set default value of env params
        env_params.setdefault('seed', None)  # random seed for shocks
        env_params.setdefault('initial_state', (1., 1.))  # (k_0, z_0) time 0 capital and shock
        env_params.setdefault('psi_func', lambda i, k: 0.)  # investment adjustment cost function psi(k, z)
        env_params.setdefault('pi_func', lambda k, z, theta: z * (
                k ** theta))  # profit function pi(k, z) = z*k**theta
        env_params.setdefault('z_proc_func', log_ar1)  # z process: z(z0, rho, sigma)
        env_params.setdefault('num_structural_params', 5)
        env_params.setdefault('action_size', 1)
        obs_size = 7 if is_mutable else 2
        env_params.setdefault('obs_size', obs_size)
        env_params.setdefault('max_steps', 100)
        env_params.setdefault('discrete_action_size', 20)

        super().__init__({"structural_params": structural_params, "env_params": env_params, "is_mutable": is_mutable})

        self.action_space = gym.spaces.Discrete(env_params['discrete_action_size'])

        if self.is_mutable:
            # 1D array (k, z, gamma, delta, theta, rho, sigma)
            self.observation_space = gym.spaces.Box(np.zeros((7,)),
                                                    np.full((7,), np.inf),
                                                    dtype=np.float32)
        else:
            # Observation: 1D array (k, z)
            self.observation_space = gym.spaces.Box(np.array([0., 0.]), np.array([np.inf, np.inf]), dtype=np.float32)

        self.state = None
        self.steps = 0
        self.done = False

    def reset(self) -> np.ndarray:
        k0, z0 = self.env_params['initial_state']
        if self.is_mutable:
            if self.allow_resample_param:
                self.sample_structural_params()
            struct_param_array = self.structural_params_to_array()
            self.state = np.concatenate((np.array([k0, z0], dtype=np.float32), struct_param_array), axis=0).flatten()
        else:
            self.state = np.array([k0, z0], dtype=np.float32)
        self.steps = 0
        self.done = False
        return self.state

    def step(self, action, resample_param=True):  # action is investment I_t over capital k_t
        k_curr = self.state[0]
        z_curr = self.state[1]
        action = int(action)
        i_curr = action * k_curr / (self.env_params['discrete_action_size'] - 1.)
        # update z
        z_new = self.env_params['z_proc_func'](z_curr, self.current_structural_params['rho'],
                                               self.current_structural_params['sigma'])
        # update k
        k_new = (1 - self.current_structural_params['delta']) * k_curr + i_curr
        # reward function
        reward = self.env_params['pi_func'](k_curr, z_curr, self.current_structural_params['theta']) - self.env_params[
            'psi_func'](i_curr, k_curr) - i_curr
        assert isinstance(reward, float)
        # new state
        if self.is_mutable:
            if resample_param:
                self.sample_structural_params()
            struct_param_array = self.structural_params_to_array()
            self.state = np.concatenate((np.array([k_new, z_new], dtype=np.float32), struct_param_array),
                                        axis=0).flatten()
        else:
            self.state = np.array([k_new, z_new], dtype=np.float32)
        # add step
        self.steps += 1
        # decide to end or not
        if self.steps == self.env_params['max_steps']:
            self.done = True
        return self.state, reward, self.done, {}

    def render(self, mode="human"):
        print(f"Capital k={self.state[0]}")
