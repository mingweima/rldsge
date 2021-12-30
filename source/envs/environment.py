from abc import ABC, abstractmethod

import gym
import numpy as np

from ..utils.math_utils import log_ar1


class StructuralModel(gym.Env, ABC):
    @abstractmethod
    def __init__(self, structural_params: dict = None, env_params: dict = None):
        self.structural_params = structural_params  # structural params to be estimated
        self.env_params = env_params  # other params related to the environment; must contain num of structural params
        assert "num_structural_params" in self.env_params

    @abstractmethod
    def reset(self):
        """Reset environment like in gym"""

    @abstractmethod
    def step(self, action):
        """Step using action like in gym"""

    def update_structural_params(self, structural_params):
        self.structural_params = structural_params


class VRToyModel(StructuralModel, ABC):
    """Toy Model in Fernández-Villaverde and Rubio-Ramírez (2004)
        https://www.sas.upenn.edu/~jesusfv/ejemplo.pdf
        This env takes in no actions, so it is a process that does not interact with the agent"""

    def __init__(self, structural_params: dict = None, env_params: dict = None):
        structural_params.setdefault('a', 0.5)
        structural_params.setdefault('b', 0.3)
        structural_params.setdefault('w_mean', 0.)
        structural_params.setdefault('w_std', 1.)
        structural_params.setdefault('v_df', 2.)

        # RL related settings
        env_params.setdefault('max_steps', 100)
        # Action: fraction of capital k to invest (I/k); if action size = 100, 0.00-0.99 of all capital as investment
        self.action_space = gym.spaces.Box(np.array([0, ]), np.array([1, ]), dtype=np.float32)
        # Observation: 1D array (k, z)
        self.observation_space = gym.spaces.Box(np.array([0., 0.]), np.array([np.inf, np.inf]), dtype=np.float32)
        self.steps = 0
        k0, z0 = self.env_params['initial_state']
        self.state = np.array([k0, z0], dtype=np.float32)
        self.done = False


class WhitedBasicModel(StructuralModel, ABC):
    """Basic Model in Strebulaev and Whited (2012), *Dynamic Models and Structural Estimation*"""

    def __init__(self, structural_params: dict = None, env_params: dict = None):
        # Set default value of structural params
        structural_params.setdefault('gamma', 0.96)  # discount rate
        structural_params.setdefault('delta', 0.15)  # depreciation of capital
        structural_params.setdefault('theta', 0.7)  # pi = z * k**theta
        structural_params.setdefault('rho', 0.7)  # persistence of the shock process
        structural_params.setdefault('sigma', 0.15)  # standard deviation of the shock process

        # Set default value of env params
        env_params.setdefault('seed', None)  # random seed for shocks
        env_params.setdefault('initial_state', (1, 1))  # (k_0, z_0) time 0 capital and shock
        env_params.setdefault('psi_func', lambda k, z: 0)  # investment adjustment cost function psi(k, z)
        env_params.setdefault('pi_func', lambda k, z: z * (
                k ** structural_params['theta']))  # profit function pi(k, z) = z*k**theta
        env_params.setdefault('z_proc_func', log_ar1)  # z process: z(z0, rho, sigma)
        env_params.setdefault('num_structural_params', 5)

        super().__init__(structural_params=structural_params, env_params=env_params)

        # RL related settings
        env_params.setdefault('max_steps', 100)
        # Action: fraction of capital k to invest (I/k); if action size = 100, 0.00-0.99 of all capital as investment
        self.action_space = gym.spaces.Box(np.array([0, ]), np.array([1, ]), dtype=np.float32)
        # Observation: 1D array (k, z)
        self.observation_space = gym.spaces.Box(np.array([0., 0.]), np.array([np.inf, np.inf]), dtype=np.float32)
        self.steps = 0
        k0, z0 = self.env_params['initial_state']
        self.state = np.array([k0, z0], dtype=np.float32)
        self.done = False

    def reset(self):
        k0, z0 = self.env_params['initial_state']
        self.state = np.array([k0, z0], dtype=np.float32)
        self.steps = 0
        self.done = False
        return self.state

    def step(self, action: float):  # action is investment I_t over capital k_t
        k_curr = self.state[0]
        z_curr = self.state[1]
        i_curr = action * k_curr
        # update z
        z_new = self.env_params['z_proc_func'](z_curr, self.structural_params['rho'], self.structural_params['sigma'])
        # update k
        k_new = (1 - self.structural_params['delta']) * k_curr + i_curr
        # reward function
        # print(self.pi(k0, z0),  self.psi(I0, k0), I0)
        reward = self.env_params['pi_func'](k_curr, z_curr) - self.env_params['psi_func'](i_curr, k_curr) - i_curr
        # new state
        self.state = np.array([k_new, z_new], dtype=np.float32)
        # add step
        self.steps += 1
        # decide end or not
        if self.steps == self.env_params['max_steps']:
            self.done = True
        return self.state, reward, self.done, {}
