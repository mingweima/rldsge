import pickle
import sys
from abc import ABC, abstractmethod

import numpy as np

sys.path.append("..")

from source.envs.environment import StructuralModel


class Solver(ABC):
    """A solver / an agent in RL setting"""

    def __init__(self, env: StructuralModel = None, solver_params: dict = None):
        self.env = env
        self.solver_params = solver_params

    @abstractmethod
    def act(self, obs: np.ndarray, evaluation: bool = True) -> np.ndarray:
        """Acts based on environment"""

    @abstractmethod
    def train(self, training_params: dict = None) -> None:
        """Train a solver/agent via interaction with the env"""

    def __str__(self):
        """Print method for solver for logging"""

    def save(self, name=None):
        name = str(self) if not name else name
        with open(f'{name}.pkl', 'wb') as handle:
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)


class LinearSolver(Solver, ABC):
    """Solver assuming linear policy"""

    def __init__(self, env: StructuralModel = None, solver_params: dict = None):
        super().__init__(env=env, solver_params=solver_params)
        self.policy_matrix = np.zeros((self.env.env_params['action_size'], self.env.env_params[
            'obs_size']))  # fill in relevant policy matrices for observations
    # TODO: JZH
    # def train(self, training_params: dict = None) -> None:
    #     ...
    #     self.policy_matrix = SOMETHING_CALCULATED


class ValueIterationSolver(Solver, ABC):
    """Grid search value function iteration solver"""
    # TODO: JZH
