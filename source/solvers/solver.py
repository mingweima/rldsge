from abc import ABC, abstractmethod

import numpy as np

from ..envs.environment import StructuralModel


class Solver(ABC):
    """A solver / an agent in RL setting"""

    def __init__(self, env: StructuralModel = None, solver_params: dict = None):
        self.env = env
        self.solver_params = solver_params

    @abstractmethod
    def act(self, obs: np.ndarray, evaluation: bool = False) -> np.ndarray:
        """Acts based on environment"""

    @abstractmethod
    def train(self, training_params: dict = None) -> None:
        """Train a solver/agent via interaction with the env"""


class LinearSolver(Solver, ABC):
    """Solver assuming linear policy"""

    def __init__(self, env: StructuralModel = None, solver_params: dict = None):
        super().__init__(env=env, solver_params=solver_params)
        self.policy_matrix_dict = {}  # fill in relevant policy matrices for observations
    # TODO: JZH


class ValueIterationSolver(Solver, ABC):
    """Grid search value function iteration solver"""
    # TODO: JZH


class A2CSolver(Solver, ABC):
    """Actor-Critic Solver"""
