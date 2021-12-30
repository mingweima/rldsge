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
