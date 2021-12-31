from abc import ABC, abstractmethod

from ..envs.environment import StructuralModel
from ..solvers.solver import Solver
from ..utils.lik_func import *


class Estimator(ABC):
    """An Estimator takes in a (trained) solver, an environment, and relevant params
       and outputs estimated structural params
    """

    def __init__(self, solver: Solver = None, env: StructuralModel = None, estimator_params: dict = None):
        self.solver = solver
        self.env = env
        self.estimator_params = estimator_params
        self.num_structural_params = env.env_params['num_structural_params']

    @abstractmethod
    def estimate(self) -> dict:
        """Outputs estimation using a dict (e.g. dict['k'] = 0.95)"""


class SMMEstimator(Estimator, ABC):
    """Estimator using Simulated Method of Moments"""

    def __init__(self, solver: Solver = None, env: StructuralModel = None, estimator_params: dict = None):
        super().__init__(solver=solver, env=env, estimator_params=estimator_params)

    def estimate(self) -> dict:
        """Use SMM to estimate structural params
            Returns a dict of estimated structural params"""
        self.estimator_params.setdefault("verbose", True)
        self.estimator_params.setdefault("weight_matrix", "identity")
        self.estimator_params.setdefault("sample_size", 1000)
        self.estimator_params.setdefault("grid_start_dict", {})
        self.estimator_params.setdefault("grid_end_dict", {})
        self.estimator_params.setdefault("gird_step_size_dict", {})
        # TODO: JZH


class LikelihoodEstimator(Estimator, ABC):
    """General likelihood estimator using some kind of given likelihood function"""

    def __init__(self, solver: Solver = None, env: StructuralModel = None, estimator_params: dict = None):
        super().__init__(solver=solver, env=env, estimator_params=estimator_params)
        assert "lik_func" in estimator_params  # class LikFunc object (likelihood function) from utils.lik_func
        self.lik_func = estimator_params['lik_func']
        assert isinstance(self.lik_func, LikFunc)
    # TODO: JZH


