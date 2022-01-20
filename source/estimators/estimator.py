from typing import Dict

import numpy as np

from ..envs.env import StructuralModel
from ..utils.lik_func import *
from ..utils.useful_class import ParameterGrid


class Estimator(ABC):
    """An Estimator takes in a (trained) solver and relevant params
       and outputs estimated structural params
    """

    def __init__(self, solver: Solver = None, estimator_params: dict = None):
        self.solver = solver
        self.env = solver.env
        self.estimator_params = estimator_params
        self.num_structural_params = self.env.env_params['num_structural_params']
        self.estimated_params = None

    @abstractmethod
    def estimate(self) -> dict:
        """Outputs estimation using a dict (e.g. dict['k'] = 0.95)"""
        """How?"""
        return self.estimator_params


class SMMEstimator(Estimator, ABC):
    """Estimator using Simulated Method of Moments"""

    def __init__(self, solver: Solver = None, env: StructuralModel = None, estimator_params: dict = None):
        super().__init__(solver=solver, env=env, estimator_params=estimator_params)
        self.estimator_params.setdefault("verbose", True)
        self.estimator_params.setdefault("weight_matrix", "identity")  # weight matrix type for GMM
        self.estimator_params.setdefault("sample_size", 1000)
        assert "grid" in self.estimator_params
        assert "num_moments" in self.estimator_params
        self.estimator_params.setdefault("grid", ParameterGrid({'this_is_an_example': [0.1]}))
        self.estimator_params.setdefault("n_moment", 1)

        if self.estimator_params['weight_matrix'] not in ["identity"]:
            raise ValueError(f"No weight matrix {self.estimator_params['weight_matrix']}")
        if self.estimator_params['weight_matrix'] == 'identity':
            self.weight_matrix = np.eye(self.estimator_params['n_moment'])

    def estimate(self) -> Dict[str, float]:
        """Use SMM to estimate structural params
            Returns a dict of estimated structural params"""
        running_min_error = np.inf
        running_best_param = None
        for param_dict in self.estimator_params['grid']:
            gmm_error = self._gmm_error(param_dict)
            if gmm_error < running_min_error:
                running_min_error = gmm_error
                running_best_param = param_dict
        return running_best_param

    def _gmm_error(self, param_dict: Dict[str, float]):
        """Perform GMM on a single param dict
        :parameter: param_dict a dict like {'delta': 0.1, 'gamma': 1}
        :returns an error term that is float of how much error this param_dict generates in simulated samples"""
        sample_size = self.estimator_params['sample_size']

        # TODO (ZHJ): write stuff that maps param_dict to float;
        # use: param_dict, sample_size, self.weight_matrix, self.solver, self.env
        for n in range(sample_size):
            obs_sample = self.solver.sample()  # np array of size (N, T); in WhitedBasicModel N=2 (k, i)
            # some method to concat/aggregate samples
        # tot_sample? use this to calculate error

        raise NotImplementedError


class LikelihoodEstimator(Estimator, ABC):
    """General likelihood estimator using some kind of given likelihood function"""

    def __init__(self, solver: Solver = None, estimator_params: dict = None):
        super().__init__(solver=solver, estimator_params=estimator_params)
        assert "lik_func" in estimator_params  # class LikFunc object (likelihood function) from utils.lik_func
        self.lik_func = estimator_params['lik_func']
        assert isinstance(self.lik_func, LikFunc)
    # TODO: JZH


if __name__ == "__main__":
    grid = {
        'delta': [0.1, 0.2, 0.3],
        'gamma': [1, 10]
    }
    pg = ParameterGrid(grid)
    for g in pg:
        print(g)
