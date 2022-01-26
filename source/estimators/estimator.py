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

    def __init__(self,
                 data: np.ndarray = None,  # (nsamples, N, T) or (N, T); N: obs dim, T: eps length
                 solver: Solver = None,
                 env: StructuralModel = None,
                 estimator_params: dict = None):
        super().__init__(solver=solver, env=env, estimator_params=estimator_params)
        self.data = data
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
            gmm_error = self._gmm_error(param_dict, self.data)
            if gmm_error < running_min_error:
                running_min_error = gmm_error
                running_best_param = param_dict
        return running_best_param

    @staticmethod
    def _data_moments(obs_vec: np.ndarray) -> np.ndarray:
        moments = []
        if obs_vec.ndim == 2:  # (N, T)
            for i in range(obs_vec.shape[0]):
                mean = obs_vec[i, :].mean()
                moments = np.append(moments, mean)
                variance = obs_vec[i, :].var()
                moments = np.append(moments, variance)
        else:
            assert obs_vec.ndim == 3  # (nsample, N, T)
            for i in range(obs_vec.shape[1]):
                mean = obs_vec[:, i, :].mean(axis=1).mean()
                moments = np.append(moments, mean)
                variance = obs_vec[:, i, :].var(axis=1).mean()
                moments = np.append(moments, variance)
        return moments

    def _gmm_error(self, param_dict: Dict[str, float], data_obs_vec: np.ndarray):
        """Perform GMM on a single param dict
        :parameter: param_dict a dict like {'delta': 0.1, 'gamma': 1}
        :returns an error term that is float of how much error this param_dict generates in simulated samples"""
        sample_size = self.estimator_params['sample_size']
        # use: param_dict, sample_size, self.weight_matrix, self.solver, self.env
        sim_obs_vec = None
        for n in range(sample_size):
            obs_sample = self.solver.sample(
                param_dict=param_dict)  # np array of size (N, T); in WhitedBasicModel N=2 (k, i)
            obs_sample = obs_sample.reshape(1, *obs_sample.shape)  # obs_sample.shape = (1, N, T)
            # some method to concat/aggregate samples
            sim_obs_vec = obs_sample if sim_obs_vec is None else np.append(sim_obs_vec, obs_sample, axis=0)
        moms_data = self._data_moments(data_obs_vec)
        moms_model = self._data_moments(sim_obs_vec)
        err = (moms_model - moms_data) / (moms_data + 1.e-9)
        crit_val = err.T @ self.weight_matrix @ err
        return crit_val


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
