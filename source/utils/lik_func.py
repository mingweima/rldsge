from abc import ABC, abstractmethod

from source.solvers.solver import Solver


class LikFunc(ABC):
    @abstractmethod
    def lik(self, structural_params: dict, solver: Solver) -> float:
        """Estimate (log)likelihood of a dict of given structural params"""

# TODO: ZHJ - write a particle filter LikFunc
