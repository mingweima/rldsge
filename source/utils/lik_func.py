from abc import ABC, abstractmethod


class LikFunc(ABC):
    @abstractmethod
    def lik(self, structural_params: dict) -> float:
        """Estimate (log)likelihood of a dict of given structural params"""

# TODO: ZHJ - write a particle filter LikFunc


