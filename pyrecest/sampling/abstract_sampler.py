from abc import ABC, abstractmethod


class AbstractSampler(ABC):
    @abstractmethod
    def sample_stochastic(self, n_samples: int, dim: int):
        raise NotImplementedError("Abstract method not implemented!")
