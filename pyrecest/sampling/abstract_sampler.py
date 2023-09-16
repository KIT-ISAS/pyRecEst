from abc import ABC, abstractmethod
import numpy as np

class AbstractSampler(ABC):       
    @abstractmethod
    def sample_stochastic(self, n_samples: int, dim: int) -> np.ndarray:
        raise NotImplementedError("Abstract method not implemented!")