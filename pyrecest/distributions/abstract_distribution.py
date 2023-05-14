from abc import ABC, abstractmethod

import numpy as np


class AbstractDistribution(ABC):
    """Abstract base class for all distributions."""

    def __init__(self, dim=None):
        self._dim = dim

    @property
    def dim(self):
        return self._dim

    @dim.setter
    def dim(self, value):
        if value is not None:
            if isinstance(value, int) and value > 0:
                self._dim = value
            else:
                raise ValueError("dim must be a positive integer or None.")
        else:
            self._dim = None

    @property
    @abstractmethod
    def input_dim(self):
        pass

    @abstractmethod
    def pdf(self, xs):
        pass

    @abstractmethod
    def mean(self):
        pass

    def sample(self, n):
        """Obtain n samples from the distribution."""
        return self.sample_metropolis_hastings(n)

    def sample_metropolis_hastings(
        self, n, proposal=None, start_point=None, burn_in=10, skipping=5
    ):
        """Metropolis Hastings sampling algorithm."""

        if proposal is None or start_point is None:
            raise NotImplementedError(
                "Default proposals and starting points should be set in inheriting classes."
            )

        total_samples = burn_in + n * skipping
        s = np.empty(
            (
                total_samples,
                self.input_dim,
            ),
        )
        s.fill(np.nan)
        x = start_point
        i = 0
        pdfx = self.pdf(x)

        while i < total_samples:
            x_new = proposal(x)
            pdfx_new = self.pdf(x_new)
            a = pdfx_new / pdfx
            if a.item() > 1 or a.item() > np.random.rand():
                s[i, :] = x_new.squeeze()
                x = x_new
                pdfx = pdfx_new
                i += 1

        relevant_samples = s[burn_in::skipping, :]
        return np.squeeze(relevant_samples)
