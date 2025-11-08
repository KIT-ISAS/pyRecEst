from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Union

import pyrecest.backend

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import empty, int32, int64, log, random, squeeze


class AbstractManifoldSpecificDistribution(ABC):
    """
    Abstract base class for distributions catering to specific manifolds.
    Should be inerhited by (abstract) classes limited to specific manifolds.
    """

    def __init__(self, dim: int):
        self._dim = dim

    @abstractmethod
    def get_manifold_size(self) -> float:
        pass

    def get_ln_manifold_size(self):
        return log(self.get_manifold_size())

    @property
    def dim(self) -> int:
        """Get dimension of the manifold."""
        return self._dim

    @dim.setter
    def dim(self, value: int):
        """Set dimension of the manifold. Must be a positive integer or None."""
        if value <= 0:
            raise ValueError("dim must be a positive integer or None.")

        self._dim = value

    @property
    @abstractmethod
    def input_dim(self) -> int:
        pass

    @abstractmethod
    def pdf(self, xs):
        pass

    def ln_pdf(self, xs):
        return log(self.pdf(xs))

    @abstractmethod
    def mean(self):
        """
        Convenient access to a reasonable "mean" for different manifolds.

        :return: The mean of the distribution.
        :rtype:
        """

    def set_mode(self, _):
        """
        Set the mode of the distribution
        """
        raise NotImplementedError("set_mode is not implemented for this distribution")

    # Need to use Union instead of | to support torch.dtype
    def sample(self, n: Union[int, int32, int64]):
        """Obtain n samples from the distribution."""
        return self.sample_metropolis_hastings(n)

    # jscpd:ignore-start
    # pylint: disable=too-many-positional-arguments
    def sample_metropolis_hastings(
        self,
        n: Union[int, int32, int64],
        burn_in: Union[int, int32, int64] = 10,
        skipping: Union[int, int32, int64] = 5,
        proposal: Callable | None = None,
        start_point=None,
    ):
        # jscpd:ignore-end
        """Metropolis Hastings sampling algorithm."""
        assert (
            pyrecest.backend.__backend_name__ != "jax"
        ), "Not supported on this backend"
        if proposal is None or start_point is None:
            raise NotImplementedError(
                "Default proposals and starting points should be set in inheriting classes."
            )

        total_samples = burn_in + n * skipping
        s = empty(
            (
                total_samples,
                self.input_dim,
            ),
        )
        x = start_point
        i = 0
        pdfx = self.pdf(x)

        while i < total_samples:
            x_new = proposal(x)
            assert (
                x_new.shape == x.shape
            ), "Proposal must return a vector of same shape as input"
            pdfx_new = self.pdf(x_new)
            a = pdfx_new / pdfx
            if a.item() > 1 or a.item() > random.rand(1):
                s[i, :] = x_new.squeeze()
                x = x_new
                pdfx = pdfx_new
                i += 1

        relevant_samples = s[burn_in::skipping, :]
        return squeeze(relevant_samples)
