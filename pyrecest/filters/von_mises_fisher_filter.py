import numpy as np
from pyrecest.distributions import VonMisesFisherDistribution

from .abstract_hyperspherical_filter import AbstractHypersphericalFilter


class VonMisesFisherFilter(AbstractHypersphericalFilter):
    def __init__(self):
        AbstractHypersphericalFilter.__init__(
            self, VonMisesFisherDistribution(np.array([1, 0]), 1)
        )

    @property
    def filter_state(self):
        return self._filter_state

    @filter_state.setter
    def filter_state(self, filter_state):
        assert isinstance(
            filter_state, VonMisesFisherDistribution
        ), "filter_state must be an instance of VMFDistribution."
        self._filter_state = filter_state

    def predict_identity(self, sys_noise):
        """
        State prediction via mulitiplication. Provide zonal density for update
        Could add support for a rotation Q
        """
        assert isinstance(
            sys_noise, VonMisesFisherDistribution
        ), "sys_noise must be an instance of VMFDistribution."
        self.filter_state = self.filter_state.convolve(sys_noise)

    def update_identity(self, meas_noise, z):
        """
        State update via mulitiplication. Provide zonal density for update
        Could add support for a rotation Q
        """
        assert isinstance(
            meas_noise, VonMisesFisherDistribution
        ), "meas_noise must be an instance of VMFDistribution."
        assert (
            meas_noise.mu[-1] == 1
        ), "Set mu of meas_noise to [0,0,...,1] to acknowledge that the mean is discarded."
        assert (
            z.shape[0] == self.filter_state.input_dim
        ), "Dimension mismatch between measurement and state."
        assert np.ndim(z) == 1, "z should be a vector."
        meas_noise.mu = z
        self.filter_state = self.filter_state.multiply(meas_noise)
