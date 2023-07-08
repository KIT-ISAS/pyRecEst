from .abstract_toroidal_filter import AbstractToroidalFilter
from pyrecest.distributions.hypertorus.toroidal_wrapped_normal_distribution import ToroidalWrappedNormalDistribution
import numpy as np

class ToroidalWrappedNormalFilter(AbstractToroidalFilter):
    def __init__(self):
        AbstractToroidalFilter.__init__(self, ToroidalWrappedNormalDistribution(np.array([0, 0]), np.eye(2)))

    def predict_identity(self, twn_sys):
        assert isinstance(twn_sys, ToroidalWrappedNormalDistribution)
        self.filter_state = self.filter_state.convolve(twn_sys)
