# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import array, eye
from pyrecest.distributions.hypertorus.toroidal_wrapped_normal_distribution import (
    ToroidalWrappedNormalDistribution,
)

from .abstract_filter import AbstractFilter
from .manifold_mixins import ToroidalFilterMixin


class ToroidalWrappedNormalFilter(AbstractFilter, ToroidalFilterMixin):
    def __init__(self):
        ToroidalFilterMixin.__init__(self)
        AbstractFilter.__init__(
            self, ToroidalWrappedNormalDistribution(array([0, 0]), eye(2))
        )

    def predict_identity(self, twn_sys):
        assert isinstance(twn_sys, ToroidalWrappedNormalDistribution)
        self.filter_state = self.filter_state.convolve(twn_sys)
