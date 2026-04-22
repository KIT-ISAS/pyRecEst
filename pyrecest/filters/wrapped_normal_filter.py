from collections.abc import Callable
from functools import partial

# pylint: disable=redefined-builtin,no-name-in-module,no-member
from pyrecest.backend import array, log, max, min, mod, pi
from pyrecest.distributions import CircularDiracDistribution, WrappedNormalDistribution

from .abstract_filter import AbstractFilter
from .manifold_mixins import CircularFilterMixin


class WrappedNormalFilter(AbstractFilter, CircularFilterMixin):
    def __init__(self, wn=None):
        """Initialize the filter."""
        if wn is None:
            wn = WrappedNormalDistribution(array(0.0), array(1.0))
        CircularFilterMixin.__init__(self)
        AbstractFilter.__init__(self, wn)

    def predict_identity(self, wn_sys):
        """Predicts using an identity system model."""
        self.filter_state = self.filter_state.convolve(wn_sys)

    def update_identity(self, wn_meas, z):
        mu_w_new = mod(z - wn_meas.mu, 2.0 * pi)
        wn_meas_shifted = WrappedNormalDistribution(mu_w_new, wn_meas.sigma)
        self.filter_state = self.filter_state.multiply_vm(wn_meas_shifted)

    def update_nonlinear_particle(self, likelihood, z):
        n = 100
        samples = self.filter_state.sample(n)
        wd = CircularDiracDistribution(samples)
        wd_new = wd.reweigh(partial(likelihood, z))
        self.filter_state = wd_new.to_wn()

    def update_nonlinear_progressive(
        self, likelihood: Callable, z: float, tau: float | None = None
    ):
        # pylint: disable=too-many-locals
        DEFAULT_TAU = 0.02
        MINIMUM_LAMBDA: float = 0.001
        tau = tau if tau else DEFAULT_TAU
        lambda_ = 1.0
        steps = 0

        while lambda_ > 0:
            wd = self.filter_state.to_dirac5()
            likelihood_vals = array([likelihood(z, x) for x in wd.d])
            likelihood_vals_min = min(likelihood_vals)
            likelihood_vals_max = max(likelihood_vals)

            if likelihood_vals_max == 0:
                raise ValueError(
                    "Progressive update failed because likelihood is 0 everywhere"
                )

            w_min = min(wd.w)
            w_max = max(wd.w)

            if likelihood_vals_min == 0 or w_min == 0:
                raise ZeroDivisionError("Cannot perform division by zero")

            current_lambda = min(
                log(tau * w_max / w_min)
                / log(likelihood_vals_min / likelihood_vals_max),
                lambda_,
            )

            if current_lambda <= 0:
                raise ValueError("Progressive update with given threshold impossible")

            current_lambda = MINIMUM_LAMBDA
            wd_new = wd.reweigh(lambda x: likelihood(z, x) ** current_lambda)
            self.filter_state = wd_new.to_wn()
            lambda_ = lambda_ - current_lambda
            steps += 1
