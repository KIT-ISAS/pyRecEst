from collections.abc import Callable
import numpy as np
from functools import partial
from pyrecest.filters.abstract_circular_filter import AbstractCircularFilter
from pyrecest.distributions import WrappedNormalDistribution, CircularDiracDistribution

class WrappedNormalFilter(AbstractCircularFilter):
    def __init__(self, wn=None):
        """Initialize the filter."""
        if wn is None:
            wn = WrappedNormalDistribution(0, 1)
        AbstractCircularFilter.__init__(self, wn)

    def predict_identity(self, wn_sys):
        """Predicts using an identity system model."""
        self.filter_state = self.filter_state.convolve(wn_sys)

    def update_identity(self, wn_meas, z):
        mu_w_new = np.mod(z - wn_meas.mu, 2 * np.pi)
        wn_meas_shifted = WrappedNormalDistribution(mu_w_new, wn_meas.sigma)
        self.filter_state = self.filter_state.multiply_vm(wn_meas_shifted)

    def update_nonlinear_particle(self, likelihood, z):
        n = 100
        samples = self.filter_state.sample(n)
        wd = CircularDiracDistribution(samples)
        wd_new = wd.reweigh(partial(likelihood, z))
        self.filter_state = wd_new.to_wn()

    def update_nonlinear_progressive(self, likelihood: Callable, z: float, tau: float | None = None):
        # pylint: disable=too-many-locals
        DEFAULT_TAU = 0.02
        MINIMUM_LAMBDA = 0.001
        tau = tau if tau else DEFAULT_TAU
        lambda_ = 1
        steps = 0

        while lambda_ > 0:
            wd = self.filter_state.to_dirac5()
            likelihood_vals = np.array([likelihood(z, x) for x in wd.d])
            likelihood_vals_min, likelihood_vals_max = np.min(likelihood_vals), np.max(likelihood_vals)

            if likelihood_vals_max == 0:
                raise ValueError('Progressive update failed because likelihood is 0 everywhere')

            w_min, w_max = np.min(wd.w), np.max(wd.w)

            if likelihood_vals_min == 0 or w_min == 0:
                raise ZeroDivisionError('Cannot perform division by zero')

            current_lambda = min(np.log(tau * w_max / w_min) / np.log(likelihood_vals_min / likelihood_vals_max), lambda_)

            if current_lambda <= 0:
                raise ValueError('Progressive update with given threshold impossible')

            current_lambda = MINIMUM_LAMBDA
            wd_new = wd.reweigh(lambda x: likelihood(z, x) ** current_lambda)
            self.filter_state = wd_new.to_wn()
            lambda_ = lambda_ - current_lambda
            steps += 1
