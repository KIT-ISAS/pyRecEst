from abc import ABC

import pyrecest.backend

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import array, full, hstack


class AbstractTrackerWithLogging(ABC):
    def __init__(self, log_prior_estimates=False, log_posterior_estimates=False):
        self.log_prior_estimates=log_prior_estimates
        self.log_posterior_estimates=log_posterior_estimates
        
        if log_prior_estimates:
            self.prior_estimates_over_time = array([[]])
        
        if log_posterior_estimates:
            self.posterior_estimates_over_time = array([[]])

    def _store_estimates(self, curr_ests, estimates_over_time):
        assert (
            pyrecest.backend.__name__ != "pyrecest.jax"
        ), "Not supported on this backend"
        import numpy as _np

        # Ensure curr_ests is a 2D array
        if curr_ests.ndim == 1:
            curr_ests = curr_ests.reshape(-1, 1)

        m, t = estimates_over_time.shape
        n = curr_ests.shape[0]

        if n <= m:
            curr_ests = _np.pad(
                curr_ests,
                ((0, m - n), (0, 0)),
                mode="constant",
                constant_values=float("NaN"),
            )
            estimates_over_time = hstack((estimates_over_time, curr_ests))
        else:
            estimates_over_time_new = full((n, t + 1), float("NaN"))
            estimates_over_time_new[:m, :t] = estimates_over_time
            estimates_over_time_new[:, -1] = curr_ests.flatten()
            estimates_over_time = estimates_over_time_new

        return estimates_over_time
