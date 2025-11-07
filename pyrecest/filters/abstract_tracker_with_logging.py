from abc import ABC
from math import nan

import pyrecest.backend

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import array, full, hstack, pad


class AbstractTrackerWithLogging(ABC):
    def __init__(self, log_prior_estimates=False, log_posterior_estimates=False):
        self.log_prior_estimates = log_prior_estimates
        self.log_posterior_estimates = log_posterior_estimates

        if log_prior_estimates:
            self.prior_estimates_over_time = array([[]])

        if log_posterior_estimates:
            self.posterior_estimates_over_time = array([[]])

    def _store_estimates(self, curr_ests, estimates_over_time):
        # Ensure curr_ests is a 2D array
        if curr_ests.ndim == 1:
            curr_ests = curr_ests.reshape(-1, 1)

        m, t = estimates_over_time.shape
        n = curr_ests.shape[0]

        if n <= m:
            # Use jnp.pad to pad the current estimates
            curr_ests = pad(
                curr_ests, ((0, m - n), (0, 0)), mode="constant", constant_values=nan
            )
            # Concatenate along the second dimension (time)
            estimates_over_time_new = hstack((estimates_over_time, curr_ests))
        else:
            estimates_over_time_new = full((n, t + 1), nan)
            if pyrecest.backend.__backend_name__ != "jax":
                estimates_over_time_new[:m, :t] = estimates_over_time
                estimates_over_time_new[:, -1] = curr_ests.flatten()
            else:
                estimates_over_time_new = estimates_over_time_new.at[:m, :t].set(
                    estimates_over_time
                )
                estimates_over_time_new = estimates_over_time_new.at[:, -1].set(
                    curr_ests.flatten()
                )

        return estimates_over_time_new
