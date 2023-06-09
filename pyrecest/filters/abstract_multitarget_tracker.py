from abc import ABC, abstractmethod

import numpy as np


class AbstractMultitargetTracker(ABC):
    def __init__(self, log_prior_estimates=True, log_posterior_estimates=True):
        self.log_prior_estimates = log_prior_estimates
        self.log_posterior_estimates = log_posterior_estimates
        if log_prior_estimates:
            self.prior_estimates_over_time = np.array([[]])
        if log_posterior_estimates:
            self.posterior_estimates_over_time = np.array([[]])

    def _store_estimates(self, curr_ests, estimates_over_time):
        # Ensure curr_ests is a 2D array
        if curr_ests.ndim == 1:
            curr_ests = curr_ests.reshape(-1, 1)

        m, t = estimates_over_time.shape
        n = np.size(curr_ests)

        if n <= m:
            curr_ests = np.pad(
                curr_ests, ((0, m - n), (0, 0)), mode="constant", constant_values=np.nan
            )
            estimates_over_time = np.hstack((estimates_over_time, curr_ests))
        else:
            estimates_over_time_new = np.full((n, t + 1), np.nan)
            estimates_over_time_new[:m, :t] = estimates_over_time
            estimates_over_time_new[:, -1] = curr_ests.flatten()
            estimates_over_time = estimates_over_time_new

        return estimates_over_time

    def store_prior_estimates(self):
        curr_ests = self.get_point_estimate(True)
        self.prior_estimates_over_time = self._store_estimates(
            curr_ests, self.prior_estimates_over_time
        )

    def store_posterior_estimates(self):
        curr_ests = self.get_point_estimate(True)
        self.posterior_estimates_over_time = self._store_estimates(
            curr_ests, self.posterior_estimates_over_time
        )

    @abstractmethod
    def get_point_estimate(self, flatten_vector=False):
        pass

    @abstractmethod
    def get_number_of_targets(self):
        pass
