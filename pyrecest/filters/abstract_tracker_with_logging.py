from abc import ABC
import numpy as np

class AbstractTrackerWithLogging(ABC):
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
            if value:
                # Remove the 'log_' prefix from the key
                clean_key = key[4:] if key.startswith("log_") else key
                setattr(self, f"{clean_key}_over_time", np.array([[]]))

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