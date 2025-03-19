"""Abstract base class for all filters"""

import copy
from abc import ABC, abstractmethod


class AbstractFilter(ABC):
    """Abstract base class for all filters."""

    @abstractmethod
    def __init__(self, initial_filter_state):
        self._filter_state = initial_filter_state

    @property
    def filter_state(self):
        return self._filter_state

    @filter_state.setter
    def filter_state(self, new_state):
        assert self._filter_state is None or isinstance(
            new_state, type(self.filter_state)
        ), "New distribution has to be of the same class as (or inherit from) the previous density."
        self._filter_state = copy.deepcopy(new_state)

    def get_point_estimate(self):
        """Get a point estimate"""
        return self.filter_state.mean()

    @property
    def dim(self) -> int:
        """Convenience function to get the dimension of the filter.
        Overwrite if the filter is not directly based on a distribution."""
        return self.filter_state.dim

    def plot_filter_state(self):
        """Plot the filter state."""
        self.filter_state.plot()
