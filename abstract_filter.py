from abc import ABC, abstractmethod

class AbstractFilter(ABC):
    """Abstract base class for all filters."""

    @abstractmethod
    def set_state(self, state):
        """Set the state of the filter."""

    @abstractmethod
    def get_estimate(self):
        """Get the estimate of the filter."""

    @abstractmethod
    def get_point_estimate(self):
        """Get the point estimate of the filter."""

    def dim(self):
        """Convenience function to get the dimension of the filter.
        Overwrite if the filter is not directly based on a distribution."""
        return self.get_estimate().dim

    def plot_filter_state(self):
        """Plot the filter state."""
        self.get_estimate().plot()
