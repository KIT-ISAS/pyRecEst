from abstract_filter import AbstractFilter

class AbstractHypertoroidalFilter(AbstractFilter):
    """Abstract base class for filters on the hypertorus."""

    def get_point_estimate(self):
        """Get the point estimate."""
        return self.get_estimate().mean_direction()