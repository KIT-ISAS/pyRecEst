from .abstract_manifold_specific_filter import AbstractManifoldSpecificFilter


class AbstractHypertoroidalFilter(AbstractManifoldSpecificFilter):
    """Abstract base class for filters on the hypertorus."""

    def get_point_estimate(self):
        """Get the point estimate."""
        return self.filter_state.mean_direction()
