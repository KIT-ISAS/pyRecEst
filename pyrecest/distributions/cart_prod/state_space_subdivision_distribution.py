import copy
from abc import abstractmethod


class StateSpaceSubdivisionDistribution:
    """
    Represents a joint distribution over a Cartesian product of a grid-based
    (periodic/bounded) space and a linear space, where the linear part is
    represented as a collection of distributions conditioned on each grid point
    of the periodic/bounded part.

    The periodic part is stored as an AbstractGridDistribution, which holds
    grid_values (unnormalized weights) at each grid point.  The linear part is
    stored as a list of distributions, one per grid point.
    """

    def __init__(self, gd, linear_distributions):
        """
        Parameters
        ----------
        gd : AbstractGridDistribution
            Grid-based distribution for the periodic/bounded part.  Its
            grid_values represent (unnormalized) marginal weights over the
            grid points.
        linear_distributions : list
            One distribution per grid point representing the conditional
            distribution of the linear state given that grid point.
        """
        assert gd.n_grid_points == len(
            linear_distributions
        ), "Number of grid points in gd must match length of linear_distributions."
        self.gd = copy.deepcopy(gd)
        self.linear_distributions = list(copy.deepcopy(linear_distributions))

    @property
    def bound_dim(self):
        """Dimension of the periodic/bounded space (ambient dimension of grid points)."""
        return self.gd.dim

    @property
    def lin_dim(self):
        """Dimension of the linear space."""
        return self.linear_distributions[0].dim

    def hybrid_mean(self):
        """
        Returns the hybrid mean, i.e. the concatenation of the mean direction
        of the periodic part and the mean of the linear marginal.
        """
        # pylint: disable=no-name-in-module,no-member
        from pyrecest.backend import concatenate

        periodic_mean = self.gd.mean_direction()
        linear_mean_val = self.marginalize_periodic().mean()
        return concatenate([periodic_mean.reshape(-1), linear_mean_val.reshape(-1)])

    @abstractmethod
    def marginalize_linear(self):
        """Marginalise out the linear dimensions, returning a distribution over
        the periodic/bounded part only."""

    @abstractmethod
    def marginalize_periodic(self):
        """Marginalise out the periodic/bounded dimensions, returning a
        distribution over the linear part only."""
