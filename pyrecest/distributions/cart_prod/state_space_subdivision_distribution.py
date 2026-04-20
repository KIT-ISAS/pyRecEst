import copy

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import (
    array,
    int32,
    int64,
)

from ..abstract_grid_distribution import AbstractGridDistribution
from .abstract_lin_periodic_cart_prod_distribution import (
    AbstractLinPeriodicCartProdDistribution,
)


class StateSpaceSubdivisionDistribution(AbstractLinPeriodicCartProdDistribution):
    """
    Rao-Blackwellized distribution with arbitrary distributions on the linear
    part. The distribution is stored as a grid distribution on the bounded
    (periodic) part and a set of conditional linear distributions, one for
    each grid point.

    Based on MATLAB implementation in libDirectional.
    """

    def __init__(self, gd, lin_distributions):
        """
        Parameters
        ----------
        gd : AbstractGridDistribution
            Grid distribution for the bounded (periodic) part. The grid values
            represent the marginal probability weights at each grid point.
        lin_distributions : list of AbstractLinearDistribution
            List of conditional linear distributions, one per grid point.
            Must have the same length as the number of grid points in gd.
        """
        assert isinstance(gd, AbstractGridDistribution), (
            "gd must be an AbstractGridDistribution."
        )
        assert len(lin_distributions) == gd.n_grid_points, (
            "The number of linear distributions and grid points must be equal."
        )
        lin_dim = lin_distributions[0].dim
        bound_dim = gd.dim
        AbstractLinPeriodicCartProdDistribution.__init__(self, bound_dim, lin_dim)
        self.gd = gd
        self.lin_distributions = lin_distributions
        # Normalize the grid part
        self.gd.normalize_in_place(warn_unnorm=False)

    @property
    def input_dim(self):
        return self.dim

    def pdf(self, xa):
        """
        Evaluate the joint pdf at the query points using nearest-neighbor
        interpolation for the conditional linear distributions.

        Parameters
        ----------
        xa : array-like, shape (n_samples, bound_dim + lin_dim)
            Query points. The first bound_dim columns are the periodic part,
            the remaining lin_dim columns are the linear part.

        Returns
        -------
        p : ndarray, shape (n_samples,)
        """
        import numpy as np

        xa = np.atleast_2d(np.asarray(xa, dtype=float))
        assert xa.shape[-1] == self.dim, (
            f"Dimension of xa ({xa.shape[-1]}) does not match the density's "
            f"dimension ({self.dim})."
        )
        x_bound = xa[:, : self.bound_dim]  # (n_samples, bound_dim)
        x_lin = xa[:, self.bound_dim :]  # (n_samples, lin_dim)

        # Find nearest grid point for each sample
        _, indices = self.gd.get_closest_point(x_bound)

        # Evaluate conditional linear distributions
        f_lin_cond_bound = np.empty(xa.shape[0])
        unique_indices = np.unique(indices)
        for curr_grid_ind in unique_indices:
            mask = indices == curr_grid_ind
            f_lin_cond_bound[mask] = np.asarray(
                self.lin_distributions[curr_grid_ind].pdf(x_lin[mask]),
                dtype=float,
            )

        # Evaluate the bounded (periodic) part
        f_bound = np.asarray(self.gd.pdf(x_bound.ravel()), dtype=float)

        return array(f_bound * f_lin_cond_bound)

    def marginalize_linear(self):
        """Return the marginal distribution over the bounded (periodic) part."""
        return self.gd

    def marginalize_periodic(self):
        """
        Return the marginal distribution over the linear part as a mixture.
        """
        import numpy as np
        from ..nonperiodic.linear_mixture import LinearMixture

        weights = np.asarray(self.gd.grid_values, dtype=float)
        weights = weights / np.sum(weights)
        return LinearMixture(list(self.lin_distributions), weights)

    def normalize(self):
        """Return a normalized copy of this distribution."""
        dist = copy.deepcopy(self)
        dist.gd.normalize_in_place(warn_unnorm=False)
        return dist

    def mode(self):
        """
        Find the mode of the joint distribution.

        The mode is computed by evaluating the pdf at the mode of each
        conditional linear distribution, weighted by the corresponding grid
        value. The grid point with the highest weighted pdf value is selected.

        Returns
        -------
        m : ndarray, shape (bound_dim + lin_dim,)
        """
        import numpy as np

        n = self.gd.n_grid_points
        pdf_at_grid_points = np.empty(n)
        lin_modes = np.empty(n)

        for i in range(n):
            lin_mode = float(np.asarray(self.lin_distributions[i].mode()).ravel()[0])
            lin_modes[i] = lin_mode
            pdf_at_grid_points[i] = float(self.gd.grid_values[i]) * float(
                np.asarray(self.lin_distributions[i].pdf(array([lin_mode]))).ravel()[0]
            )

        best = int(np.argmax(pdf_at_grid_points))
        bound_point = self.gd.get_grid_point(best)
        lin_point = array([lin_modes[best]])
        return array(list(np.asarray(bound_point)) + list(np.asarray(lin_point)))

    def sample(self, n: int):
        """
        Draw n samples from the distribution.

        First, samples the periodic part from the grid distribution, then
        draws from the corresponding conditional linear distribution.

        Parameters
        ----------
        n : int
            Number of samples.

        Returns
        -------
        s : ndarray, shape (n, bound_dim + lin_dim)
        """
        import numpy as np

        samples_bounded = np.asarray(self.gd.sample(n), dtype=float).ravel()
        _, indices = self.gd.get_closest_point(samples_bounded)

        unique_indices = np.unique(indices)
        samples_linear = np.empty((n, self.lin_dim))
        for curr_grid_ind in unique_indices:
            mask = indices == curr_grid_ind
            count = int(np.sum(mask))
            lin_samp = np.asarray(
                self.lin_distributions[curr_grid_ind].sample(count), dtype=float
            )
            if lin_samp.ndim == 1:
                lin_samp = lin_samp.reshape(-1, 1)
            samples_linear[mask] = lin_samp

        s = np.column_stack([samples_bounded.reshape(-1, 1), samples_linear])
        return array(s)

    def mean(self):
        """
        Return the hybrid mean: (mean direction of periodic part,
        mean of linear part).
        """
        return self.hybrid_mean()
