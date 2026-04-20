import copy

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import argmax, argmin, array, abs as backend_abs
from pyrecest.backend import pi as backend_pi
from pyrecest.backend import sum as backend_sum

from pyrecest.distributions.cart_prod.abstract_hypercylindrical_distribution import (
    AbstractHypercylindricalDistribution,
)
from pyrecest.distributions.cart_prod.state_space_subdivision_distribution import (
    StateSpaceSubdivisionDistribution,
)
from pyrecest.distributions.hypertorus.hypertoroidal_grid_distribution import (
    HypertoroidalGridDistribution,
)
from pyrecest.distributions.nonperiodic.custom_linear_distribution import (
    CustomLinearDistribution,
)
from pyrecest.distributions.nonperiodic.linear_mixture import LinearMixture


class HypercylindricalStateSpaceSubdivisionDistribution(
    StateSpaceSubdivisionDistribution, AbstractHypercylindricalDistribution
):
    """
    Hypercylindrical state space subdivision distribution.

    Represents a distribution on the Cartesian product of a circle and the
    real line by storing a grid distribution on the circle and a conditional
    linear distribution for each grid point.

    Based on MATLAB implementation in libDirectional.
    """

    def __init__(self, gd, lin_distributions):
        StateSpaceSubdivisionDistribution.__init__(self, gd, lin_distributions)
        # Calling AbstractHypercylindricalDistribution.__init__ directly would
        # fail because it tries to set self.bound_dim / self.lin_dim as instance
        # attributes, which conflicts with the read-only properties defined in
        # StateSpaceSubdivisionDistribution. We therefore call only the part of
        # the init chain that sets self._dim.
        # pylint: disable=non-parent-init-called
        from pyrecest.distributions.abstract_manifold_specific_distribution import (
            AbstractManifoldSpecificDistribution,
        )

        AbstractManifoldSpecificDistribution.__init__(
            self, gd.dim + lin_distributions[0].dim
        )

    def pdf(self, xs):
        """
        Evaluate the joint pdf at the query points using nearest-neighbor
        interpolation for the conditional linear distributions.

        Parameters
        ----------
        xs : array-like, shape (n_samples, bound_dim + lin_dim)

        Returns
        -------
        p : array, shape (n_samples,)
        """
        import numpy as np

        xs_np = np.atleast_2d(np.asarray(xs, dtype=float))
        n_eval = xs_np.shape[0]
        x_bound = xs_np[:, : self.bound_dim]  # (n_eval, bound_dim)
        x_lin = xs_np[:, self.bound_dim :]  # (n_eval, lin_dim)

        # Find nearest grid indices (vectorised toroidal distance)
        grid_np = np.asarray(self.gd.get_grid(), dtype=float)  # (n_grid, bound_dim)
        delta = grid_np[None, :, :] - x_bound[:, None, :]  # (n_eval, n_grid, bound_dim)
        abs_delta = np.abs(delta)
        dists = np.sum(
            np.minimum(abs_delta**2, (2.0 * np.pi - abs_delta) ** 2), axis=-1
        )  # (n_eval, n_grid)
        indices = np.argmin(dists, axis=1)  # (n_eval,)

        # Evaluate periodic marginal pdf (nearest-neighbor)
        f_bound = np.asarray(self.gd.pdf(array(x_bound)), dtype=float).ravel()

        # Evaluate conditional linear pdfs
        f_lin = np.zeros(n_eval)
        for i_grid in np.unique(indices):
            mask = indices == i_grid
            f_lin[mask] = np.asarray(
                self.linear_distributions[i_grid].pdf(array(x_lin[mask])),
                dtype=float,
            ).ravel()

        return array(f_bound * f_lin)

    def marginalize_linear(self):
        """Return the marginal distribution over the bounded (periodic) part."""
        return copy.deepcopy(self.gd)

    def marginalize_periodic(self):
        """Return the marginal distribution over the linear part as a mixture."""
        weights = self.gd.grid_values / backend_sum(self.gd.grid_values)
        return LinearMixture(list(self.linear_distributions), weights)

    def mode(self):
        """
        Find the mode of the joint distribution by searching over grid points.

        Returns
        -------
        m : array, shape (bound_dim + lin_dim,)
        """
        import numpy as np

        n = self.gd.n_grid_points
        pdf_at_grid_points = np.empty(n)
        lin_modes = np.empty((n, self.lin_dim))

        for i in range(n):
            lin_mode = np.asarray(
                self.linear_distributions[i].mode(), dtype=float
            ).ravel()
            lin_modes[i] = lin_mode
            pdf_at_grid_points[i] = float(self.gd.grid_values[i]) * float(
                np.asarray(
                    self.linear_distributions[i].pdf(array(lin_mode.reshape(1, -1))),
                    dtype=float,
                ).ravel()[0]
            )

        best = int(np.argmax(pdf_at_grid_points))
        bound_point = np.asarray(self.gd.get_grid_point(best), dtype=float).ravel()
        return array(np.concatenate([bound_point, lin_modes[best]]))

    def sample(self, n: int):
        """
        Draw n samples from the distribution.

        Parameters
        ----------
        n : int

        Returns
        -------
        s : array, shape (n, bound_dim + lin_dim)
        """
        import numpy as np

        # Sample indices from the grid distribution weighted by grid_values
        weights = np.asarray(self.gd.grid_values, dtype=float).ravel()
        weights = weights / np.sum(weights)
        indices = np.random.choice(len(weights), size=n, p=weights)

        # Get the corresponding grid points: shape (n, bound_dim)
        grid_np = np.asarray(self.gd.get_grid(), dtype=float)  # (n_grid, bound_dim)
        samples_bounded = grid_np[indices]  # (n, bound_dim)

        samples_linear = np.empty((n, self.lin_dim))
        for i_grid in np.unique(indices):
            mask = indices == i_grid
            count = int(np.sum(mask))
            lin_samp = np.asarray(
                self.linear_distributions[i_grid].sample(count), dtype=float
            )
            if lin_samp.ndim == 1:
                lin_samp = lin_samp.reshape(-1, 1)
            samples_linear[mask] = lin_samp

        return array(np.column_stack([samples_bounded, samples_linear]))

    @staticmethod
    def from_distribution(distribution, no_of_grid_points, grid_type="cartesian_prod"):
        """
        Create a HypercylindricalStateSpaceSubdivisionDistribution from an
        AbstractHypercylindricalDistribution.

        Parameters
        ----------
        distribution : AbstractHypercylindricalDistribution
        no_of_grid_points : int
        grid_type : str, optional

        Returns
        -------
        HypercylindricalStateSpaceSubdivisionDistribution
        """
        return HypercylindricalStateSpaceSubdivisionDistribution.from_function(
            distribution.pdf,
            no_of_grid_points,
            distribution.lin_dim,
            distribution.bound_dim,
            grid_type,
        )

    # pylint: disable=too-many-positional-arguments
    @staticmethod
    def from_function(
        fun,
        no_of_grid_points,
        dim_lin,
        dim_bound=1,
        grid_type="cartesian_prod",
        int_range=(-float("inf"), float("inf")),
    ):
        """
        Create a HypercylindricalStateSpaceSubdivisionDistribution from a
        function handle.

        Parameters
        ----------
        fun : callable
            PDF function accepting an array of shape (n_samples, dim_bound + dim_lin)
            and returning shape (n_samples,).
        no_of_grid_points : int
            Number of equidistant grid points on the circle.
        dim_lin : int
            Dimension of the linear part. Currently must be 1.
        dim_bound : int, optional
            Dimension of the bounded (periodic) part (default 1).
        grid_type : str, optional
            Grid type passed to HypertoroidalGridDistribution (default
            "cartesian_prod").
        int_range : tuple of float, optional
            Integration range for the linear part (default (-inf, inf)).

        Returns
        -------
        HypercylindricalStateSpaceSubdivisionDistribution
        """
        import numpy as np

        assert dim_lin == 1, "Currently, linear dimension must be 1."
        assert dim_bound == 1, "Currently, bounded dimension must be 1."

        # Generate equidistant grid on the circle: shape (no_of_grid_points, 1)
        grid = HypertoroidalGridDistribution.generate_cartesian_product_grid(
            (no_of_grid_points,)
        )
        grid_np = np.asarray(grid, dtype=float)  # shape (n, 1)

        grid_values = np.zeros(no_of_grid_points)
        cds = []

        for i in range(no_of_grid_points):
            grid_i = float(grid_np[i, 0])

            def fun_curr(y, _grid_i=grid_i):
                """Evaluate fun with the periodic part fixed to _grid_i."""
                y_1d = np.asarray(y, dtype=float).ravel()
                m = y_1d.shape[0]
                x_input = np.column_stack([np.full(m, _grid_i), y_1d])
                return np.asarray(fun(array(x_input)), dtype=float).ravel()

            # Unnormalized conditional to compute the marginal weight
            cd_unnorm = CustomLinearDistribution(
                lambda x, fc=fun_curr: array(fc(np.asarray(x))), dim_lin
            )

            integral_val = float(
                cd_unnorm.integrate(
                    left=array([float(int_range[0])]),
                    right=array([float(int_range[1])]),
                )
            )
            grid_values[i] = integral_val

            # Normalized conditional: p(lin | bound = grid_i)
            cds.append(
                CustomLinearDistribution(
                    lambda x, fc=fun_curr: array(fc(np.asarray(x))),
                    dim_lin,
                    scale_by=1.0 / integral_val,
                )
            )

        gd = HypertoroidalGridDistribution(
            array(grid_values),
            grid_type=grid_type,
            grid=grid,
            enforce_pdf_nonnegative=False,
        )
        return HypercylindricalStateSpaceSubdivisionDistribution(gd, cds)
