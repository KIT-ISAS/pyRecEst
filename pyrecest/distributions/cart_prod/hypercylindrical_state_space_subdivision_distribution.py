import numpy as np

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import array, atleast_1d, column_stack, full, reshape

from pyrecest.distributions.cart_prod.abstract_hypercylindrical_distribution import (
    AbstractHypercylindricalDistribution,
)
from pyrecest.distributions.cart_prod.state_space_subdivision_distribution import (
    StateSpaceSubdivisionDistribution,
)
from pyrecest.distributions.circle.circular_uniform_distribution import (
    CircularUniformDistribution,
)
from pyrecest.distributions.hypertorus.hypertoroidal_grid_distribution import (
    HypertoroidalGridDistribution,
)
from pyrecest.distributions.nonperiodic.custom_linear_distribution import (
    CustomLinearDistribution,
)


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

    @property
    def input_dim(self):
        return self.dim

    def plot(self, interpolate=False):
        return AbstractHypercylindricalDistribution.plot(self)

    def plot_interpolated(self):
        return self.plot(interpolate=True)

    def mode(self):
        return StateSpaceSubdivisionDistribution.mode(self)

    @staticmethod
    def from_distribution(distribution, no_of_grid_points, grid_type="CartesianProd"):
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

    @staticmethod
    def from_function(
        fun,
        no_of_grid_points,
        dim_lin,
        dim_bound=1,
        grid_type="CartesianProd",
        int_range=(-float("inf"), float("inf")),
    ):
        """
        Create a HypercylindricalStateSpaceSubdivisionDistribution from a
        function handle.

        Parameters
        ----------
        fun : callable
            PDF function taking an array of shape (n_samples, dim_bound + dim_lin)
            and returning an array of shape (n_samples,).
        no_of_grid_points : int
            Number of equidistant grid points on the circle.
        dim_lin : int
            Dimension of the linear part. Currently must be 1.
        dim_bound : int, optional
            Dimension of the bounded (periodic) part (default 1).
        grid_type : str, optional
            Grid type (currently unused, kept for API compatibility).
        int_range : tuple of float, optional
            Integration range for the linear part (default (-inf, inf)).

        Returns
        -------
        HypercylindricalStateSpaceSubdivisionDistribution
        """
        assert dim_lin == 1, "Currently, linear dimension must be 1."

        gd = HypertoroidalGridDistribution.from_distribution(
            CircularUniformDistribution(), no_of_grid_points
        )
        grid = gd.get_grid()  # shape (n,)
        grid_values = np.zeros(no_of_grid_points)
        cds = []

        for i in range(no_of_grid_points):
            grid_i = float(grid[i])

            def fun_curr(y, _grid_i=grid_i):
                """Evaluate fun with the periodic part fixed to _grid_i."""
                y_1d = np.asarray(y, dtype=float).ravel()
                m = y_1d.shape[0]
                x_input = np.column_stack(
                    [np.full(m, _grid_i), y_1d]
                )  # shape (m, 2)
                return np.asarray(fun(array(x_input)), dtype=float).ravel()

            # Build an unnormalized conditional distribution to compute the
            # marginal probability for this grid point.
            cd_unnorm = CustomLinearDistribution(
                lambda x, fc=fun_curr: array(fc(np.asarray(x))), dim_lin
            )

            # Integrate to get the marginal weight for this grid point.
            integral_val = float(
                cd_unnorm.integrate(
                    left=array([float(int_range[0])]),
                    right=array([float(int_range[1])]),
                )
            )
            grid_values[i] = integral_val

            # Normalized conditional distribution: p(lin | bound = grid_i).
            cds.append(
                CustomLinearDistribution(
                    lambda x, fc=fun_curr: array(fc(np.asarray(x))),
                    dim_lin,
                    scale_by=1.0 / integral_val,
                )
            )

        gd.grid_values = grid_values
        return HypercylindricalStateSpaceSubdivisionDistribution(gd, cds)
