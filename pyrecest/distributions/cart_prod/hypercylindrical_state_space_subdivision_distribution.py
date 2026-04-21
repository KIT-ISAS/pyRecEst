import copy

import numpy as np
from scipy.integrate import quad

from ..circle.circular_uniform_distribution import CircularUniformDistribution
from ..hypertorus.hypertoroidal_grid_distribution import HypertoroidalGridDistribution
from ..nonperiodic.custom_linear_distribution import CustomLinearDistribution
from ..nonperiodic.linear_mixture import LinearMixture
from .abstract_hypercylindrical_distribution import AbstractHypercylindricalDistribution
from .state_space_subdivision_distribution import StateSpaceSubdivisionDistribution


class HypercylindricalStateSpaceSubdivisionDistribution(
    StateSpaceSubdivisionDistribution, AbstractHypercylindricalDistribution
):
    @property
    def bound_dim(self):
        if hasattr(self, "gd"):
            return self.gd.dim
        return self._bound_dim

    @bound_dim.setter
    def bound_dim(self, value):
        self._bound_dim = value

    @property
    def lin_dim(self):
        if hasattr(self, "linear_distributions") and self.linear_distributions:
            return self.linear_distributions[0].dim
        return self._lin_dim

    @lin_dim.setter
    def lin_dim(self, value):
        self._lin_dim = value

    def __init__(self, gd, lin_distributions):
        if gd.dim != 1:
            raise ValueError(
                "Hypercylindrical distributions require exactly one bounded dimension."
            )
        if not lin_distributions:
            raise ValueError("lin_distributions must not be empty.")

        AbstractHypercylindricalDistribution.__init__(
            self, gd.dim, lin_distributions[0].dim
        )
        StateSpaceSubdivisionDistribution.__init__(self, gd, lin_distributions)

    def plot(self, interpolate=False):
        if interpolate:
            return AbstractHypercylindricalDistribution.plot(self)
        return AbstractHypercylindricalDistribution.plot(self)

    def plot_interpolated(self):
        return self.plot(interpolate=True)

    def marginalize_linear(self):
        return copy.deepcopy(self.gd)

    def marginalize_periodic(self):
        weights = np.asarray(self.gd.grid_values).reshape(-1)
        weights = weights / np.sum(weights)
        return LinearMixture(list(self.linear_distributions), weights)

    def _closest_grid_indices(self, bounded_xs):
        bounded_xs = np.asarray(bounded_xs)
        if bounded_xs.ndim == 1:
            bounded_xs = bounded_xs.reshape(1, -1)

        if bounded_xs.shape[1] != self.bound_dim:
            raise ValueError(
                "Expected bounded_xs with shape "
                f"(n_eval, {self.bound_dim}), got {bounded_xs.shape}."
            )

        if getattr(self.gd, "grid_type", None) == "cartesian_prod":
            n_grid_points = int(self.gd.n_grid_points)
            step = 2.0 * np.pi / n_grid_points
            wrapped = np.mod(bounded_xs[:, 0], 2.0 * np.pi)
            return np.floor((wrapped + step / 2.0) / step).astype(int) % n_grid_points

        grid = np.asarray(self.gd.get_grid()).reshape(-1, self.bound_dim)
        delta = np.abs(grid[None, :, :] - bounded_xs[:, None, :])
        toroidal_delta = np.minimum(delta, 2.0 * np.pi - delta)
        return np.argmin(np.sum(toroidal_delta**2, axis=-1), axis=1)

    def pdf(self, xs):
        xs = np.asarray(xs)
        if xs.ndim == 1:
            xs = xs.reshape(1, -1)

        expected_dim = self.bound_dim + self.lin_dim
        if xs.shape[1] != expected_dim:
            raise ValueError(
                f"Expected xs with shape (n_eval, {expected_dim}), got {xs.shape}."
            )

        bounded_xs = xs[:, : self.bound_dim]
        linear_xs = xs[:, self.bound_dim :]
        closest_indices = self._closest_grid_indices(bounded_xs)
        bounded_pdf = np.asarray(self.gd.grid_values).reshape(-1)[closest_indices]

        linear_pdf = np.empty(xs.shape[0], dtype=float)
        for row_index, dist_index in enumerate(closest_indices):
            linear_pdf[row_index] = float(
                np.asarray(
                    self.linear_distributions[int(dist_index)].pdf(linear_xs[row_index])
                ).reshape(-1)[0]
            )

        values = bounded_pdf * linear_pdf
        return values[0] if values.size == 1 else values

    @staticmethod
    def from_distribution(distribution, no_of_grid_points, grid_type="cartesian_prod"):
        lin_dim = getattr(distribution, "lin_dim", getattr(distribution, "linD", None))
        bound_dim = getattr(
            distribution, "bound_dim", getattr(distribution, "boundD", None)
        )
        if lin_dim is None or bound_dim is None:
            raise AttributeError(
                "distribution must provide lin_dim/bound_dim (or linD/boundD)."
            )

        return HypercylindricalStateSpaceSubdivisionDistribution.from_function(
            distribution.pdf,
            no_of_grid_points,
            lin_dim,
            dim_bound=bound_dim,
            grid_type=grid_type,
        )

    @staticmethod
    def from_function(
        fun,
        no_of_grid_points,
        dim_lin,
        *,
        dim_bound=1,
        grid_type="cartesian_prod",
        int_range=(-np.inf, np.inf),
    ):
        if dim_bound != 1:
            raise NotImplementedError(
                "Hypercylindrical distributions require exactly one bounded dimension."
            )
        if dim_lin != 1:
            raise NotImplementedError(
                "from_function currently supports only one linear dimension."
            )

        normalized_grid_type = str(grid_type).lower().replace(" ", "_")
        if normalized_grid_type == "cartesianprod":
            normalized_grid_type = "cartesian_prod"
        if normalized_grid_type != "cartesian_prod":
            raise ValueError(f"Unsupported grid_type: {grid_type!r}")

        if np.isscalar(no_of_grid_points):
            periodic_grid_shape = [int(no_of_grid_points)]
        else:
            periodic_grid_shape = [int(n) for n in no_of_grid_points]

        if len(periodic_grid_shape) != dim_bound:
            raise ValueError(
                "no_of_grid_points must specify one entry per bounded dimension."
            )

        gd = HypertoroidalGridDistribution.from_distribution(
            CircularUniformDistribution(),
            periodic_grid_shape,
            grid_type=normalized_grid_type,
        )
        grid = np.asarray(gd.get_grid()).reshape(-1, dim_bound)
        marginals = np.empty(gd.n_grid_points, dtype=float)
        cds = [None] * gd.n_grid_points

        def evaluate_slice(linear_xs, periodic_point, joint_fun):
            linear_xs = np.asarray(linear_xs)
            scalar_input = linear_xs.ndim == 0
            linear_xs_2d = np.atleast_1d(linear_xs).reshape(-1, dim_lin)
            bounded_part = np.repeat(
                np.asarray(periodic_point).reshape(1, -1),
                linear_xs_2d.shape[0],
                axis=0,
            )
            joint_points = np.hstack((bounded_part, linear_xs_2d))
            values = np.asarray(joint_fun(joint_points)).reshape(-1)
            return float(values[0]) if scalar_input else values

        for idx, periodic_point in enumerate(grid):
            def integrand(y, periodic_point=periodic_point, joint_fun=fun):
                return float(evaluate_slice(y, periodic_point, joint_fun))

            marginal_value, _ = quad(integrand, int_range[0], int_range[1])
            if marginal_value <= 0.0:
                raise ValueError(
                    "The marginal density at a grid point must be strictly positive."
                )

            marginals[idx] = marginal_value

            def conditional_pdf(
                x,
                periodic_point=periodic_point,
                joint_fun=fun,
                marginal_value=marginal_value,
            ):
                return evaluate_slice(x, periodic_point, joint_fun) / marginal_value

            cds[idx] = CustomLinearDistribution(conditional_pdf, dim_lin)

        gd.grid_values = marginals.reshape(gd.grid_values.shape)
        gd.normalize_in_place(warn_unnorm=False)

        return HypercylindricalStateSpaceSubdivisionDistribution(gd, cds)
