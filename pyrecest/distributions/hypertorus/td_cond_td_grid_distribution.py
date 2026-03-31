"""Conditional grid distribution on the hypertorus (TdCondTdGridDistribution)."""

from __future__ import annotations

import warnings

import numpy as np
from beartype import beartype

# pylint: disable=redefined-builtin,no-name-in-module,no-member
from pyrecest.backend import abs, all, any, array, mean, pi, reshape

from ..conditional.abstract_conditional_distribution import (
    AbstractConditionalDistribution,
)
from .hypertoroidal_grid_distribution import HypertoroidalGridDistribution


class TdCondTdGridDistribution(AbstractConditionalDistribution):
    """Conditional distribution on the hypertorus described by grid values.

    Represents ``f(a | b)`` where both ``a`` and ``b`` live on the same
    ``d``-dimensional hypertorus.  The combined state space is the Cartesian
    product of two hypertori, and :attr:`grid_values` is a square 2-D matrix
    where::

        grid_values[i, j]  =  f(grid[i] | grid[j])

    i.e. the *rows* index the conditioned variable ``a`` and the *columns*
    index the conditioning variable ``b``.

    Parameters
    ----------
    grid:
        Grid points on the individual hypertorus, shape
        ``(n_grid_points, dim_half)``.
    grid_values:
        Conditional probability values, shape
        ``(n_grid_points, n_grid_points)``.  Must be square; entry ``[i, j]``
        is ``f(grid[i] | grid[j])``.
    enforce_pdf_nonnegative:
        Whether to enforce non-negativity of the pdf values (default ``True``).
    """

    @beartype
    def __init__(
        self,
        grid,
        grid_values,
        enforce_pdf_nonnegative: bool = True,
        n_grid_points_per_dim: tuple | None = None,
    ):
        super().__init__()

        if grid_values.ndim != 2:
            raise ValueError("grid_values must be a 2-D array")
        if grid_values.shape[0] != grid_values.shape[1]:
            raise ValueError("grid_values must be square")
        if grid.shape[0] != grid_values.shape[0]:
            raise ValueError(
                "grid.shape[0] must equal grid_values.shape[0] (number of grid points)"
            )

        self.grid = grid  # (n_grid_points, dim_half)
        self.grid_values = grid_values  # (n_grid_points, n_grid_points)
        self.enforce_pdf_nonnegative = enforce_pdf_nonnegative
        # dim is the combined dimension of the two-hypertorus Cartesian product
        self.dim = 2 * grid.shape[1]
        # Used to reshape slices back to the correct multi-dimensional layout
        # when creating HypertoroidalGridDistribution instances.
        self._n_grid_points_per_dim: tuple[int, ...] | None = n_grid_points_per_dim

        self._check_normalization()

    # ----------------------------------------------------------------- internal

    def _check_normalization(self, tol: float = 0.01) -> None:
        """Warn if the distribution is not column-normalized.

        For each fixed ``b = grid[j]`` the values ``grid_values[:, j]``
        should integrate to 1 over the first variable, i.e.
        ``mean(grid_values[:, j]) * (2π)^(dim/2) ≈ 1``.
        """
        dim_half = self.dim // 2
        manifold_size = (2.0 * float(pi)) ** dim_half

        ints = mean(self.grid_values, axis=0) * manifold_size
        if any(abs(ints - 1.0) > tol):
            # Check whether swapping rows/cols would give a normalized distribution
            ints_swapped = mean(self.grid_values, axis=1) * manifold_size
            if all(abs(ints_swapped - 1.0) <= tol):
                raise ValueError(
                    "Normalization:maybeWrongOrder: The distribution is not "
                    "normalized, but would be normalized if the order of the "
                    "tori were swapped.  Check your input."
                )
            warnings.warn(
                "Normalization:notNormalized: When conditioning values for the "
                "first torus on the second, normalization is not guaranteed.  "
                "Check your input or increase the tolerance.",
                RuntimeWarning,
            )

    # ------------------------------------------------------------------ public

    def normalize(self):
        """Return *self* unchanged (normalization check only; no rescaling).

        For compatibility with the rest of the API.  Normalization must be
        ensured by the caller when constructing :class:`TdCondTdGridDistribution`.
        """
        self._check_normalization()
        return self

    @beartype
    def marginalize_out(self, first_or_second: int):
        """Marginalize out one of the two variables.

        Parameters
        ----------
        first_or_second:
            ``1`` to sum over the first (conditioned) variable ``a``;
            ``2`` to sum over the second (conditioning) variable ``b``.

        Returns
        -------
        HypertoroidalGridDistribution
            Marginal distribution on the remaining variable.
        """
        if first_or_second == 1:
            # Sum rows → proportional to the marginal over b
            grid_values_sgd = np.sum(np.asarray(self.grid_values), axis=0)
        elif first_or_second == 2:
            # Sum cols → proportional to the marginal over a
            grid_values_sgd = np.sum(np.asarray(self.grid_values), axis=1)
        else:
            raise ValueError("first_or_second must be 1 or 2")

        return self._make_hypertoroidal_grid_dist(array(grid_values_sgd))

    @beartype
    def fix_dim(self, first_or_second: int, point):
        """Fix one variable to a grid point and return the slice distribution.

        Parameters
        ----------
        first_or_second:
            ``1`` to fix the first variable ``a`` to *point*;
            ``2`` to fix the second variable ``b`` to *point*.
        point:
            The value to fix; must be an exact grid point,
            shape ``(dim_half,)`` or ``(dim_half, 1)``.

        Returns
        -------
        HypertoroidalGridDistribution
            Conditional (or likelihood) distribution on the remaining variable.

        Raises
        ------
        ValueError
            If *point* is not found in the grid.
        """
        dim_half = self.dim // 2
        point_arr = np.asarray(point).ravel()
        if point_arr.shape != (dim_half,):
            raise ValueError(
                f"point must have shape ({dim_half},), got {point_arr.shape}"
            )

        grid_np = np.asarray(self.grid)
        diffs = np.all(np.isclose(grid_np, point_arr[None, :]), axis=1)
        indices = np.where(diffs)[0]
        if indices.size == 0:
            raise ValueError(
                "Cannot fix value at this point because it is not on the grid"
            )
        locb = int(indices[0])

        gv = np.asarray(self.grid_values)
        if first_or_second == 1:
            grid_values_slice = gv[locb, :]
        elif first_or_second == 2:
            grid_values_slice = gv[:, locb]
        else:
            raise ValueError("first_or_second must be 1 or 2")

        return self._make_hypertoroidal_grid_dist(array(grid_values_slice))

    # -------------------------------------------------------------- factory

    @staticmethod
    @beartype
    def from_function(
        fun,
        n_grid_points: int | tuple | list,
        fun_does_cartesian_product: bool,
        grid_type: str = "CartesianProd",
        dim: int = 2,
    ) -> "TdCondTdGridDistribution":
        """Construct a :class:`TdCondTdGridDistribution` from a function handle.

        Parameters
        ----------
        fun:
            ``f(a, b)`` where ``a`` and ``b`` each have shape
            ``(n_eval, dim_half)``.  When *fun_does_cartesian_product* is
            ``False``, ``n_eval = n_total ** 2`` (all pairs); the function
            must return shape ``(n_total ** 2,)``.
            When *fun_does_cartesian_product* is ``True``, ``a`` and ``b``
            are both the full grid (shape ``(n_total, dim_half)``) and the
            function must return shape ``(n_total, n_total)``.
        n_grid_points:
            Number of grid points per dimension of a *single* hypertorus.
            Pass an ``int`` to use the same count for every dimension, or a
            sequence to specify each dimension individually.
        fun_does_cartesian_product:
            Whether *fun* handles all grid combinations internally.
        grid_type:
            Grid layout; currently only ``'CartesianProd'`` /
            ``'CartesianProduct'`` is supported.
        dim:
            Total dimension of the combined space (must be even; equals
            ``2 * dim_half``).

        Returns
        -------
        TdCondTdGridDistribution
        """
        if dim % 2 != 0:
            raise ValueError(
                "dim must be even (it represents two copies of a hypertorus)."
            )
        # User-facing API mirrors the MATLAB convention ('CartesianProd').
        # Internally HypertoroidalGridDistribution uses 'cartesian_prod'.
        if grid_type not in ("CartesianProd", "CartesianProduct"):
            raise ValueError(
                "Grid scheme not recognized; only 'CartesianProd' / "
                "'CartesianProduct' is currently supported."
            )

        dim_half = dim // 2

        if isinstance(n_grid_points, int):
            n_grid_points_seq = [n_grid_points] * dim_half
        else:
            n_grid_points_seq = list(n_grid_points)

        # Generate the grid on a single hypertorus: (n_total, dim_half)
        grid = HypertoroidalGridDistribution.generate_cartesian_product_grid(
            n_grid_points_seq
        )
        n_total = grid.shape[0]

        if fun_does_cartesian_product:
            fvals = reshape(array(fun(grid, grid)), (n_total, n_total))
        else:
            # Build index arrays so that fvals[i, j] = f(grid[i], grid[j]).
            # Using plain numpy for index generation (pure control flow).
            idx_i, idx_j = (
                m.ravel()
                for m in np.meshgrid(
                    np.arange(n_total), np.arange(n_total), indexing="ij"
                )
            )
            fvals = reshape(array(fun(grid[idx_i], grid[idx_j])), (n_total, n_total))

        return TdCondTdGridDistribution(
            grid, fvals, n_grid_points_per_dim=tuple(n_grid_points_seq)
        )

    # --------------------------------------------------------------- helpers

    def _make_hypertoroidal_grid_dist(
        self, grid_values_flat
    ) -> HypertoroidalGridDistribution:
        """Create a :class:`HypertoroidalGridDistribution` from a 1-D slice.

        When the instance was created via :meth:`from_function` the
        per-dimension point counts are known and the flat slice is reshaped to
        the correct multi-dimensional layout required by the ``cartesian_prod``
        grid type.  Otherwise the ``custom`` grid type is used.
        """
        if self._n_grid_points_per_dim is not None:
            grid_values_shaped = reshape(
                grid_values_flat, self._n_grid_points_per_dim
            )
            return HypertoroidalGridDistribution(
                grid_values=grid_values_shaped,
                grid_type="cartesian_prod",
                grid=self.grid,
                enforce_pdf_nonnegative=self.enforce_pdf_nonnegative,
            )
        return HypertoroidalGridDistribution(
            grid_values=grid_values_flat,
            grid_type="custom",
            grid=self.grid,
            enforce_pdf_nonnegative=self.enforce_pdf_nonnegative,
        )
