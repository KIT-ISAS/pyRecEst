import copy
import warnings

# pylint: disable=redefined-builtin,no-name-in-module,no-member
from pyrecest.backend import (
    abs,
    all,
    any,
    arange,
    argmin,
    array_equal,
    linalg,
    mean,
    meshgrid,
    pi,
    sum,
)

from .abstract_conditional_distribution import AbstractConditionalDistribution


class TdCondTdGridDistribution(AbstractConditionalDistribution):
    """
    Conditional distribution on Td x Td represented by a grid of values.

    For a conditional distribution f(a|b), ``grid_values[i, j]`` stores
    the value f(grid[i] | grid[j]).

    Convention
    ----------
    - ``grid`` has shape ``(n_points, d)`` where ``d`` is the number of
      dimensions of the individual torus (e.g. d=1 for T1).
    - ``grid_values`` has shape ``(n_points, n_points)``.
    - ``dim = 2 * d`` is the dimension of the Cartesian product space
      (convention inherited from libDirectional).
    """

    def __init__(self, grid, grid_values, enforce_pdf_nonnegative=True):
        """
        Parameters
        ----------
        grid : array of shape (n_points, d)
            Grid points on the torus.
        grid_values : array of shape (n_points, n_points)
            Conditional pdf values: ``grid_values[i, j] = f(grid[i] | grid[j])``.
            Must be non-negative when ``enforce_pdf_nonnegative=True``.
        enforce_pdf_nonnegative : bool
            Whether non-negativity of ``grid_values`` is required.
        """
        if grid.ndim != 2:
            raise ValueError("grid must be a 2D array of shape (n_points, d).")

        n_points, d = grid.shape

        if grid_values.ndim != 2 or grid_values.shape != (n_points, n_points):
            raise ValueError(
                f"grid_values must be a square 2D array of shape ({n_points}, {n_points})."
            )

        if enforce_pdf_nonnegative and any(grid_values < 0):
            raise ValueError("grid_values must be non-negative.")

        self.grid = grid
        self.grid_values = grid_values
        self.enforce_pdf_nonnegative = enforce_pdf_nonnegative
        # Embedding dimension of the Cartesian product space (convention from
        # libDirectional: dim = 2 * dim_of_individual_torus).
        self.dim = 2 * d

        self._check_normalization()

    # ------------------------------------------------------------------
    # Normalization
    # ------------------------------------------------------------------

    def _check_normalization(self, tol=0.01):
        """Warn if any column is not normalized to 1 over the torus."""
        d = self.dim // 2
        manifold_size = float((2.0 * pi) ** d)
        # For each fixed second argument j, the mean over i times the torus
        # volume should equal 1.
        ints = mean(self.grid_values, axis=0) * manifold_size
        if any(abs(ints - 1) > tol):
            # Check whether swapping the two arguments would yield normalisation.
            ints_swapped = mean(self.grid_values, axis=1) * manifold_size
            if all(abs(ints_swapped - 1) <= tol):
                raise ValueError(
                    "Normalization:maybeWrongOrder: Not normalized but would be if "
                    "the order of the two tori were swapped. Check input."
                )
            warnings.warn(
                "Normalization:notNormalized: When conditioning values for the first "
                "torus on the second, normalisation is not ensured. "
                "Check input or increase tolerance. "
                "No normalisation is performed; you may want to do this manually.",
                UserWarning,
            )

    def normalize(self):
        """No-op – returns ``self`` for compatibility."""
        return self

    # ------------------------------------------------------------------
    # Arithmetic
    # ------------------------------------------------------------------

    def multiply(self, other):
        """
        Element-wise multiply two conditional grid distributions.

        The resulting distribution is *not* normalized.

        Parameters
        ----------
        other : TdCondTdGridDistribution
            Must be defined on the same grid.

        Returns
        -------
        TdCondTdGridDistribution
        """
        if not array_equal(self.grid, other.grid):
            raise ValueError(
                "Multiply:IncompatibleGrid: Can only multiply distributions "
                "defined on identical grids."
            )
        warnings.warn(
            "Multiply:UnnormalizedResult: Multiplication does not yield a "
            "normalized result.",
            UserWarning,
        )
        result = copy.deepcopy(self)
        result.grid_values = result.grid_values * other.grid_values
        return result

    # ------------------------------------------------------------------
    # Marginalisation and conditioning
    # ------------------------------------------------------------------

    def marginalize_out(self, first_or_second):
        """
        Marginalize out one of the two tori.

        Parameters
        ----------
        first_or_second : int  (1 or 2)
            ``1`` marginalizes out the *first* dimension (sums over rows);
            ``2`` marginalizes out the *second* dimension (sums over columns).

        Returns
        -------
        HypertoroidalGridDistribution
        """
        # Import here to avoid circular imports
        from pyrecest.distributions.hypertorus.hypertoroidal_grid_distribution import (  # pylint: disable=import-outside-toplevel
            HypertoroidalGridDistribution,
        )

        if first_or_second == 1:
            grid_values_sgd = sum(self.grid_values, axis=0)
        elif first_or_second == 2:
            grid_values_sgd = sum(self.grid_values, axis=1)
        else:
            raise ValueError("first_or_second must be 1 or 2.")

        return HypertoroidalGridDistribution(grid_values_sgd, "custom", self.grid)

    def fix_dim(self, first_or_second, point):
        """
        Return the conditional slice for a fixed grid point.

        The supplied ``point`` must be an existing grid point.

        Parameters
        ----------
        first_or_second : int  (1 or 2)
            ``1`` fixes the *first* argument at ``point`` and returns values
            over the second torus;
            ``2`` fixes the *second* argument and returns values over the
            first torus.
        point : array of shape (d,)
            Grid point at which to evaluate.

        Returns
        -------
        HypertoroidalGridDistribution
        """
        # Import here to avoid circular imports
        from pyrecest.distributions.hypertorus.hypertoroidal_grid_distribution import (  # pylint: disable=import-outside-toplevel
            HypertoroidalGridDistribution,
        )

        d = self.grid.shape[1]
        if point.shape[0] != d:
            raise ValueError(
                f"point must have length {d} (dimension of the torus)."
            )

        diffs = linalg.norm(self.grid - point[None, :], axis=1)
        locb = argmin(diffs)
        if diffs[locb] > 1e-10:
            raise ValueError(
                "Cannot fix value at this point because it is not on the grid."
            )

        if first_or_second == 1:
            grid_values_slice = self.grid_values[locb, :]
        elif first_or_second == 2:
            grid_values_slice = self.grid_values[:, locb]
        else:
            raise ValueError("first_or_second must be 1 or 2.")

        return HypertoroidalGridDistribution(grid_values_slice, "custom", self.grid)

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @staticmethod
    def from_function(
        fun,
        no_of_grid_points,
        fun_does_cartesian_product=False,
        grid_type="CartesianProd",
        dim=2,
    ):
        """
        Construct a :class:`TdCondTdGridDistribution` from a callable.

        Parameters
        ----------
        fun : callable
            Conditional pdf function ``f(a, b)``.

            *   When ``fun_does_cartesian_product=False`` (default): ``fun``
                is called once with two arrays of shape ``(n_pairs, d)``
                containing all ``n_points²`` pairs, and must return a 1-D
                array of length ``n_points²``.
            *   When ``fun_does_cartesian_product=True``: ``fun`` is called
                with both full grids of shape ``(n_points, d)`` and must
                return an array of shape ``(n_points, n_points)``.
        no_of_grid_points : int
            Number of grid points for each torus.
        fun_does_cartesian_product : bool
            See ``fun`` description above.
        grid_type : str
            Grid type passed to
            :meth:`~pyrecest.distributions.hypertorus.hypertoroidal_grid_distribution.HypertoroidalGridDistribution.generate_cartesian_product_grid`.
            Defaults to ``'CartesianProd'``.
        dim : int
            Dimension of the Cartesian product space
            (``2 * dim_of_individual_torus``).
            Defaults to 2 (T1 × T1).

        Returns
        -------
        TdCondTdGridDistribution
        """
        # Import inside the function to avoid circular imports at module level.
        from pyrecest.distributions.hypertorus.hypertoroidal_grid_distribution import (  # pylint: disable=import-outside-toplevel
            HypertoroidalGridDistribution,
        )

        if dim % 2 != 0:
            raise ValueError(
                "dim must be even (it represents two copies of a hypertorus)."
            )
        if grid_type not in ("CartesianProd", "CartesianProduct"):
            raise ValueError(
                "Grid scheme not recognized; only 'CartesianProd' / "
                "'CartesianProduct' is currently supported."
            )

        dim_half = dim // 2
        n = no_of_grid_points
        grid = HypertoroidalGridDistribution.generate_cartesian_product_grid(
            [n] * dim_half
        )

        if fun_does_cartesian_product:
            fvals = fun(grid, grid)
            grid_values = fvals.reshape(n, n)
        else:
            idx_a, idx_b = meshgrid(arange(n), arange(n), indexing="ij")
            grid_a = grid[idx_a.ravel()]
            grid_b = grid[idx_b.ravel()]
            fvals = fun(grid_a, grid_b)

            if fvals.shape == (n**2, n**2):
                raise ValueError(
                    "Function apparently performs the Cartesian product itself. "
                    "Set fun_does_cartesian_product=True."
                )

            grid_values = fvals.reshape(n, n)

        return TdCondTdGridDistribution(grid, grid_values)
