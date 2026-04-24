import warnings

# pylint: disable=redefined-builtin,no-name-in-module,no-member
from pyrecest.backend import (
    abs,
    all,
    any,
    mean,
    sum,
)
from pyrecest.distributions.hypersphere_subset.abstract_hypersphere_subset_distribution import (
    AbstractHypersphereSubsetDistribution,
)

from .abstract_conditional_distribution import AbstractConditionalDistribution


class SdCondSdGridDistribution(AbstractConditionalDistribution):
    """
    Conditional distribution on Sd x Sd represented by a grid of values.

    For a conditional distribution f(a|b), ``grid_values[i, j]`` stores
    the value f(grid[i] | grid[j]).

    Convention
    ----------
    - ``grid`` has shape ``(n_points, d)`` where ``d`` is the embedding
      dimension of the individual sphere (e.g. d=3 for S2).
    - ``grid_values`` has shape ``(n_points, n_points)``.
    - ``dim = 2 * d`` is the embedding dimension of the Cartesian product
      space (convention inherited from libDirectional).
    """

    def __init__(self, grid, grid_values, enforce_pdf_nonnegative=True):
        """
        Parameters
        ----------
        grid : array of shape (n_points, d)
            Grid points on the sphere.  All coordinate values must lie in
            [-1, 1] (unit sphere).
        grid_values : array of shape (n_points, n_points)
            Conditional pdf values: ``grid_values[i, j] = f(grid[i] | grid[j])``.
            Must be non-negative.
        enforce_pdf_nonnegative : bool
            Whether non-negativity of ``grid_values`` is required.
        """
        super().__init__(grid, grid_values, enforce_pdf_nonnegative)
        if any(abs(self.grid) > 1 + 1e-12):
            raise ValueError(
                "Grid points must have coordinates in [-1, 1] (unit sphere)."
            )
        self._check_normalization()

    # ------------------------------------------------------------------
    # Normalization
    # ------------------------------------------------------------------

    def _check_normalization(self, tol=0.01):
        """Warn if any column is not normalized to 1 over the sphere."""
        sphere_surface = (
            AbstractHypersphereSubsetDistribution.compute_unit_hypersphere_surface(
                self.grid.shape[1] - 1
            )
        )
        # For each fixed second argument j, the mean over i times the sphere
        # surface area should equal 1.
        ints = mean(self.grid_values, axis=0) * sphere_surface
        if any(abs(ints - 1) > tol):
            # Check whether swapping the two arguments would yield normalisation.
            ints_swapped = mean(self.grid_values, axis=1) * sphere_surface
            if all(abs(ints_swapped - 1) <= tol):
                raise ValueError(
                    "Normalization:maybeWrongOrder: Not normalized but would be if "
                    "the order of the two spheres were swapped. Check input."
                )
            warnings.warn(
                "Normalization:notNormalized: When conditioning values for the first "
                "sphere on the second, normalisation is not ensured. "
                "Check input or increase tolerance. "
                "No normalisation is performed; you may want to do this manually.",
                UserWarning,
            )

    # ------------------------------------------------------------------
    # Marginalisation and conditioning
    # ------------------------------------------------------------------

    def marginalize_out(self, first_or_second):
        """
        Marginalize out one of the two spheres.

        Parameters
        ----------
        first_or_second : int  (1 or 2)
            ``1`` marginalizes out the *first* dimension (sums over rows);
            ``2`` marginalizes out the *second* dimension (sums over columns).

        Returns
        -------
        HypersphericalGridDistribution
        """
        # Import here to avoid circular imports
        from pyrecest.distributions.hypersphere_subset.hyperspherical_grid_distribution import (  # pylint: disable=import-outside-toplevel
            HypersphericalGridDistribution,
        )

        if first_or_second == 1:
            grid_values_sgd = sum(self.grid_values, axis=0)
        elif first_or_second == 2:
            grid_values_sgd = sum(self.grid_values, axis=1)
        else:
            raise ValueError("first_or_second must be 1 or 2.")

        return HypersphericalGridDistribution(self.grid, grid_values_sgd)

    def fix_dim(self, first_or_second, point):
        """
        Return the conditional slice for a fixed grid point.

        The supplied ``point`` must be an existing grid point.

        Parameters
        ----------
        first_or_second : int  (1 or 2)
            ``1`` fixes the *first* argument at ``point`` and returns values
            over the second sphere;
            ``2`` fixes the *second* argument and returns values over the
            first sphere.
        point : array of shape (d,)
            Grid point at which to evaluate.

        Returns
        -------
        HypersphericalGridDistribution
        """
        # Import here to avoid circular imports
        from pyrecest.distributions.hypersphere_subset.hyperspherical_grid_distribution import (  # pylint: disable=import-outside-toplevel
            HypersphericalGridDistribution,
        )

        grid_values_slice = self._get_grid_slice(first_or_second, point)
        return HypersphericalGridDistribution(self.grid, grid_values_slice)

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @staticmethod
    def from_function(
        fun,
        no_of_grid_points,
        fun_does_cartesian_product=False,
        grid_type="leopardi",
        dim=6,
    ):
        """
        Construct a :class:`SdCondSdGridDistribution` from a callable.

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
            Number of grid points for each sphere.
        fun_does_cartesian_product : bool
            See ``fun`` description above.
        grid_type : str
            Grid type passed to :func:`~pyrecest.sampling.hyperspherical_sampler.get_grid_hypersphere`.
            Defaults to ``'leopardi'``.
        dim : int
            Embedding dimension of the Cartesian product space
            (``2 * embedding_dim_of_individual_sphere``).
            Defaults to 6 (S2 × S2).

        Returns
        -------
        SdCondSdGridDistribution
        """
        # Import inside the function to avoid circular imports at module level.
        from pyrecest.sampling.hyperspherical_sampler import (  # pylint: disable=import-outside-toplevel
            get_grid_hypersphere,
        )

        n = no_of_grid_points
        # Convert from Cartesian-product embedding dim to individual sphere
        # manifold dim: embedding_dim = dim // 2, manifold_dim = embedding_dim - 1.
        manifold_dim = dim // 2 - 1
        grid, _ = get_grid_hypersphere(grid_type, n, manifold_dim)

        grid_values = SdCondSdGridDistribution._evaluate_on_grid(
            fun, grid, n, fun_does_cartesian_product
        )
        return SdCondSdGridDistribution(grid, grid_values)
