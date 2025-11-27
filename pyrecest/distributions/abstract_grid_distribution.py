import copy
import warnings
from abc import abstractmethod

from beartype import beartype

from .abstract_distribution_type import AbstractDistributionType

# pylint: disable=redefined-builtin,no-name-in-module,no-member
from pyrecest.backend import mean, abs, any

class AbstractGridDistribution(AbstractDistributionType):
    # pylint: disable=too-many-positional-arguments
    @beartype
    def __init__(
        self,
        grid_values,
        grid_type: str = "custom",
        grid=None,
        dim=None,
        enforce_pdf_nonnegative: bool = True,
    ):
        assert (
            not grid_type == "custom" or grid is not None
        )  # if grid_type is custom, grid needs to be given
        assert (
            grid is None or grid.shape == () or grid.shape[0] == grid_values.shape[0]
        )
        assert (
            grid is None or grid.shape == () or grid.ndim == 1 or grid.shape[1] == dim
        )
        if grid is None or grid.ndim > 1 and grid.shape[0] < grid.shape[1]:
            warnings.warn(
                "Warning: Dimension is higher than number of grid points. Verify that this is really intended."
            )
        self.grid_values = grid_values
        self.grid_type = grid_type
        self.grid = grid
        self.enforce_pdf_nonnegative = enforce_pdf_nonnegative
        # Overwrite with more descriptive parameterization
        self.grid_density_description = {"n_grid_values": grid_values.shape[0], "grid_type": grid_type}

    def pdf(self, xs):
        # Use nearest neighbor interpolation by default
        _, indices = self.get_closest_point(xs)
        return self.grid_values[indices].T

    @property
    def n_grid_points(self):
        # Overwrite if grid_values contains values that are not used as grid values
        return self.grid_values.shape[0]

    @abstractmethod
    def get_closest_point(self, xs):
        pass

    @abstractmethod
    def get_manifold_size(self):
        pass

    def integrate(self, integration_boundaries=None):
        assert (
            integration_boundaries is None
        ), "Custom integration boundaries are currently not supported"
        return self.get_manifold_size() * mean(self.grid_values)

    def normalize_in_place(self, tol=1e-4, warn_unnorm=True):
        int_val = self.integrate()
        if any(self.grid_values < 0):
            warnings.warn(
                "Warning: There are negative values. This usually points to a user error."
            )
        elif abs(int_val) < 1e-200:
            raise ValueError(
                "Sum of grid values is too close to zero, this usually points to a user error."
            )
        elif abs(int_val - 1) > tol:
            if warn_unnorm:
                warnings.warn(
                    "Warning: Grid values apparently do not belong to a normalized density. Normalizing..."
                )

        self.grid_values = self.grid_values / int_val
        return self

    def normalize(self, tol=1e-4, warn_unnorm=True):
        result = copy.deepcopy(self)
        return result.normalize_in_place(tol=tol, warn_unnorm=warn_unnorm)

    def get_grid(self):
        # Overload if .grid should stay empty
        return self.grid

    def get_grid_point(self, indices):
        # To avoid passing all points if only one or few are needed.
        # Overload if .grid should stay empty
        return self.grid[indices, :]

    def multiply(self, other):
        assert self.enforce_pdf_nonnegative == other.enforce_pdf_nonnegative
        gd = copy.deepcopy(self)
        gd.grid_values = gd.grid_values * other.grid_values
        gd = gd.normalize(warn_unnorm=False)
        return gd
