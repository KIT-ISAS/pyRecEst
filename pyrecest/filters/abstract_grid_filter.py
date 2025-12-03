import warnings

from beartype import beartype
from pyrecest.distributions.abstract_grid_distribution import AbstractGridDistribution
from pyrecest.distributions.abstract_manifold_specific_distribution import (
    AbstractManifoldSpecificDistribution,
)

from .abstract_filter_type import AbstractFilterType


class AbstractGridFilter(AbstractFilterType):
    @beartype
    def __init__(self, state_init: AbstractGridDistribution):
        AbstractFilterType.__init__(self, state_init)

    @property
    def filter_state(self):
        """Expose the parent property so we can attach a setter to it."""
        return super().filter_state

    @filter_state.setter
    @beartype
    def filter_state(self, new_state: AbstractManifoldSpecificDistribution):
        if not isinstance(new_state, AbstractGridDistribution):
            warnings.warn(
                "new_state is not a GridDistribution. Transforming the distribution with a number of coefficients equal to that of the filter.",
                RuntimeWarning,
            )
            new_state = self.filter_state.from_distribution(
                new_state,
                self.filter_state.grid_values.shape[0],
                self.filter_state.enforce_pdf_nonnegative,
            )
        elif self.filter_state.grid_values.shape != new_state.grid_values.shape:
            warnings.warn(
                "New grid has a different number of grid points.", RuntimeWarning
            )

        self._filter_state = new_state

    def update_nonlinear(self, likelihood, z):
        grid_vals_new = self.filter_state.grid_values * likelihood(
            z, self.filter_state.get_grid()
        )
        assert grid_vals_new.shape == self.filter_state.grid_values.shape

        self.filter_state.grid_values = grid_vals_new
        self.filter_state.normalize_in_place(warn_unnorm=False)

    def plot_filter_state(self):
        self.filter_state.plot_state()
