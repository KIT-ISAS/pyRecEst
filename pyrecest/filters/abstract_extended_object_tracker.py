from abc import abstractmethod

from .abstract_tracker_with_logging import AbstractTrackerWithLogging

# pylint: disable=too-many-instance-attributes
class AbstractExtendedObjectTracker(AbstractTrackerWithLogging):
    def __init__(self, log_prior_estimates=True, log_posterior_estimates=True, log_prior_extent=True, log_posterior_extent=True):
        super().__init__(log_prior_estimates=log_prior_estimates, 
                         log_posterior_estimates=log_posterior_estimates, 
                         log_prior_extent=log_prior_extent, 
                         log_posterior_extent=log_posterior_extent)

    def store_prior_estimates(self):
        curr_ests = self.get_point_estimate()
        # pylint: disable=W0201
        self.prior_estimates_over_time = self._store_estimates(
            curr_ests, self.prior_estimates_over_time
        )

    def store_posterior_estimates(self):
        curr_ests = self.get_point_estimate()
        # pylint: disable=W0201
        self.posterior_estimates_over_time = self._store_estimates(
            curr_ests, self.posterior_estimates_over_time
        )

    def store_prior_extent(self):
        curr_ext = self.get_point_estimate_extent()
        # pylint: disable=W0201
        self.prior_extents_over_time = self._store_estimates(
            curr_ext, self.prior_extents_over_time
        )

    def store_posterior_extent(self):
        curr_ext = self.get_point_estimate_extent()
        # pylint: disable=W0201
        self.posterior_extents_over_time = self._store_estimates(
            curr_ext, self.posterior_extents_over_time
        )

    @abstractmethod
    def get_point_estimate(self):
        """
        Retrieve the estimated kinematic state and extent of the object,
        all flattened into a single vector.

        Returns:
        - A vector representing both the estimated kinematic state and extent.
        """

    @abstractmethod
    def get_point_estimate_kinematics(self):
        """
        Retrieve the estimated kinematic state of the object.

        Returns:
        - A vector representing the estimated kinematic state.
        """

    @abstractmethod
    def get_point_estimate_extent(self, flatten_matrix=False):
        """
        Retrieve the estimated extent of the object.

        Parameters:
        - flatten_matrix: whether to flatten the extent matrix or not

        Returns:
        - A matrix or vector representing the estimated extent.
        """
