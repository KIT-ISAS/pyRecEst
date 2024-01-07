import copy

from .abstract_manifold_specific_filter import AbstractManifoldSpecificFilter


class AbstractHypertoroidalFilter(AbstractManifoldSpecificFilter):
    """
    This class serves as an intermediate inheritance stage between
    AbstractHypertoroidalFilter and any filters specifically designed for toroidal data.
    Additional functionality specific to toroidal filters should be implemented here.
    Currently, it does not add any functionality to AbstractHypertoroidalFilter.
    """

    def __init__(self, filter_state=None):
        self._filter_state = copy.deepcopy(filter_state)

    def get_point_estimate(self):
        """
        Get the point estimate.

        This method is responsible for getting the point estimate.

        :return: The mean direction of the filter state.
        """
        return self.filter_state.mean_direction()
