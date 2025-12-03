
from abc import abstractmethod, ABC


class AbstractFilterManifoldMixin:
    @property
    @abstractmethod
    def filter_state(self):
        """
        Contract: Any class inheriting this Mixin must provide 'filter_state'.
        """

    def get_point_estimate(self):
        """
        Get the point estimate.

        This method is responsible for getting the point estimate.

        :return: The mean direction of the filter state.
        """
        return self.filter_state.mean_direction()
    

class EuclideanFilterMixin(AbstractFilterManifoldMixin, ABC):
    def get_point_estimate(self):
        return self.filter_state.mean()

class HypertoroidalFilterMixin(AbstractFilterManifoldMixin, ABC):
    pass

class ToroidalFilterMixin(HypertoroidalFilterMixin, ABC):
    pass

class CircularFilterMixin(HypertoroidalFilterMixin, ABC):
    pass

class AbstractHypersphereSubsetFilter(AbstractFilterManifoldMixin, ABC):
    pass
class HypersphericalFilterMixin(AbstractHypersphereSubsetFilter, ABC):
    pass

class HyperhemisphericalFilterMixin(AbstractHypersphereSubsetFilter, ABC):
    pass

class LinBoundedFilterMixin(AbstractFilterManifoldMixin, ABC):
    pass

class LinPeriodicFilterMixin(LinBoundedFilterMixin, ABC):
    pass

class HypercylindricalFilterMixin(LinPeriodicFilterMixin, ABC):
    def get_point_estimate(self):
        return self.filter_state.hybrid_mean()
    
class SE2FilterMixin(HypercylindricalFilterMixin, ABC):
    pass

