from .abstract_manifold_specific_distribution import (
    AbstractManifoldSpecificDistribution,
)


class AbstractBoundedDomainDistribution(AbstractManifoldSpecificDistribution):
    """
    Abstract class for distributions with bounded domains.

    This class extends the AbstractManifoldSpecificDistribution class, and
    serves as the base class for all distributions that are defined over
    bounded domains. This class does not define any methods or properties and
    is intended to be subclassed by specific distributions.
    """