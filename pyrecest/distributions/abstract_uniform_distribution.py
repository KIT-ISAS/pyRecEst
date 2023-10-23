from abc import abstractmethod

from pyrecest.backend import ones

from .abstract_distribution_type import AbstractDistributionType


class AbstractUniformDistribution(AbstractDistributionType):
    """Abstract class for a uniform distribution on a manifold."""

    def pdf(self, xs):
        """Compute the probability density function at each point in xs.

        :param xs: Points at which to compute the pdf.
        :type xs:

        :return: The pdf evaluated at each point in xs.
        :rtype:
        """
        return 1 / self.get_manifold_size() * ones(xs.shape[0])

    @abstractmethod
    def get_manifold_size(self):
        """
        Compute the probability density function at each point in xs.

        :param xs: Points at which to compute the pdf.
        :type xs:

        :return: The pdf evaluated at each point in xs.
        :rtype:
        """

    def mode(self):
        """Mode is not defined for a uniform distribution."""
        raise AttributeError("Mode not available for uniform distribution")
