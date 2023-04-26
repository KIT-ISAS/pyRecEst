from .abstract_linear_distribution import AbstractLinearDistribution
from .custom_distribution import CustomDistribution


class CustomLinearDistribution(AbstractLinearDistribution, CustomDistribution):
    """
    Linear distribution with custom pdf.
    """

    def __init__(self, f, dim):
        """
        Constructor, it is the user's responsibility to ensure that f is a valid
        linear density.

        Parameters:
        f_ (function handle)
            pdf of the distribution
        dim_ (scalar)
            dimension of the Euclidean space
        """
        CustomDistribution.__init__(self, f, dim)

    @staticmethod
    def from_distribution(dist):
        """
        Creates a CustomLinearDistribution from some other distribution

        Parameters:
        dist (AbstractLinearDistribution)
            distribution to convert

        Returns:
        chd (CustomLinearDistribution)
            CustomLinearDistribution with identical pdf
        """
        chd = CustomLinearDistribution(dist.pdf, dist.dim)
        return chd

    def integrate(self, left=None, right=None):
        return AbstractLinearDistribution.integrate(self, left, right)
