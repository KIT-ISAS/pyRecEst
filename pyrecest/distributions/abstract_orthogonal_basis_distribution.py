from pyrecest.backend import real
from pyrecest.backend import imag
from pyrecest.backend import exp
from pyrecest.backend import all
import copy
import warnings
from abc import abstractmethod

from beartype import beartype

from .abstract_distribution_type import AbstractDistributionType


class AbstractOrthogonalBasisDistribution(AbstractDistributionType):
    """
    Abstract base class for distributions based on orthogonal basis functions.
    """

    def __init__(self, coeff_mat, transformation):
        """
        Initialize the distribution.

        :param coeff_mat: Coefficient matrix.
        :param transformation: Transformation function. Possible values are "sqrt", "identity", "log".
        """
        self.transformation = transformation
        self.coeff_mat = coeff_mat
        self.normalize_in_place()

    @abstractmethod
    def normalize_in_place(self):
        """
        Abstract method to normalize the distribution. Implementation required in subclasses.
        """

    @abstractmethod
    def value(self, xs):
        """
        Abstract method to get value of the distribution for given input. Implementation required in subclasses.

        :param xs: Input data for value calculation.
        """

    def normalize(self):
        """
        Normalizes the distribution.

        :return: Normalized distribution.
        """
        result = copy.deepcopy(self)
        return result.normalize_in_place()

    def pdf(self, xs):
        """
        Calculates probability density function for the given input.

        :param xa: Input data for PDF calculation.
        :return: PDF value.
        """
        val = self.value(xs)
        if self.transformation == "sqrt":
            assert all(imag(val) < 0.0001)
            return real(val) ** 2

        if self.transformation == "identity":
            return val

        if self.transformation == "log":
            warnings.warn("Density may not be normalized")
            return exp(val)

        raise ValueError("Transformation not recognized or unsupported")