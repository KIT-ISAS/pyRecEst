import copy
import warnings
from abc import abstractmethod

import numpy as np

from .abstract_distribution_type import AbstractDistributionType
from beartype import beartype
from typing import Union

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
    def value(self, xs: Union[np.ndarray, np.number]) -> Union[np.ndarray, np.number]:
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

    @beartype
    def pdf(self, xs: Union[np.ndarray, np.number]) -> Union[np.ndarray, np.number]:
        """
        Calculates probability density function for the given input.

        :param xa: Input data for PDF calculation.
        :return: PDF value.
        """
        val = self.value(xs)
        if self.transformation == "sqrt":
            assert np.all(np.imag(val) < 0.0001)
            return np.real(val) ** 2

        if self.transformation == "identity":
            return val

        if self.transformation == "log":
            warnings.warn("Density may not be normalized")
            return np.exp(val)

        raise ValueError("Transformation not recognized or unsupported")
