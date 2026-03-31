"""Abstract base class for filters on the hypersphere with antipodal symmetry"""

# pylint: disable=redefined-builtin,no-name-in-module,no-member
from pyrecest.backend import array

from .abstract_filter import AbstractFilter


def _complex_multiplication(a, b):
    """Multiply two unit complex numbers represented as 2D real vectors."""
    return array([a[0] * b[0] - a[1] * b[1], a[1] * b[0] + a[0] * b[1]])


def _complex_multiplication_derivative(a, b):
    """Jacobian of complex multiplication w.r.t. [a; b]."""
    return array(
        [
            [b[0], -b[1], a[0], -a[1]],
            [b[1], b[0], a[1], a[0]],
        ]
    )


def _quaternion_multiplication(a, b):
    """Multiply two unit quaternions represented as 4D real vectors [w, x, y, z]."""
    return array(
        [
            a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
            a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
            a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
            a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0],
        ]
    )


def _quaternion_multiplication_derivative(a, b):
    """Jacobian of quaternion multiplication w.r.t. [a; b]."""
    return array(
        [
            [b[0], -b[1], -b[2], -b[3], a[0], -a[1], -a[2], -a[3]],
            [b[1], b[0], b[3], -b[2], a[1], a[0], -a[3], a[2]],
            [b[2], -b[3], b[0], b[1], a[2], a[3], a[0], -a[1]],
            [b[3], b[2], -b[1], b[0], a[3], -a[2], a[1], a[0]],
        ]
    )


class AbstractAxialFilter(AbstractFilter):
    """Abstract base class for filters on the hypersphere with antipodal symmetry."""

    def __init__(self, initial_filter_state=None):
        AbstractFilter.__init__(self, initial_filter_state)
        self.composition_operator = None
        self.composition_operator_derivative = None

    def _set_composition_operator(self):
        """Set the composition operator and its derivative based on the filter dimension.

        For dim=1 (embedding_dim=2): uses complex multiplication.
        For dim=3 (embedding_dim=4): uses quaternion multiplication.
        """
        embedding_dim = self.dim + 1
        if embedding_dim == 2:
            self.composition_operator = _complex_multiplication
            self.composition_operator_derivative = _complex_multiplication_derivative
        elif embedding_dim == 4:
            self.composition_operator = _quaternion_multiplication
            self.composition_operator_derivative = _quaternion_multiplication_derivative
        else:
            raise ValueError("Invalid dimension")

    def get_point_estimate(self):
        return self.filter_state.mode()
