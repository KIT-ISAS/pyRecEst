import unittest

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import array
from pyrecest.distributions.abstract_orthogonal_basis_distribution import (
    AbstractOrthogonalBasisDistribution,
)


class DummyOrthogonalBasisDistribution(AbstractOrthogonalBasisDistribution):
    def __init__(self, values, transformation="sqrt"):
        self._values = values
        super().__init__(coeff_mat=array([1.0]), transformation=transformation)

    def normalize_in_place(self):
        return self

    def value(self, xs):  # pylint: disable=unused-argument
        return self._values


class AbstractOrthogonalBasisDistributionTest(unittest.TestCase):
    def test_pdf_rejects_large_negative_imaginary_part(self):
        dist = DummyOrthogonalBasisDistribution(array([1.0 - 1j]), "sqrt")

        with self.assertRaises(AssertionError):
            dist.pdf(array([0.0]))

    def test_pdf_accepts_small_negative_imaginary_roundoff(self):
        dist = DummyOrthogonalBasisDistribution(array([2.0 - 1e-8j]), "sqrt")

        self.assertAlmostEqual(float(dist.pdf(array([0.0]))[0]), 4.0)


if __name__ == "__main__":
    unittest.main()
