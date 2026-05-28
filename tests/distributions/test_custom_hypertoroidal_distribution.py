import unittest

import numpy.testing as npt

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import array
from pyrecest.distributions import CustomHypertoroidalDistribution


class CustomHypertoroidalDistributionTest(unittest.TestCase):
    def test_pdf_accepts_list_inputs_with_list_shift(self):
        dist = CustomHypertoroidalDistribution(
            lambda xs: xs * 0.0 + 1.0, 1, shift_by=[0.1]
        )

        list_pdf = dist.pdf([0.1, 0.2])
        array_pdf = dist.pdf(array([0.1, 0.2]))

        self.assertEqual(list_pdf.shape, (2,))
        npt.assert_allclose(list_pdf, array_pdf)

    def test_pdf_accepts_multidimensional_list_inputs_with_list_shift(self):
        dist = CustomHypertoroidalDistribution(
            lambda xs: xs[:, 0], 2, shift_by=[0.1, 0.2]
        )

        list_pdf = dist.pdf([[0.1, 0.2], [0.3, 0.4]])
        array_pdf = dist.pdf(array([[0.1, 0.2], [0.3, 0.4]]))

        npt.assert_allclose(list_pdf, array_pdf)

    def test_constructor_rejects_wrong_shift_shape(self):
        with self.assertRaisesRegex(ValueError, "shift_by"):
            CustomHypertoroidalDistribution(lambda xs: xs, 2, shift_by=[0.1])


if __name__ == "__main__":
    unittest.main()
