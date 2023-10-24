import unittest

import numpy.testing as npt

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import array, random, repeat, stack
from pyrecest.utils.metrics import anees


class TestANEES(unittest.TestCase):
    def setUp(self):
        self.groundtruths = array([[1.5, 2.5], [2.5, 3.5], [4.5, 5.5]])
        self.uncertainties = array(
            [[[1.0, 0.5], [0.5, 2.0]], [[1.0, 0.0], [0.0, 1.0]], [[0.5, 0.0], [0.0, 1.5]]]
        )
        self.num_samples = 10000

    def test_ANEES_is_close_to_one(self):
        samples = []

        for i in range(len(self.groundtruths)):
            samples_for_i = random.multivariate_normal(
                mean=self.groundtruths[i],
                cov=self.uncertainties[i],
                size=self.num_samples,
            )
            samples.append(samples_for_i)

        samples_mat = stack(samples, axis=-1)
        repeated_groundtruths = repeat(self.groundtruths, self.num_samples, axis=0)
        repeated_uncertainties = repeat(self.uncertainties, self.num_samples, axis=0)

        computed_ANEES = anees(samples_mat, repeated_uncertainties, repeated_groundtruths)

        # Assert that computed ANEES is close to 1 with a tolerance of 0.05.
        npt.assert_almost_equal(computed_ANEES, self.groundtruths.shape[-1], decimal=2)


if __name__ == "__main__":
    unittest.main()
