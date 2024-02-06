import unittest

import numpy.testing as npt

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import array, random, repeat, vstack
from pyrecest.utils.metrics import anees
from pyrecest.evaluation.eot_shape_database import (
    Cross,
    Star,
)
from pyrecest.utils.metrics import iou_polygon
import matplotlib
import matplotlib.pyplot as plt

class TestANEES(unittest.TestCase):
    def setUp(self):
        self.groundtruths = array([[1.5, 2.5], [2.5, 3.5], [4.5, 5.5]])
        self.uncertainties = array(
            [
                [[1.0, 0.5], [0.5, 2.0]],
                [[1.0, 0.0], [0.0, 1.0]],
                [[0.5, 0.0], [0.0, 1.5]],
            ]
        )
        self.n_timesteps_constant = 10000

    def test_ANEES_is_close_to_one(self):
        """Test that the ANEES is close to 1 when we sample from the groundtruths with the given uncertainties.
        Simulate that the state stays constant for 10000 time steps, then changes, stays constant for another 10000 time steps
        and then changes once more before staying constant for the remaining 10000 time steps.
        """
        samples = []

        for i in range(len(self.groundtruths)):
            samples_for_i = random.multivariate_normal(
                mean=self.groundtruths[i],
                cov=self.uncertainties[i],
                size=self.n_timesteps_constant,
            )
            samples.append(samples_for_i)

        samples_mat = vstack(samples)

        repeated_groundtruths = repeat(
            self.groundtruths, repeats=self.n_timesteps_constant, axis=0
        )
        repeated_uncertainties = repeat(
            self.uncertainties, repeats=self.n_timesteps_constant, axis=0
        )

        computed_ANEES = anees(
            samples_mat, repeated_uncertainties, repeated_groundtruths
        )

        # Assert that computed ANEES is close to 1 with a tolerance of 0.05.
        npt.assert_allclose(computed_ANEES, self.groundtruths.shape[-1], atol=0.05)


class TestIoU(unittest.TestCase):
    def test_iou_plolygon(self):
        cross = Cross(2.0, 1.0, 2.0, 3.0)
        self.assertGreater(iou_polygon(cross, Star(0.5)), 0.05)
        self.assertGreater(iou_polygon(cross, Star(1.0)), iou_polygon(cross, Star(0.5)))
        self.assertEqual(iou_polygon(cross, cross), 1.0)


if __name__ == "__main__":
    unittest.main()
