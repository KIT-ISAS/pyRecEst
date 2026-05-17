import unittest

from pyrecest.backend import array
from pyrecest.distributions.abstract_grid_distribution import AbstractGridDistribution


class DummyGridDistribution(AbstractGridDistribution):
    def __init__(self, grid_values):
        super().__init__(
            grid_values,
            grid_type="custom",
            grid=array([[0.0], [1.0]]),
            dim=1,
        )

    def get_closest_point(self, xs):
        raise NotImplementedError

    def get_manifold_size(self):
        return 1.0


class AbstractGridDistributionTest(unittest.TestCase):
    def test_normalize_rejects_near_zero_integral_even_with_negative_values(self):
        dist = DummyGridDistribution(array([1.0, -1.0]))

        with self.assertRaisesRegex(ValueError, "too close to zero"):
            dist.normalize_in_place(warn_unnorm=False)
