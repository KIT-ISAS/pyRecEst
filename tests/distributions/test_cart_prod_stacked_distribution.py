import unittest

from pyrecest.backend import allclose, array, eye
from pyrecest.distributions import GaussianDistribution
from pyrecest.distributions.cart_prod.cart_prod_stacked_distribution import (
    CartProdStackedDistribution,
)


class ConcreteCartProdStackedDistribution(CartProdStackedDistribution):
    @property
    def input_dim(self):
        return sum(dist.input_dim for dist in self.dists)

    def get_manifold_size(self):
        return float("inf")


class TestCartProdStackedDistribution(unittest.TestCase):
    def test_shift_uses_cumulative_component_dimensions(self):
        dist = ConcreteCartProdStackedDistribution(
            [
                GaussianDistribution(array([1.0, 2.0]), eye(2)),
                GaussianDistribution(array([3.0, 4.0, 5.0]), eye(3)),
            ]
        )

        shifted = dist.shift(array([10.0, 20.0, 30.0, 40.0, 50.0]))

        self.assertIsInstance(shifted, ConcreteCartProdStackedDistribution)
        self.assertTrue(allclose(shifted.dists[0].mu, array([11.0, 22.0])))
        self.assertTrue(allclose(shifted.dists[1].mu, array([33.0, 44.0, 55.0])))

    def test_set_mode_preserves_concrete_distribution_type(self):
        dist = ConcreteCartProdStackedDistribution(
            [
                GaussianDistribution(array([1.0, 2.0]), eye(2)),
                GaussianDistribution(array([3.0, 4.0, 5.0]), eye(3)),
            ]
        )

        shifted = dist.set_mode(array([10.0, 20.0, 30.0, 40.0, 50.0]))

        self.assertIsInstance(shifted, ConcreteCartProdStackedDistribution)
        self.assertTrue(allclose(shifted.dists[0].mu, array([10.0, 20.0])))
        self.assertTrue(allclose(shifted.dists[1].mu, array([30.0, 40.0, 50.0])))


if __name__ == "__main__":
    unittest.main()
