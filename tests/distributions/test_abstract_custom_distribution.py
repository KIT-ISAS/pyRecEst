import numpy as np
import pytest

from pyrecest.distributions.abstract_custom_distribution import (
    AbstractCustomDistribution,
)


class ScalarIntegralCustomDistribution(AbstractCustomDistribution):
    dim = 1
    input_dim = 1

    def __init__(self, base_integral):
        super().__init__(lambda xs: np.ones_like(xs))
        self.base_integral = base_integral

    def integrate(self, integration_boundaries=None):
        return self.scale_by * self.base_integral


def test_normalize_verify_accepts_scalar_integral():
    dist = ScalarIntegralCustomDistribution(2.0)

    normalized = dist.normalize(verify=True)

    assert normalized.scale_by == pytest.approx(0.5)
    assert normalized.integrate() == pytest.approx(1.0)


def test_normalize_verify_accepts_one_element_array_integral():
    dist = ScalarIntegralCustomDistribution(np.array([2.0]))

    normalized = dist.normalize(verify=True)

    assert normalized.scale_by == pytest.approx(0.5)
    assert normalized.integrate() == pytest.approx(np.array([1.0]))


def test_normalize_rejects_vector_integral():
    dist = ScalarIntegralCustomDistribution(np.array([1.0, 2.0]))

    with pytest.raises(ValueError, match="scalar integral"):
        dist.normalize()
