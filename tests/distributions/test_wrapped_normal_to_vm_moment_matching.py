import numpy as np
from scipy.special import iv

from pyrecest.distributions.circle.wrapped_normal_distribution import (
    WrappedNormalDistribution,
)


def test_wrapped_normal_to_vm_matches_first_trigonometric_moment():
    sigma = 1.5
    wnd = WrappedNormalDistribution(0.7, sigma)

    vm = wnd.to_vm()

    expected_resultant_length = np.exp(-(sigma**2) / 2.0)
    actual_resultant_length = iv(1, vm.kappa) / iv(0, vm.kappa)
    old_small_variance_resultant_length = iv(1, 1.0 / sigma**2) / iv(
        0, 1.0 / sigma**2
    )

    assert np.isclose(vm.mu, wnd.mu)
    assert np.isclose(actual_resultant_length, expected_resultant_length)
    assert not np.isclose(
        old_small_variance_resultant_length, expected_resultant_length
    )


def test_wrapped_normal_sigma_to_kappa_matches_resultant_length():
    sigma = 1.5

    kappa = WrappedNormalDistribution.sigma_to_kappa(sigma)

    expected_resultant_length = np.exp(-(sigma**2) / 2.0)
    actual_resultant_length = iv(1, kappa) / iv(0, kappa)
    assert np.isclose(actual_resultant_length, expected_resultant_length)
