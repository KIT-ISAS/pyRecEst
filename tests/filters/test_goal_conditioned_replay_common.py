# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import array, eye, zeros
from pyrecest.distributions import GaussianDistribution


def assert_association_likelihood_linear_positive(test_case, filt, state_dim):
    """Shared helper: verify association_likelihood_linear returns a positive value."""
    H_vel = zeros((2, state_dim))
    H_vel[:, 2:4] = eye(2)
    meas_noise = GaussianDistribution(zeros(2), 0.05 * eye(2))
    assoc = filt.association_likelihood_linear(
        array([0.2, 0.0]),
        H_vel,
        meas_noise,
    )
    test_case.assertGreater(float(assoc), 0.0)
