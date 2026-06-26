import numpy.testing as npt

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import array
from pyrecest.distributions import LinearDiracDistribution


def test_linear_dirac_distribution_flattens_column_weights_for_sampling():
    dist = LinearDiracDistribution(
        array(
            [
                [0.0],
                [1.0],
            ]
        ),
        array([[0.0], [1.0]]),
    )

    npt.assert_allclose(dist.w, array([0.0, 1.0]))
    npt.assert_allclose(
        dist.sample(3),
        array(
            [
                [1.0],
                [1.0],
                [1.0],
            ]
        ),
    )
