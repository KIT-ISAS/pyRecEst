import numpy.testing as npt
import pytest

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


def test_linear_dirac_distribution_rejects_boolean_weights():
    with pytest.raises(ValueError, match="boolean"):
        LinearDiracDistribution(
            array(
                [
                    [0.0],
                    [1.0],
                ]
            ),
            array([False, True]),
        )


def test_linear_dirac_distribution_rejects_boolean_reweighing_factors():
    dist = LinearDiracDistribution(
        array(
            [
                [0.0],
                [1.0],
            ]
        ),
        array([0.5, 0.5]),
    )

    with pytest.raises(ValueError, match="boolean"):
        dist.reweigh(lambda _: array([False, True]))
