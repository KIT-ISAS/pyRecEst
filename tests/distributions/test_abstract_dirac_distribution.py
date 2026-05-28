import importlib.util
import os
import subprocess
import sys
import unittest
import warnings

import matplotlib
import numpy.testing as npt
import pytest

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import array, random
from pyrecest.distributions import LinearDiracDistribution


class TestAbstractDiracDistribution(unittest.TestCase):
    def test_sample_selects_complete_multidimensional_diracs(self):
        dist = LinearDiracDistribution(
            array(
                [
                    [0.0, 1.0],
                    [2.0, 3.0],
                    [4.0, 5.0],
                ]
            ),
            array([0.0, 0.0, 1.0]),
        )

        samples = dist.sample(4)

        npt.assert_allclose(
            samples,
            array(
                [
                    [4.0, 5.0],
                    [4.0, 5.0],
                    [4.0, 5.0],
                    [4.0, 5.0],
                ]
            ),
        )

    def test_sample_rejects_invalid_count(self):
        dist = LinearDiracDistribution(
            array([[0.0], [1.0]]),
            array([0.25, 0.75]),
        )

        for n in (0, -1, 1.5, True):
            with self.subTest(n=n):
                with self.assertRaisesRegex(ValueError, "positive integer"):
                    dist.sample(n)

    def test_mode_returns_highest_weighted_dirac(self):
        dist = LinearDiracDistribution(
            array(
                [
                    [0.0, 0.0],
                    [1.0, 2.0],
                    [3.0, 4.0],
                ]
            ),
            array([0.1, 0.7, 0.2]),
        )

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            mode = dist.mode()

        npt.assert_allclose(mode, array([1.0, 2.0]))

    def test_apply_function_non_vectorized_maps_complete_diracs(self):
        dist = LinearDiracDistribution(
            array(
                [
                    [0.0, 1.0],
                    [2.0, 3.0],
                ]
            ),
            array([0.5, 0.5]),
        )

        mapped = dist.apply_function(
            lambda point: array([point[1], point[0] + point[1]]),
            function_is_vectorized=False,
        )

        npt.assert_allclose(
            mapped.d,
            array(
                [
                    [1.0, 1.0],
                    [3.0, 5.0],
                ]
            ),
        )

    def test_rejects_negative_weights(self):
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            LinearDiracDistribution(
                array(
                    [
                        [0.0],
                        [1.0],
                    ]
                ),
                array([1.2, -0.2]),
            )

    def test_rejects_zero_total_weight(self):
        with self.assertRaisesRegex(ValueError, "positive finite total mass"):
            LinearDiracDistribution(
                array(
                    [
                        [0.0],
                        [1.0],
                    ]
                ),
                array([0.0, 0.0]),
            )

    def test_rejects_nonfinite_weights(self):
        with self.assertRaisesRegex(ValueError, "finite"):
            LinearDiracDistribution(
                array(
                    [
                        [0.0],
                        [1.0],
                    ]
                ),
                array([float("inf"), 1.0]),
            )

    def test_normalizes_valid_unnormalized_weights(self):
        with self.assertWarns(RuntimeWarning):
            dist = LinearDiracDistribution(
                array(
                    [
                        [0.0],
                        [1.0],
                    ]
                ),
                array([2.0, 1.0]),
            )

        npt.assert_allclose(dist.w, array([2.0 / 3.0, 1.0 / 3.0]))

    def test_rejects_weight_count_mismatch(self):
        with self.assertRaisesRegex(ValueError, "Number of Diracs"):
            LinearDiracDistribution(
                array(
                    [
                        [0.0],
                        [1.0],
                    ]
                ),
                array([1.0]),
            )

    def test_reweigh_rejects_wrong_output_shape(self):
        dist = LinearDiracDistribution(
            array([[0.0], [1.0]]),
            array([0.5, 0.5]),
        )

        with self.assertRaisesRegex(ValueError, "wrong output dimensions"):
            dist.reweigh(lambda _: array([[1.0, 1.0]]))

    def test_reweigh_rejects_zero_posterior_weight_mass(self):
        dist = LinearDiracDistribution(
            array(
                [
                    [0.0],
                    [1.0],
                ]
            ),
            array([1.0, 0.0]),
        )

        with self.assertRaisesRegex(ValueError, "positive finite total mass"):
            dist.reweigh(lambda _: array([0.0, 1.0]))

    def _test_plot_helper(self, name, dist, dim, dirac_cls, **kwargs):
        if dirac_cls is None:
            return  # Prevent failure if no classes are set

        matplotlib.pyplot.close("all")
        matplotlib.use("Agg")

        # Seed the random number generator for reproducibility
        random.seed(0)
        # Sample data and create LinearDiracDistribution instance
        # pylint: disable=not-callable
        ddist = dirac_cls(d=dist.sample(10), w=None, **kwargs)

        try:
            # Attempt to plot
            ddist.plot()
        except (ValueError, RuntimeError) as e:
            self.fail(f"{name}: Plotting failed for dimension {dim} with error: {e}")


@pytest.mark.backend_portable
def test_pytorch_non_vectorized_dirac_transform_maps_particles_not_coordinates():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is not installed")

    env = os.environ.copy()
    env["PYRECEST_BACKEND"] = "pytorch"
    src_path = os.path.abspath("src")
    env["PYTHONPATH"] = (
        src_path
        if not env.get("PYTHONPATH")
        else os.pathsep.join([src_path, env["PYTHONPATH"]])
    )

    code = """
import numpy.testing as npt
from pyrecest.backend import array, to_numpy
from pyrecest.distributions import LinearDiracDistribution

dist = LinearDiracDistribution(
    array([[0.0, 1.0], [2.0, 3.0]]),
    array([0.5, 0.5]),
)
mapped = dist.apply_function(
    lambda point: array([point[1], point[0] + point[1]]),
    function_is_vectorized=False,
)
npt.assert_allclose(to_numpy(mapped.d), [[1.0, 1.0], [3.0, 5.0]])
"""
    subprocess.run([sys.executable, "-c", code], check=True, env=env)
