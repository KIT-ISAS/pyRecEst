import os
import subprocess
import sys
import unittest

import numpy as np
import numpy.testing as npt

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import array, diag
from pyrecest.distributions import EllipsoidalBallUniformDistribution
from pyrecest.exceptions import ShapeError, ValidationError


class TestEllipsoidalBallUniformDistribution(unittest.TestCase):
    def test_pdf(self):
        dist = EllipsoidalBallUniformDistribution(
            array([0.0, 0.0, 0.0]), diag(array([4.0, 9.0, 16.0]))
        )
        npt.assert_allclose(dist.pdf(array([0.0, 0.0, 0.0])), 1 / 100.53096491)

    def test_pdf_accepts_scalar_and_list_inputs_for_one_dimensional_ball(self):
        dist = EllipsoidalBallUniformDistribution(array([0.0]), diag(array([1.0])))

        scalar_pdf = dist.pdf(0.0)
        list_pdf = dist.pdf([0.0, 0.5, 2.0])
        array_pdf = dist.pdf(array([0.0, 0.5, 2.0]))

        npt.assert_allclose(scalar_pdf, array_pdf[0])
        npt.assert_allclose(list_pdf, array_pdf)
        npt.assert_allclose(array_pdf[-1], 0.0)

    def test_pdf_accepts_list_inputs_for_multidimensional_ball(self):
        dist = EllipsoidalBallUniformDistribution(
            array([0.0, 0.0]), diag(array([1.0, 1.0]))
        )

        single_pdf = dist.pdf([0.0, 0.0])
        batch_pdf = dist.pdf([[0.0, 0.0], [2.0, 0.0]])
        array_pdf = dist.pdf(array([[0.0, 0.0], [2.0, 0.0]]))

        npt.assert_allclose(single_pdf, array_pdf[0])
        npt.assert_allclose(batch_pdf, array_pdf)
        npt.assert_allclose(batch_pdf[-1], 0.0)

    def test_pdf_rejects_wrong_point_dimension(self):
        dist = EllipsoidalBallUniformDistribution(
            array([0.0, 0.0]), diag(array([1.0, 1.0]))
        )

        with self.assertRaisesRegex(ShapeError, "xs"):
            dist.pdf([0.0, 0.0, 0.0])

    def test_pdf_rejects_dimension_mismatch(self):
        dist = EllipsoidalBallUniformDistribution(
            array([0.0, 0.0]), diag(array([1.0, 1.0]))
        )

        with self.assertRaisesRegex(ShapeError, "xs"):
            dist.pdf(array([0.0]))

    def test_mean_and_covariance(self):
        center = array([2.0, 3.0])
        shape_matrix = array([[4.0, 3.0], [3.0, 9.0]])
        dist = EllipsoidalBallUniformDistribution(center, shape_matrix)

        npt.assert_allclose(dist.mean(), center)
        npt.assert_allclose(dist.covariance(), shape_matrix / (dist.dim + 2))

    def test_rejects_invalid_shapes(self):
        with self.assertRaisesRegex(ShapeError, "center"):
            EllipsoidalBallUniformDistribution(
                array([[0.0, 0.0]]), diag(array([1.0, 1.0]))
            )

        with self.assertRaisesRegex(ShapeError, "shape_matrix"):
            EllipsoidalBallUniformDistribution(
                array([0.0, 0.0]), array([[1.0, 0.0, 0.0]])
            )

    def test_rejects_invalid_shape_matrix(self):
        center = array([0.0, 0.0])
        invalid_shape_matrices = [
            array([[1.0, 2.0], [0.0, 1.0]]),
            array([[1.0, 0.0], [0.0, 0.0]]),
            array([[1.0, 0.0], [0.0, -1.0]]),
            array([[1.0, 0.0], [0.0, float("nan")]]),
        ]

        for shape_matrix in invalid_shape_matrices:
            with self.subTest(shape_matrix=shape_matrix):
                with self.assertRaises(ValidationError):
                    EllipsoidalBallUniformDistribution(center, shape_matrix)

    def test_validation_survives_optimized_python(self):
        env = os.environ.copy()
        src_path = os.path.abspath("src")
        env["PYTHONPATH"] = (
            src_path
            if not env.get("PYTHONPATH")
            else os.pathsep.join([src_path, env["PYTHONPATH"]])
        )

        code = """
from pyrecest.backend import array, diag
from pyrecest.distributions import EllipsoidalBallUniformDistribution
from pyrecest.exceptions import ShapeError, ValidationError

valid = EllipsoidalBallUniformDistribution(array([0.0, 0.0]), diag(array([1.0, 1.0])))
cases = (
    lambda: EllipsoidalBallUniformDistribution(array([[0.0, 0.0]]), diag(array([1.0, 1.0]))),
    lambda: EllipsoidalBallUniformDistribution(array([0.0, 0.0]), array([[1.0, 0.0, 0.0]])),
    lambda: EllipsoidalBallUniformDistribution(array([0.0, 0.0]), array([[1.0, 2.0], [0.0, 1.0]])),
    lambda: valid.pdf(array([0.0])),
)
for case in cases:
    try:
        case()
    except (ShapeError, ValidationError):
        pass
    else:
        raise AssertionError("invalid ellipsoidal ball input was accepted under optimized Python")
"""
        subprocess.run([sys.executable, "-O", "-c", code], check=True, env=env)

    def test_sampling(self):
        dist = EllipsoidalBallUniformDistribution(
            array([2.0, 3.0]), array([[4.0, 3.0], [3.0, 9.0]])
        )
        samples = dist.sample(10)
        self.assertEqual(samples.shape[-1], dist.dim)
        self.assertEqual(samples.shape[0], 10.0)
        p = dist.pdf(samples)
        self.assertTrue(all(p == p[0]))

    def test_sampling_accepts_integer_like_count(self):
        dist = EllipsoidalBallUniformDistribution(
            array([2.0, 3.0]), array([[4.0, 3.0], [3.0, 9.0]])
        )

        samples = dist.sample(np.int64(4))

        self.assertEqual(samples.shape, (4, dist.dim))

    def test_sampling_rejects_invalid_count(self):
        dist = EllipsoidalBallUniformDistribution(
            array([2.0, 3.0]), array([[4.0, 3.0], [3.0, 9.0]])
        )

        for n in (0, -1, 1.5, True):
            with self.subTest(n=n):
                with self.assertRaisesRegex(ValueError, "positive integer"):
                    dist.sample(n)


if __name__ == "__main__":
    unittest.main()
