import unittest
from math import pi
import numpy.testing as npt

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import array, cos, linalg, sin, random, max, abs, concatenate, column_stack, stack, atleast_2d
from pyrecest.distributions import VonMisesFisherDistribution
from parameterized import parameterized
from pyrecest.distributions import AbstractHypersphereSubsetDistribution


class TestAbstractHypersphereSubsetDistribution(unittest.TestCase):
    @staticmethod
    def hyperspherical_to_cartesian_2_sphere(angles):
        """
        Converts hyperspherical angles to Cartesian coordinates for a 2-sphere (embedded in R^3).
        """
        theta, phi = angles[:, 0], angles[:, 1]

        x = sin(theta) * sin(phi)
        y = sin(theta) * cos(phi)
        z = cos(theta)
        return column_stack([x, y, z])

    @staticmethod
    def hyperspherical_to_cartesian_3_sphere(angles):
        """
        Converts hyperspherical angles to Cartesian coordinates for a 3-sphere (embedded in R^4).
        """
        theta1, theta2, theta3 = angles[:, 0], angles[:, 1], angles[:, 2]
        
        w = sin(theta1) * sin(theta2) * sin(theta3)
        x = sin(theta1) * sin(theta2) * cos(theta3)
        y = sin(theta1) * cos(theta2)
        z = cos(theta1)
        return column_stack([w, x, y, z])

    @staticmethod
    def hyperspherical_to_cartesian_4_sphere(angles):
        """
        Converts hyperspherical angles to Cartesian coordinates for a 4-sphere (embedded in R^5).
        """
        theta1, theta2, theta3, theta4 = angles[:, 0], angles[:, 1], angles[:, 2], angles[:, 3]
        
        v = sin(theta1) * sin(theta2) * sin(theta3) * sin(theta4)
        w = sin(theta1) * sin(theta2) * sin(theta3) * cos(theta4)
        x = sin(theta1) * sin(theta2) * cos(theta3)
        y = sin(theta1) * cos(theta2)
        z = cos(theta1)
        return column_stack([v, w, x, y, z])

    # Now we test these explicit implementations against the provided generic function
    @parameterized.expand([
        (2, hyperspherical_to_cartesian_2_sphere),
        (3, hyperspherical_to_cartesian_3_sphere),
        (4, hyperspherical_to_cartesian_4_sphere),
    ])
    def test_hyperspherical_to_cartesian_specific(self, dimensions, specific_function):
        num_tests: int = 10
        
        angles = 2.0 * pi * random.uniform(size=dimensions)    
        cartesian_specific = specific_function(atleast_2d(angles)).squeeze()
        cartesian_given = AbstractHypersphereSubsetDistribution.hypersph_to_cart(angles)
        npt.assert_allclose(cartesian_specific, cartesian_given)
        
    def test_pdf_hyperspherical_coords_1d(self):
        mu_ = array([0.5, 1.0]) / linalg.norm(array([0.5, 1.0]))
        kappa_ = 2.0
        vmf = VonMisesFisherDistribution(mu_, kappa_)

        pdf_hyperspherical = vmf.gen_pdf_hyperspherical_coords()

        def fangles_1d(phi):
            return vmf.pdf(array([sin(phi), cos(phi)]).T)

        phi_test = array([1.0, 2.0, 0.0, 0.3, 1.1])

        npt.assert_array_almost_equal(
            pdf_hyperspherical(phi_test), fangles_1d(phi_test)
        )

    def test_pdf_hyperspherical_coords_2d(self):
        mu_ = array([0.5, 1.0, 1.0]) / linalg.norm(array([0.5, 1.0, 1.0]))
        kappa_ = 2.0
        vmf = VonMisesFisherDistribution(mu_, kappa_)

        pdf_hyperspherical = vmf.gen_pdf_hyperspherical_coords()

        def fangles_2d(phi1, phi2):
            return vmf.pdf(TestAbstractHypersphereSubsetDistribution.hyperspherical_to_cartesian_2_sphere(array([phi1, phi2]).T))

        phi1_test = array([1.0, 2.0, 0.0, 0.3, 1.1])
        phi2_test = array([2.0, 3.0, 0.1, 3.0, 1.1])

        npt.assert_array_almost_equal(
            pdf_hyperspherical(array([phi1_test, phi2_test]).T),
            fangles_2d(phi1_test, phi2_test)
        )

    def test_pdf_hyperspherical_coords_3d(self):
        mu_ = array([0.5, 1.0, 1.0, -0.5]) / linalg.norm(array([0.5, 1.0, 1.0, -0.5]))
        kappa_ = 2.0
        vmf = VonMisesFisherDistribution(mu_, kappa_)

        pdf_hyperspherical = vmf.gen_pdf_hyperspherical_coords()

        def fangles_3d(phi1, phi2, phi3):
            return vmf.pdf(TestAbstractHypersphereSubsetDistribution.hyperspherical_to_cartesian_3_sphere(array([phi1, phi2, phi3]).T))

        phi1_test = array([1.0, 2.0, 0.0, 0.3, 1.1])
        phi2_test = array([2.0, 3.0, 0.1, 3.0, 1.1])
        phi3_test = phi2_test + 0.2

        npt.assert_array_almost_equal(
            pdf_hyperspherical(stack((phi1_test, phi2_test, phi3_test), axis=1)),
            fangles_3d(phi1_test, phi2_test, phi3_test)
        )


if __name__ == "__main__":
    unittest.main()