from pyrecest.backend import linalg
from pyrecest.backend import sin
from pyrecest.backend import cos
from pyrecest.backend import array
import unittest
import numpy.testing as npt

from pyrecest.distributions import VonMisesFisherDistribution


class TestAbstractHypersphereSubsetDistribution(unittest.TestCase):
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
            r = 1
            return vmf.pdf(
                array(
                    [
                        r * sin(phi1) * sin(phi2),
                        r * cos(phi1) * sin(phi2),
                        r * cos(phi2),
                    ]
                ).T
            )

        phi1_test = array([1.0, 2.0, 0.0, 0.3, 1.1])
        phi2_test = array([2.0, 3.0, 0.1, 3.0, 1.1])

        npt.assert_array_almost_equal(
            pdf_hyperspherical(phi1_test, phi2_test), fangles_2d(phi1_test, phi2_test)
        )

    def test_pdf_hyperspherical_coords_3d(self):
        mu_ = array([0.5, 1.0, 1.0, -0.5]) / linalg.norm(array([0.5, 1.0, 1.0, -0.5]))
        kappa_ = 2.0
        vmf = VonMisesFisherDistribution(mu_, kappa_)

        pdf_hyperspherical = vmf.gen_pdf_hyperspherical_coords()

        def fangles_3d(phi1, phi2, phi3):
            r = 1
            return vmf.pdf(
                array(
                    [
                        r * sin(phi1) * sin(phi2) * sin(phi3),
                        r * cos(phi1) * sin(phi2) * sin(phi3),
                        r * cos(phi2) * sin(phi3),
                        r * cos(phi3),
                    ]
                ).T
            )

        phi1_test = array([1.0, 2.0, 0.0, 0.3, 1.1])
        phi2_test = array([2.0, 3.0, 0.1, 3.0, 1.1])
        phi3_test = phi2_test + 0.2

        npt.assert_array_almost_equal(
            pdf_hyperspherical(phi1_test, phi2_test, phi3_test),
            fangles_3d(phi1_test, phi2_test, phi3_test),
        )


if __name__ == "__main__":
    unittest.main()