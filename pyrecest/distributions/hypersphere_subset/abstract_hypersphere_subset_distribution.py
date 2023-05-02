from abc import abstractmethod

import numpy as np
from scipy.integrate import nquad, quad
from scipy.special import gamma

from ..abstract_periodic_distribution import AbstractPeriodicDistribution


class AbstractHypersphereSubsetDistribution(AbstractPeriodicDistribution):
    def mean_direction(self):
        return self.mean_direction_numerical()

    @staticmethod
    @abstractmethod
    def get_full_integration_boundaries(dim):
        pass

    def mean_direction_numerical(self, integration_boundaries=None):
        if integration_boundaries is None:
            integration_boundaries = self.__class__.get_full_integration_boundaries(
                self.dim
            )

        mu = np.full(self.dim + 1, np.nan)

        if 1 <= self.dim <= 3:
            for i in range(self.dim + 1):

                def f(x, i=i):
                    return x[i] * self.pdf(x)

                fangles = self.gen_fun_hyperspherical_coords(f, self.dim)

                if self.dim == 1:
                    mu[i], _ = quad(
                        fangles,
                        integration_boundaries[0, 0],
                        integration_boundaries[0, 1],
                        epsabs=1e-3,
                        epsrel=1e-3,
                    )
                elif self.dim == 2:
                    mu[i], _ = nquad(
                        fangles,
                        integration_boundaries,
                        opts={"epsabs": 1e-3, "epsrel": 1e-3},
                    )
                elif self.dim == 3:
                    mu[i], _ = nquad(
                        fangles,
                        integration_boundaries,
                        opts={"epsabs": 1e-3, "epsrel": 1e-3},
                    )
        else:
            raise ValueError("Unsupported")

        if np.linalg.norm(mu) < 1e-9:
            print(
                "Warning: Density may not actually have a mean direction because integral yields a point very close to the origin."
            )

        mu = mu / np.linalg.norm(mu)
        return mu

    def gen_pdf_hyperspherical_coords(self):
        """
        Generate the PDF in hyperspherical coordinates.

        :return: A function that computes the PDF value at given angles.
        """
        return AbstractHypersphereSubsetDistribution.gen_fun_hyperspherical_coords(
            self.pdf, self.dim
        )

    @staticmethod
    def gen_fun_hyperspherical_coords(f, dim):
        r = 1

        def generate_input(angles):
            dim_eucl = dim + 1
            # To make it work both when all inputs are single-elementary
            # or multi-elementary, use vstack to ensure they are concatenated
            # along the first dimension, transpose afterward to get an
            # (n, dim,) numpy array
            angles = np.vstack(angles).T
            input_arr = np.zeros((angles.shape[0], dim_eucl))
            # Start at last, which is just cos
            input_arr[:, -1] = r * np.cos(angles[:, -1])
            sin_product = np.sin(angles[:, -1])
            # Now, iterate over all from end to back and accumulate the sines
            for i in range(2, dim_eucl):
                # All except the final one have a cos factor as their last one
                input_arr[:, -i] = r * sin_product * np.cos(angles[:, -i])
                sin_product *= np.sin(angles[:, -i])
            # The last one is all sines
            input_arr[:, 0] = r * sin_product
            return np.squeeze(input_arr)

        def fangles(*angles):
            input_arr = generate_input(angles)
            return f(input_arr)

        return fangles

    def moment(self):
        return self.moment_numerical()

    def moment_numerical(self):
        m = np.full(
            (
                self.dim + 1,
                self.dim + 1,
            ),
            np.nan,
        )

        def f_gen(i, j):
            def f(points):
                return self.pdf(points) * points[i] * points[j]

            return f

        def g_gen(f_hypersph_coords, dim):
            if dim == 1:

                def g_1d(phi):
                    return f_hypersph_coords(phi)

                return g_1d
            elif dim == 2:

                def g_2d(phi1, phi2):
                    return f_hypersph_coords(phi1, phi2) * np.sin(phi2)

                return g_2d
            elif dim == 3:

                def g_3d(phi1, phi2, phi3):
                    return (
                        f_hypersph_coords(phi1, phi2, phi3)
                        * np.sin(phi2)
                        * np.sin(phi3) ** 2
                    )

                return g_3d
            else:
                raise ValueError("Dimension not supported.")

        for i in range(self.dim + 1):
            for j in range(self.dim + 1):
                f_curr = f_gen(i, j)
                fangles = self.__class__.gen_fun_hyperspherical_coords(f_curr, self.dim)
                g_curr = g_gen(fangles, self.dim)
                m[i, j] = self.__class__.integrate_fun_over_domain(g_curr, self.dim)

        return m

    @staticmethod
    def _compute_mean_axis_from_moment(moment_matrix):
        D, V = np.linalg.eig(moment_matrix)
        Dsorted = np.sort(D)
        Vsorted = V[:, D.argsort()]
        if abs(Dsorted[-1] / Dsorted[-2]) < 1.01:
            print("Eigenvalues are very similar. Axis may be unreliable.")
        if Vsorted[-1, -1] >= 0:
            m = Vsorted[:, -1]
        else:
            m = -Vsorted[:, -1]
        return m

    def mean_axis(self):
        mom = self.moment()
        return AbstractHypersphereSubsetDistribution._compute_mean_axis_from_moment(mom)

    def mean_axis_numerical(self):
        mom = self.moment_numerical()
        return AbstractHypersphereSubsetDistribution._compute_mean_axis_from_moment(mom)

    def integrate(self, integration_boundaries=None):
        if integration_boundaries is None:
            integration_boundaries = self.__class__.get_full_integration_boundaries(
                self.dim
            )
        return self.integrate_numerically(integration_boundaries)

    @staticmethod
    @abstractmethod
    def integrate_fun_over_domain(f_hypersph_coords, dim):
        pass

    @staticmethod
    def integrate_fun_over_domain_part(f_hypersph_coords, dim, integration_boundaries):
        if dim == 1:
            i, _ = quad(
                f_hypersph_coords,
                integration_boundaries[0],
                integration_boundaries[1],
                epsabs=0.01,
            )
        elif dim == 2:

            def g_2d(phi1, phi2):
                return f_hypersph_coords(phi1, phi2) * np.sin(phi2)

            i, _ = nquad(
                g_2d,
                integration_boundaries,
                opts={"epsabs": 1e-3, "epsrel": 1e-3},
            )
        elif dim == 3:

            def g_3d(phi1, phi2, phi3):
                return (
                    f_hypersph_coords(phi1, phi2, phi3)
                    * np.sin(phi2)
                    * (np.sin(phi3)) ** 2
                )

            i, _ = nquad(
                g_3d,
                integration_boundaries,
                opts={"epsabs": 1e-3, "epsrel": 1e-3},
            )
        else:
            raise ValueError("Dimension not supported.")

        return i

    def integrate_numerically(self, integration_boundaries=None):
        """integration_boundaries have to be given in (hyper)spherical coordinates"""
        if integration_boundaries is None:
            integration_boundaries = self.__class__.get_full_integration_boundaries(
                self.dim
            )
        f = self.gen_pdf_hyperspherical_coords()
        return AbstractHypersphereSubsetDistribution.integrate_fun_over_domain_part(
            f, self.dim, integration_boundaries
        )

    def mode(self):
        return self.mode_numerical()

    def mode_numerical(self):
        raise NotImplementedError("Method is not implemented yet.")

    def entropy_numerical(self, integration_boundaries=None):
        if integration_boundaries is None:
            integration_boundaries = self.__class__.get_full_integration_boundaries(
                self.dim
            )

        def entropy_f_gen():
            def f(points):
                return self.pdf(points) * np.log(self.pdf(points))

            return f

        f_entropy = entropy_f_gen()
        fangles_entropy = (
            AbstractHypersphereSubsetDistribution.gen_fun_hyperspherical_coords(
                f_entropy, self.dim
            )
        )

        entropy_integral = self.__class__.integrate_fun_over_domain(
            fangles_entropy, self.dim
        )

        return -entropy_integral

    def _distance_f_gen(self, other, distance_func):
        def f(points):
            return distance_func(self.pdf(points), other.pdf(points))

        return f

    def hellinger_distance_numerical(self, other, integration_boundaries=None):
        if integration_boundaries is None:
            integration_boundaries = self.__class__.get_full_integration_boundaries(
                self.dim
            )
        assert (
            self.dim == other.dim
        ), "Cannot compare distributions with different number of dimensions"

        def hellinger_distance(pdf1, pdf2):
            return (np.sqrt(pdf1) - np.sqrt(pdf2)) ** 2

        f_hellinger = self._distance_f_gen(other, hellinger_distance)
        fangles_hellinger = (
            AbstractHypersphereSubsetDistribution.gen_fun_hyperspherical_coords(
                f_hellinger, self.dim
            )
        )

        distance_integral = self.__class__.integrate_fun_over_domain(
            fangles_hellinger, self.dim
        )

        return 0.5 * distance_integral

    def total_variation_distance_numerical(self, other, integration_boundaries=None):
        if integration_boundaries is None:
            integration_boundaries = self.__class__.get_full_integration_boundaries(
                self.dim
            )
        assert (
            self.dim == other.dim
        ), "Cannot compare distributions with different number of dimensions"

        def total_variation_distance(pdf1, pdf2):
            return np.abs(pdf1 - pdf2)

        f_total_variation = self._distance_f_gen(other, total_variation_distance)
        fangles_total_variation = (
            AbstractHypersphereSubsetDistribution.gen_fun_hyperspherical_coords(
                f_total_variation, self.dim
            )
        )

        distance_integral = self.__class__.integrate_fun_over_domain(
            fangles_total_variation, self.dim
        )

        return 0.5 * distance_integral

    @staticmethod
    def polar2cart(polar_coords):
        polar_coords = np.atleast_2d(polar_coords)

        coords = np.zeros(
            (
                polar_coords.shape[0],
                polar_coords.shape[1] + 1,
            )
        )
        coords[:, 0] = np.sin(polar_coords[:, 0]) * np.cos(polar_coords[:, 1])
        coords[:, 1] = np.sin(polar_coords[:, 0]) * np.sin(polar_coords[:, 1])
        coords[:, 2] = np.cos(polar_coords[:, 0])
        for i in range(2, polar_coords.shape[1]):
            coords[:, :-i] *= np.sin(polar_coords[:, i])  # noqa: E203
            coords[:, -i] = np.cos(polar_coords[:, i])
        return np.squeeze(coords)

    @staticmethod
    def compute_unit_hypersphere_surface(dim):
        if dim == 1:
            surface_area = 2 * np.pi
        elif dim == 2:
            surface_area = 4 * np.pi
        elif dim == 3:
            surface_area = 2 * np.pi**2
        else:
            surface_area = 2 * np.pi ** ((dim + 1) / 2) / gamma((dim + 1) / 2)
        return surface_area
