from abc import abstractmethod

import numpy as np
from scipy.integrate import nquad, quad
from scipy.special import gamma

from .abstract_periodic_distribution import AbstractPeriodicDistribution


class AbstractHypersphereSubsetDistribution(AbstractPeriodicDistribution):
    def mean_direction(self):
        return self.mean_direction_numerical()

    @abstractmethod
    def _get_full_integration_boundaries(self):
        pass

    def mean_direction_numerical(self, integration_boundaries=None):
        if integration_boundaries is None:
            integration_boundaries = self._get_full_integration_boundaries()

        mu = np.full(self.dim + 1, np.nan)

        if 1 <= self.dim <= 3:
            for i in range(self.dim + 1):

                def f(x):
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

    def moment_numerical(self, integration_boundaries = None):
        if integration_boundaries is None:
            integration_boundaries = self._get_full_integration_boundaries()
        m = np.full((self.dim + 1, self.dim  + 1,), np.nan)

        def f_gen(i, j):
            def f(points):
                return self.pdf(points) * points[i] * points[j]
            return f

        if self.dim == 1:
            for i in range(2):
                for j in range(2):
                    f_curr = f_gen(i, j)
                    
                    fangles_1d = self.gen_fun_hyperspherical_coords(f_curr, self.dim)

                    def g(phi):
                        return fangles_1d(phi)

                    m[i, j], _ = quad(
                        g,
                        integration_boundaries[0, 0],
                        integration_boundaries[0, 1]
                    )

        elif self.dim == 2:
            for i in range(3):
                for j in range(3):
                    f_curr = f_gen(i, j)
                    
                    fangles_2d = self.gen_fun_hyperspherical_coords(f_curr, self.dim)
                    def g(phi1, phi2):
                        return fangles_2d(phi1, phi2) * np.sin(phi2)

                    m[i, j], _ = nquad(
                        g,
                        integration_boundaries
                    )

        elif self.dim == 3:
            for i in range(4):
                for j in range(4):
                    f_curr = f_gen(i, j)

                    fangles_3d = self.gen_fun_hyperspherical_coords(f_curr, self.dim)

                    def g(phi1, phi2, phi3):
                        return fangles_3d(phi1, phi2, phi3) * np.sin(phi2) * np.sin(phi3) ** 2

                    m[i, j], _ = nquad(
                        g,
                        integration_boundaries
                    )
        else:
            raise ValueError("Dimension not supported.")

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

    def integral(self, integration_boundaries=None):
        if integration_boundaries is None:
            integration_boundaries = self._get_full_integration_boundaries()
        return self.integral_numerical(integration_boundaries)

    def integral_numerical(self, integration_boundaries=None):
        if integration_boundaries is None:
            integration_boundaries = self._get_full_integration_boundaries()
        dim = self.dim

        if dim == 1:
            fangles_1d = self.gen_pdf_hyperspherical_coords()

            i, _ = quad(
                fangles_1d,
                integration_boundaries[0],
                integration_boundaries[1],
                epsabs=0.01,
            )
        elif dim == 2:
            fangles_2d = self.gen_pdf_hyperspherical_coords()

            def g_2d(phi1, phi2):
                return fangles_2d(phi1, phi2) * np.sin(phi2)

            i, _ = nquad(
                g_2d,
                integration_boundaries,
                opts={"epsabs": 1e-3, "epsrel": 1e-3},
            )
        elif dim == 3:
            fangles_3d = self.gen_pdf_hyperspherical_coords()

            def g_3d(phi1, phi2, phi3):
                return fangles_3d(phi1, phi2, phi3) * np.sin(phi2) * (np.sin(phi3)) ** 2

            i, _ = nquad(
                g_3d,
                integration_boundaries,
                opts={"epsabs": 1e-3, "epsrel": 1e-3},
            )
        else:
            raise ValueError("Dimension not supported.")

        return i

    def mode(self):
        return self.mode_numerical()

    def mode_numerical(self):
        raise NotImplementedError("Method is not implemented yet.")
    
    def entropy_numerical(self, integration_boundaries = None):
        if integration_boundaries is None:
            integration_boundaries = self._get_full_integration_boundaries()
        
        if self.dim == 1:
            f = lambda phi: self.pdf(np.array([np.cos(phi), np.sin(phi)])) * np.log(self.pdf(np.array([np.cos(phi), np.sin(phi)])))
            fangles_1d = self.gen_fun_hyperspherical_coords(f, self.dim)
            i, _ = quad(fangles_1d, integration_boundaries[0, 0], integration_boundaries[0, 1], epsabs=0.01)
            return -i
        
        elif self.dim == 2:
            f = lambda x: self.pdf(x) * np.log(self.pdf(x))

            fangles_2d = self.gen_fun_hyperspherical_coords(f, self.dim)

            g = lambda phi1, phi2: fangles_2d(phi1, phi2) * np.sin(phi2)
            i, _ = nquad(g, integration_boundaries, opts={"epsabs": 1e-3, "epsrel": 1e-3})
            return -i
        
        elif self.dim == 3:
            f = lambda x: self.pdf(x) * np.log(self.pdf(x))
            
            fangles_3d = self.gen_fun_hyperspherical_coords(f, self.dim)

            g = lambda phi1, phi2, phi3: fangles_3d(phi1, phi2, phi3) * np.sin(phi2) * (np.sin(phi3))**2
            i, _ = nquad(g, integration_boundaries, opts={"epsabs": 1e-3, "epsrel": 1e-3})
            return -i
        
        else:
            raise ValueError("Not supported")
    
    def hellinger_distance_numerical(self, other, integration_boundaries = None):
        if integration_boundaries is None:
            integration_boundaries = self._get_full_integration_boundaries()
        assert self.dim == other.dim, "Cannot compare distributions with different number of dimensions"

        if self.dim == 1:
            f = lambda phi: (np.sqrt(self.pdf(np.array([np.cos(phi), np.sin(phi)]))) - np.sqrt(other.pdf(np.array([np.cos(phi), np.sin(phi)]))))**2
            distance, _ = quad(f, integration_boundaries[0, 0], integration_boundaries[0, 1], epsabs=0.01)
            return 0.5 * distance
        
        elif self.dim == 2:
            f = lambda x: (np.sqrt(self.pdf(x)) - np.sqrt(other.pdf(x)))**2

            fangles_2d = self.gen_fun_hyperspherical_coords(f, self.dim)

            g = lambda phi1, phi2: fangles_2d(phi1, phi2) * np.sin(phi2)
            distance, _ = nquad(g, integration_boundaries, opts={"epsabs": 1e-3, "epsrel": 1e-3})
            return 0.5 * distance
        
        elif self.dim == 3:
            f = lambda x: (np.sqrt(self.pdf(x)) - np.sqrt(other.pdf(x)))**2

            fangles_3d = self.gen_fun_hyperspherical_coords(f, self.dim)

            g = lambda phi1, phi2, phi3: fangles_3d(phi1, phi2, phi3) * np.sin(phi2) * (np.sin(phi3))**2
            distance, _ = nquad(g, integration_boundaries, opts={"epsabs": 1e-3, "epsrel": 1e-3})
            return 0.5 * distance
        
        else:
            raise ValueError("Numerical calculation of Hellinger distance is currently not supported for this dimension.")
    
    def total_variation_distance_numerical(self, other, integration_boundaries = None):
        if integration_boundaries is None:
            integration_boundaries = self._get_full_integration_boundaries()
        assert self.dim == other.dim, "Cannot compare distributions with different number of dimensions"

        if self.dim == 1:
            f = lambda phi: np.abs(self.pdf(np.array([np.cos(phi), np.sin(phi)])) - other.pdf(np.array([np.cos(phi), np.sin(phi)])))
            dist, _ = quad(f, integration_boundaries[0, 0], integration_boundaries[0, 1], epsabs=0.01)
            return 0.5 * dist
        
        elif self.dim == 2:
            f = lambda x: np.abs(self.pdf(x) - other.pdf(x))

            fangles_2d = self.gen_fun_hyperspherical_coords(f, self.dim)

            g = lambda phi1, phi2: fangles_2d(phi1, phi2) * np.sin(phi2)
            dist, _ = nquad(g, integration_boundaries, opts={"epsabs": 1e-3, "epsrel": 1e-3})
            return 0.5 * dist
        
        elif self.dim == 3:
            f = lambda x: np.abs(self.pdf(x) - other.pdf(x))

            fangles_3d = self.gen_fun_hyperspherical_coords(f, self.dim)

            g = lambda phi1, phi2, phi3: fangles_3d(phi1, phi2, phi3) * np.sin(phi2) * (np.sin(phi3))**2
            dist, _ = nquad(g, integration_boundaries, opts={"epsabs": 1e-3, "epsrel": 1e-3})
            return 0.5 * dist
        
        else:
            raise ValueError("Numerical calculation of total variation distance is currently not supported for this dimension.")

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
