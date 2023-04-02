import numpy as np
from abstract_periodic_distribution import AbstractPeriodicDistribution
from scipy.integrate import quad, nquad
from scipy.special import gamma

class AbstractHypersphereSubsetDistribution(AbstractPeriodicDistribution):
    def mean_direction(self):
        return self.mean_direction_numerical()

    def mean_direction_numerical(self, integration_boundaries):
        raise NotImplementedError("Method is not implemented yet.")

    def moment(self):
        return self.moment_numerical()

    def moment_numerical(self, integration_boundaries):
        raise NotImplementedError("Method is not implemented yet.")

    def mean_axis(self):
        mom = self.moment()
        D, V = np.linalg.eig(mom)
        Dsorted = np.sort(D)
        Vsorted = V[:, D.argsort()]
        if abs(Dsorted[-1] / Dsorted[-2]) < 1.01:
            print("Eigenvalues are very similar. Axis may be unreliable.")
        if Vsorted[-1, -1] >= 0:
            m = Vsorted[:, -1]
        else:
            m = -Vsorted[:, -1]
        return m

    def mean_axis_numerical(self):
        mom = self.moment_numerical()
        D, V = np.linalg.eig(mom)
        Dsorted = np.sort(D)
        Vsorted = V[:, D.argsort()]
        if abs(Dsorted[-1] / Dsorted[-2]) < 1.01:
            print("Eigenvalues are very similar. Axis may be unreliable.")
        if Vsorted[-1, -1] >= 0:
            m = Vsorted[:, -1]
        else:
            m = -Vsorted[:, -1]
        return m

    def integral(self):
        return self.integral_numerical()

    def integral_numerical(this, integration_boundaries):
        dim = this.dim

        if dim == 2:
            f = lambda phi: this.pdf(np.array([np.cos(phi), np.sin(phi)]))
            i, _ = quad(f, integration_boundaries[0, 0], integration_boundaries[0, 1], epsabs=0.01)
        elif dim == 3:
            f = lambda x: this.pdf(x)

            def fangles(phi1, phi2):
                r = 1
                return f(np.array([r * np.sin(phi1) * np.sin(phi2),
                                r * np.cos(phi1) * np.sin(phi2),
                                r * np.cos(phi2)]))

            def g(phi1, phi2):
                return fangles(phi1, phi2) * np.sin(phi2)

            i, _ = nquad(g, [[integration_boundaries[0, 0], integration_boundaries[0, 1]],
                            [integration_boundaries[1, 0], integration_boundaries[1, 1]]],
                        opts={'epsabs': 1e-3, 'epsrel': 1e-3})
        elif dim == 4:
            f = lambda x: this.pdf(x)

            def fangles(phi1, phi2, phi3):
                r = 1
                return f(np.array([r * np.sin(phi1) * np.sin(phi2) * np.sin(phi3),
                                r * np.cos(phi1) * np.sin(phi2) * np.sin(phi3),
                                r * np.cos(phi2) * np.sin(phi3),
                                r * np.cos(phi3)]))

            def g(phi1, phi2, phi3):
                return fangles(phi1, phi2, phi3) * np.sin(phi2) * (np.sin(phi3)) ** 2

            i, _ = nquad(g, [[integration_boundaries[0, 0], integration_boundaries[0, 1]],
                            [integration_boundaries[1, 0], integration_boundaries[1, 1]],
                            [integration_boundaries[2, 0], integration_boundaries[2, 1]]],
                        opts={'epsabs': 1e-3, 'epsrel': 1e-3})
        else:
            raise ValueError("Dimension not supported.")
        
        return i


    def mode(self):
        return self.mode_numerical()

    def mode_numerical(self):
        raise NotImplementedError("Method is not implemented yet.")
    
    @staticmethod
    def polar2cart(polar_coords):
        coords = np.zeros((polar_coords.shape[0], polar_coords.shape[1] + 1,))
        coords[:, 0] = np.sin(polar_coords[:, 0]) * np.cos(polar_coords[:, 1])
        coords[:, 1] = np.sin(polar_coords[:, 0]) * np.sin(polar_coords[:, 1])
        coords[:, 2] = np.cos(polar_coords[:, 0])
        for i in range(2, polar_coords.shape[1]):
            coords[:, :-i] *= np.sin(polar_coords[:, i])
            coords[:, -i] = np.cos(polar_coords[:, i])
        return coords

    @staticmethod
    def compute_unit_hypersphere_surface(dim):
        if dim == 2:
            surface_area = 2 * np.pi
        elif dim == 3:
            surface_area = 4 * np.pi
        elif dim == 4:
            surface_area = 2 * np.pi**2
        else:
            surface_area = 2 * np.pi**(dim / 2) / gamma(dim / 2)
        return surface_area
