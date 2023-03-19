import numpy as np
from scipy.integrate import quad, nquad
from abstract_periodic_distribution import AbstractPeriodicDistribution
import matplotlib.pyplot as plt

class AbstractHypertoroidalDistribution(AbstractPeriodicDistribution):
    def plot(self, *args, resolution=None):
        if resolution is None:
            resolution = 128

        if self.dim == 1:
            theta = np.linspace(0, 2 * np.pi, resolution)
            ftheta = self.pdf(theta)
            p = plt.plot(theta, ftheta, *args)
            # Call setupAxisCircular('x') function here
        elif self.dim == 2:
            step = 2 * np.pi / resolution
            alpha, beta = np.meshgrid(np.arange(0, 2 * np.pi, step), np.arange(0, 2 * np.pi, step))
            f = self.pdf(np.vstack((alpha.ravel(), beta.ravel())))
            f = f.reshape(alpha.shape)
            p = plt.contourf(alpha, beta, f, *args)
            # Call setupAxisCircular('x', 'y') function here
        elif self.dim == 3:
            # Implement the plot for 3D case as needed
            raise NotImplementedError("Plotting for this dimension is currently not supported")
        else:
            raise ValueError("Plotting for this dimension is currently not supported")
        return p

    def mean_direction(self):
        a = self.trigonometric_moment(1)
        m = np.mod(np.angle(a), 2 * np.pi)
        return m

    def mode(self):
        return self.mode_numerical()

    def mode_numerical(self):
        # Implement the optimization function fminunc equivalent in Python (e.g., using scipy.optimize.minimize)
        raise NotImplementedError("Mode calculation is not implemented")

    def trigonometric_moment(self, n):
        return self.trigonometric_moment_numerical(n)

    def integral(self, l=None, r=None):
        if l is None:
            l = np.zeros((self.dim, 1))
        if r is None:
            r = 2 * np.pi * np.ones((self.dim, 1))

        assert l.shape == (self.dim, 1)
        assert r.shape == (self.dim, 1)

        return self.integral_numerical(l, r)

    def integral_numerical(self, l, r):
        if self.dim == 1:
            result, _ = quad(lambda x: self.pdf(x), l, r)
        elif self.dim == 2:
            result, _ = nquad(lambda x, y: self.pdf(np.array([x, y])), [[l[0], r[0]], [l[1], r[1]]])
        elif self.dim == 3:
            result, _ = nquad(lambda x, y, z: self.pdf(np.array([x, y, z])), [[l[0], r[0]], [l[1], r[1]], [l[2], r[2]]])
        else:
            raise ValueError("Numerical moment calculation for this dimension is currently not supported")
        return result

    def trigonometric_moment_numerical(self, n):
        def f1_real(*args):
            x, y, z = args
            if y is None:
                value = self.pdf(x)
            elif z is None:
                value = self.pdf(np.array([x, y]))
            else:
                value = self.pdf(np.array([x, y, z]))
            return value * np.cos(n * x)

        def f1_imag(*args):
            x, y, z = args
            if y is None:
                value = self.pdf(x)
            elif z is None:
                value = self.pdf(np.array([x, y]))
            else:
                value = self.pdf(np.array([x, y, z]))
            return value * np.sin(n * x)
        
        def f2_real(*args):
            x, y, z = args
            if y is None:
                value = self.pdf(x)
            elif z is None:
                value = self.pdf(np.array([x, y]))
            else:
                value = self.pdf(np.array([x, y, z]))
            return value * np.cos(n * y)

        def f2_imag(*args):
            x, y, z = args
            if y is None:
                value = self.pdf(x)
            elif z is None:
                value = self.pdf(np.array([x, y]))
            else:
                value = self.pdf(np.array([x, y, z]))
            return value * np.sin(n * y)
        
        def f3_real(*args):
            x, y, z = args
            if y is None:
                value = self.pdf(x)
            elif z is None:
                value = self.pdf(np.array([x, y]))
            else:
                value = self.pdf(np.array([x, y, z]))
            return value * np.cos(n * z)

        def f3_imag(*args):
            x, y, z = args
            if y is None:
                value = self.pdf(x)
            elif z is None:
                value = self.pdf(np.array([x, y]))
            else:
                value = self.pdf(np.array([x, y, z]))
            return value * np.sin(n * z)
            
        if self.dim == 1:
            m_real, _ = nquad(f1_real, [(0, 2 * np.pi)])
            m_imag, _ = nquad(f1_imag, [(0, 2 * np.pi)])
            m = m_real + 1j * m_imag
        elif self.dim == 2:
            m = np.zeros(2, dtype=complex)
            m[0], _ = nquad(f1_real, [(0, 2 * np.pi), (0, 2 * np.pi)])
            m[1], _ = nquad(f2_real, [(0, 2 * np.pi), (0, 2 * np.pi)])
            m[0] += 1j * nquad(f1_imag, [(0, 2 * np.pi), (0, 2 * np.pi)])[0]
            m[1] += 1j * nquad(f2_imag, [(0, 2 * np.pi), (0, 2 * np.pi)])[0]
        elif self.dim == 3:
            m = np.zeros(3, dtype=complex)
            m[0], _ = nquad(f1_real, [(0, 2 * np.pi), (0, 2 * np.pi), (0, 2 * np.pi)])
            m[1], _ = nquad(f2_real, [(0, 2 * np.pi), (0, 2 * np.pi), (0, 2 * np.pi)])
            m[2], _ = nquad(f3_real, [(0, 2 * np.pi), (0, 2 * np.pi), (0, 2 * np.pi)])
            m[0] += 1j * nquad(f1_imag, [(0, 2 * np.pi), (0, 2 * np.pi), (0, 2 * np.pi)])[0]
            m[1] += 1j * nquad(f2_imag, [(0, 2 * np.pi), (0, 2 * np.pi), (0, 2 * np.pi)])[0]
            m[2] += 1j * nquad(f3_imag, [(0, 2 * np.pi), (0, 2 * np.pi), (0, 2 * np.pi)])[0]
        else:
            raise NotImplementedError("Numerical moment calculation for this dimension is currently not supported")

        return m

    def to_circular(self):
        if self.dim != 1:
            raise ValueError("Can only convert distributions of dimension 1.")
        return CustomCircularDistribution(lambda x: self.pdf(x))

    def entropy(self):
        return self.entropy_numerical()

    def sample_metropolis_hastings(self, n, proposal=None, start_point=None, burn_in=10, skipping=5):
        if proposal is None:
            proposal = lambda x: np.mod(x + np.random.randn(self.dim, 1), 2 * np.pi)
        if start_point is None:
            start_point = self.mean_direction()

        s = super().sample_metropolis_hastings(n, proposal, start_point, burn_in, skipping)
        return s
    @staticmethod
    def angular_error(alpha, beta):
        assert not np.isnan(alpha).any() and not np.isnan(beta).any()
        
        alpha = np.mod(alpha, 2 * np.pi)
        beta = np.mod(beta, 2 * np.pi)
        
        diff = np.abs(alpha - beta)
        e = np.minimum(diff, 2 * np.pi - diff)
        
        return e
