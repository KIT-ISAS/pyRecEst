import numpy as np
import matplotlib.pyplot as plt
from abstract_hypersphere_subset_distribution import AbstractHypersphereSubsetDistribution
from abc import abstractmethod
from scipy.optimize import minimize

class AbstractHypersphericalDistribution(AbstractHypersphereSubsetDistribution):
    @abstractmethod
    def pdf(self, x):
        pass

    def plot(self, faces=100, grid_faces=20):
        if self.dim == 2:
            phi = np.linspace(0, 2 * np.pi, 320)
            x = np.array([np.cos(phi), np.sin(phi)])
            p = self.pdf(x)
            plt.plot(phi, p)
            plt.show()

        elif self.dim == 3:
            x_sphere_outer, y_sphere_outer, z_sphere_outer = self.create_sphere(grid_faces)
            x_sphere_inner, y_sphere_inner, z_sphere_inner = self.create_sphere(faces)

            c_sphere = self.pdf(np.array([x_sphere_inner.flatten(), y_sphere_inner.flatten(), z_sphere_inner.flatten()])).reshape(x_sphere_inner.shape)

            x_sphere_inner *= 0.99
            y_sphere_inner *= 0.99
            z_sphere_inner *= 0.99

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            if grid_faces > 0:
                ax.plot_surface(x_sphere_outer, y_sphere_outer, z_sphere_outer, facecolors='none', edgecolors='k', alpha=0.1, linewidth=0.5)
            ax.plot_surface(x_sphere_inner, y_sphere_inner, z_sphere_inner, cmap='viridis', rstride=1, cstride=1, facecolors=plt.cm.viridis(c_sphere), shade=False)
            ax.set_box_aspect([1, 1, 1])
            plt.colorbar(plt.cm.ScalarMappable(cmap='viridis'), ax=ax)
            plt.show()

        else:
            raise ValueError("Cannot plot hyperspherical distribution with this number of dimensions.")
        
    def moment_numerical(this):
        return super().moment_numerical(np.column_stack((np.zeros(this.dim - 1), np.hstack((2 * np.pi, np.pi * np.ones(this.dim - 2))))))

    def integral_numerical(this):
        if this.dim <= 4:
            i = AbstractHypersphereSubsetDistribution.integral_numerical(this, np.column_stack((np.zeros(this.dim - 1), np.hstack((2 * np.pi, np.pi * np.ones(this.dim - 2))))))
        else:
            from hyperspherical_uniform_distribution import HypersphericalUniformDistribution
            n = 10000
            r = HypersphericalUniformDistribution(this.dim).sample(n)
            p = this.pdf(r)
            Sd = AbstractHypersphericalDistribution.compute_unit_hypersphere_surface(this.dim)
            i = np.sum(p) / n * Sd
        return i

    def entropy_numerical(this):
        return this.entropy_numerical(np.column_stack((np.zeros(this.dim - 1), np.hstack((2 * np.pi, np.pi * np.ones(this.dim - 2))))))

    def mode_numerical(this):
        fun = lambda s: -this.pdf(AbstractHypersphereSubsetDistribution.polar2cart(s))
        s0 = np.random.rand(this.dim - 1) * np.pi
        res = minimize(fun, s0, method='BFGS', options={'disp': False, 'gtol': 1e-12, 'maxiter': 2000})
        m = AbstractHypersphereSubsetDistribution.polar2cart(res.x)
        return m

    def hellinger_distance_numerical(this, other):
        return this.hellinger_distance_numerical(other, np.column_stack((np.zeros(this.dim - 1), np.hstack((2 * np.pi, np.pi * np.ones(this.dim - 2))))))

    def total_variation_distance_numerical(this, other):
        return this.total_variation_distance_numerical(other, np.column_stack((np.zeros(this.dim - 1), np.hstack((2 * np.pi, np.pi * np.ones(this.dim - 2))))))

    @staticmethod
    def create_sphere(faces):
        phi, theta = np.mgrid[0.0:np.pi:complex(0, faces), 0.0:2.0 * np.pi:complex(0, faces)]
        x = np.sin(phi) * np.cos(theta)
        y = np.sin(phi) * np.sin(theta)
        z = np.cos(phi)
        return x, y, z
        
    @staticmethod
    def plot_unit_sphere():
        from mpl_toolkits.mplot3d import Axes3D

        # Define the number of points to generate around the circle
        num_points = 1000

        # Generate theta and phi angles (in radians)
        theta = np.linspace(0, 2 * np.pi, num_points)
        phi = np.linspace(0, np.pi, num_points)

        # Create a meshgrid for theta and phi angles
        theta, phi = np.meshgrid(theta, phi)

        # Calculate the x, y, and z coordinates
        x = np.sin(phi) * np.cos(theta)
        y = np.sin(phi) * np.sin(theta)
        z = np.cos(phi)

        # Plot the unit circle in 3D space
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_surface(x, y, z, color="c", alpha=0.7)

        ax.set_xlabel("X-axis")
        ax.set_ylabel("Y-axis")
        ax.set_zlabel("Z-axis")
        ax.set_title("Unit Circle in 3D Space")

        plt.show()


    def get_manifold_size(self):
        return AbstractHypersphereSubsetDistribution.compute_unit_hypersphere_surface(self.dim)
