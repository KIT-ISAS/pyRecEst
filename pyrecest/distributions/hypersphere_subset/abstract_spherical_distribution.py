import matplotlib.pyplot as plt
from pyrecest.backend import cos, linspace, meshgrid, sin, pi

from .abstract_hyperspherical_distribution import AbstractHypersphericalDistribution
from .abstract_sphere_subset_distribution import AbstractSphereSubsetDistribution


class AbstractSphericalDistribution(
    AbstractSphereSubsetDistribution, AbstractHypersphericalDistribution
):
    def __init__(self):
        AbstractSphereSubsetDistribution.__init__(self)
        AbstractHypersphericalDistribution.__init__(self, dim=2)

    @staticmethod
    def plot_unit_sphere():
        # Define the number of points to generate around the circle
        num_points = 1000

        # Generate theta and phi angles (in radians)
        theta = linspace(0, 2 * pi, num_points)
        phi = linspace(0, pi, num_points)

        # Create a meshgrid for theta and phi angles
        theta, phi = meshgrid(theta, phi)

        # Calculate the x, y, and z coordinates
        x = sin(phi) * cos(theta)
        y = sin(phi) * sin(theta)
        z = cos(phi)

        # Plot the unit circle in 3D space
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_surface(x, y, z, color="c", alpha=0.7)

        ax.set_xlabel("X-axis")
        ax.set_ylabel("Y-axis")
        ax.set_zlabel("Z-axis")
        ax.set_title("Unit Circle in 3D Space")

        plt.show()
