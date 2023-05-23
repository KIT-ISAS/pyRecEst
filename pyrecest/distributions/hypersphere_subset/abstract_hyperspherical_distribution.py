from abc import abstractmethod

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

from .abstract_hypersphere_subset_distribution import (
    AbstractHypersphereSubsetDistribution,
)


class AbstractHypersphericalDistribution(AbstractHypersphereSubsetDistribution):
    @abstractmethod
    def pdf(self, xs):
        pass

    def mean(self):
        return self.mean_direction()

    def sample_metropolis_hastings(
        self, n, burn_in=10, skipping=5, proposal=None, start_point=None
    ):
        if proposal is None:
            # For unimodal densities, other proposals may be far better.
            from .hyperspherical_uniform_distribution import (
                HypersphericalUniformDistribution,
            )

            def proposal(_):
                return HypersphericalUniformDistribution(self.dim).sample(1)

        if start_point is None:
            start_point = HypersphericalUniformDistribution(self.dim).sample(1)
        # Call the sample_metropolis_hastings method of AbstractDistribution
        # pylint: disable=duplicate-code
        return super().sample_metropolis_hastings(
            n,
            burn_in=burn_in,
            skipping=skipping,
            proposal=proposal,
            start_point=start_point,
        )

    def plot(self, faces=100, grid_faces=20):
        if self.dim == 1:
            phi = np.linspace(0, 2 * np.pi, 320)
            x = np.array([np.sin(phi), np.cos(phi)])
            p = self.pdf(x)
            plt.plot(phi, p)
            plt.show()

        elif self.dim == 2:
            x_sphere_outer, y_sphere_outer, z_sphere_outer = self.create_sphere(
                grid_faces
            )
            x_sphere_inner, y_sphere_inner, z_sphere_inner = self.create_sphere(faces)

            c_sphere = self.pdf(
                np.array(
                    [
                        x_sphere_inner.flatten(),
                        y_sphere_inner.flatten(),
                        z_sphere_inner.flatten(),
                    ]
                ).T
            ).reshape(x_sphere_inner.shape)

            x_sphere_inner *= 0.99
            y_sphere_inner *= 0.99
            z_sphere_inner *= 0.99

            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")
            if grid_faces > 0:
                print("x_sphere_outer.shape:", x_sphere_outer.shape)
                print("y_sphere_outer.shape:", y_sphere_outer.shape)
                print("z_sphere_outer.shape:", z_sphere_outer.shape)
                ax.plot_surface(
                    x_sphere_outer,
                    y_sphere_outer,
                    z_sphere_outer,
                    facecolors=None,
                    edgecolors="k",
                    alpha=0.1,
                    linewidth=0.5,
                )
            print("x_sphere_inner.shape:", x_sphere_inner.shape)
            print("y_sphere_inner.shape:", y_sphere_inner.shape)
            print("z_sphere_inner.shape:", z_sphere_inner.shape)
            print("c_sphere.shape:", c_sphere.shape)
            ax.plot_surface(
                x_sphere_inner,
                y_sphere_inner,
                z_sphere_inner,
                cmap="viridis",
                rstride=1,
                cstride=1,
                facecolors=plt.cm.viridis(c_sphere),  # pylint: disable=no-member
                shade=False,
            )
            ax.set_box_aspect([1, 1, 1])
            plt.colorbar(plt.cm.ScalarMappable(cmap="viridis"), ax=ax)
            plt.show()
        else:
            raise ValueError(
                "Cannot plot hyperspherical distribution with this number of dimensions."
            )

    def moment(self):
        return self.moment_numerical()

    @staticmethod
    def get_full_integration_boundaries(dim):
        if dim == 1:
            return [0, 2 * np.pi]

        return np.vstack(
            (
                np.zeros((dim)),
                np.concatenate(([2 * np.pi], np.pi * np.ones(dim - 1))),
            )
        ).T

    def integrate(self, integration_boundaries=None):
        if integration_boundaries is None:
            integration_boundaries = self.__class__.get_full_integration_boundaries(
                self.dim
            )

        return super().integrate(integration_boundaries)

    def integrate_numerically(self, integration_boundaries=None):
        if integration_boundaries is None:
            integration_boundaries = self.__class__.get_full_integration_boundaries(
                self.dim
            )

        return super().integrate_numerically(integration_boundaries)

    def entropy(self):
        return super().entropy_numerical()

    def mode_numerical(self):
        def fun(s):
            return -self.pdf(AbstractHypersphereSubsetDistribution.polar2cart(s))

        s0 = np.random.rand(self.dim) * np.pi
        res = minimize(
            fun,
            s0,
            method="BFGS",
            options={"disp": False, "gtol": 1e-12, "maxiter": 2000},
        )
        m = AbstractHypersphereSubsetDistribution.polar2cart(res.x)
        return m

    def hellinger_distance(self, other):
        return super().hellinger_distance_numerical(other)

    def total_variation_distance(self, other):
        return super().total_variation_distance_numerical(other)

    @staticmethod
    def create_sphere(faces):
        phi, theta = np.mgrid[
            0.0 : np.pi : complex(0, faces),  # noqa: E203
            0.0 : 2.0 * np.pi : complex(0, faces),  # noqa: E203
        ]
        x = np.sin(phi) * np.cos(theta)
        y = np.sin(phi) * np.sin(theta)
        z = np.cos(phi)
        return x, y, z

    @staticmethod
    def integrate_fun_over_domain(f_hypersph_coords, dim):
        integration_boundaries = (
            AbstractHypersphericalDistribution.get_full_integration_boundaries(dim)
        )
        return AbstractHypersphereSubsetDistribution.integrate_fun_over_domain_part(
            f_hypersph_coords, dim, integration_boundaries
        )

    @staticmethod
    def plot_unit_sphere():
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
        return AbstractHypersphereSubsetDistribution.compute_unit_hypersphere_surface(
            self.dim
        )
