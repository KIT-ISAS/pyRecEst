import itertools
from abc import abstractmethod
from math import pi

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import (
    arange,
    arccos,
    arctan2,
    array,
    column_stack,
    cos,
    deg2rad,
    empty,
    sin,
    sqrt,
    vstack,
)
from pyrecest.distributions import (
    AbstractSphericalDistribution,
    HypersphericalUniformDistribution,
)

from .abstract_sampler import AbstractSampler
from .hypertoroidal_sampler import CircularUniformSampler


def get_grid_hypersphere(method: str, grid_density_parameter: int):
    if method == "healpix":
        samples, grid_specific_description = HealpixSampler().get_grid(
            grid_density_parameter
        )
    elif method == "driscoll_healy":
        samples, grid_specific_description = DriscollHealySampler().get_grid(
            grid_density_parameter
        )
    elif method in ("fibonacci", "spherical_fibonacci"):
        samples, grid_specific_description = SphericalFibonacciSampler().get_grid(
            grid_density_parameter
        )
    elif method == "healpix_hopf":
        samples, grid_specific_description = HealpixHopfSampler().get_grid(
            grid_density_parameter
        )
    else:
        raise ValueError(f"Unknown method {method}")

    return samples, grid_specific_description


get_grid_sphere = get_grid_hypersphere


class AbstractHypersphericalUniformSampler(AbstractSampler):
    def sample_stochastic(self, n_samples: int, dim: int):
        return HypersphericalUniformDistribution(dim).sample(n_samples)

    @abstractmethod
    def get_grid(self, grid_density_parameter: int, dim: int):
        raise NotImplementedError()


class AbstractSphericalUniformSampler(AbstractHypersphericalUniformSampler):
    def sample_stochastic(
        self, n_samples: int, dim: int = 2
    ):  # Only having dim there for interface compatibility
        assert dim == 2
        return HypersphericalUniformDistribution(2).sample(n_samples)


class AbstractSphericalCoordinatesBasedSampler(AbstractSphericalUniformSampler):
    @abstractmethod
    def get_grid_spherical_coordinates(
        self,
        grid_density_parameter: int,
    ):
        raise NotImplementedError()

    def get_grid(self, grid_density_parameter: int, dim: int = 2):
        assert (
            dim == 2
        ), "AbstractSphericalCoordinatesBasedSampler is supposed to be used for the circle (which is one-dimensional) only."
        phi, theta, grid_specific_description = self.get_grid_spherical_coordinates(
            grid_density_parameter
        )
        x, y, z = AbstractSphericalDistribution.sph_to_cart(phi, theta)
        grid = column_stack((x, y, z))

        return grid, grid_specific_description


class HealpixSampler(AbstractHypersphericalUniformSampler):
    def get_grid(self, grid_density_parameter: int):
        import healpy as hp

        n_side = grid_density_parameter
        n_areas = hp.nside2npix(n_side)
        x, y, z = hp.pix2vec(n_side, arange(n_areas))
        grid = column_stack((x, y, z))

        grid_specific_description = {
            "scheme": "healpix",
            "n_side": grid_density_parameter,
        }

        return grid, grid_specific_description


class DriscollHealySampler(AbstractSphericalCoordinatesBasedSampler):
    def get_grid_spherical_coordinates(self, grid_density_parameter: int):
        import pyshtools as pysh

        grid = pysh.SHGrid.from_zeros(grid_density_parameter)

        # Get the longitudes (phi) and latitudes (theta) directly from the grid
        phi_deg_mat = grid.lons()
        theta_deg_mat = grid.lats()

        phi_theta_stacked_deg = array(
            list(itertools.product(phi_deg_mat, theta_deg_mat))
        )
        phi_theta_stacked_rad = deg2rad(phi_theta_stacked_deg)

        phi = phi_theta_stacked_rad[:, 0]
        theta = phi_theta_stacked_rad[:, 1]

        grid_specific_description = {
            "scheme": "driscoll_healy",
            "l_max": grid_density_parameter,
            "n_lat": grid.nlat,
            "n_lon": grid.nlon,
        }

        return phi, theta, grid_specific_description


class SphericalFibonacciSampler(AbstractSphericalCoordinatesBasedSampler):
    def get_grid_spherical_coordinates(self, grid_density_parameter: int):
        indices = arange(0, grid_density_parameter, dtype=float) + 0.5
        phi = pi * (1 + 5**0.5) * indices
        theta = arccos(1 - 2 * indices / grid_density_parameter)
        grid_specific_description = {
            "scheme": "spherical_fibonacci",
            "n_samples": grid_density_parameter,
        }
        return phi, theta, grid_specific_description


class AbstractHopfBasedS3Sampler(AbstractHypersphericalUniformSampler):
    @staticmethod
    def hopf_coordinates_to_quaterion_yershova(θ, ϕ, ψ):
        """
        One possible way to index the S3-sphere via the hopf fibration.
        Using the convention from
        "Generating Uniform Incremental Grids on SO(3) Using the Hopf Fibration"
        by
        Anna Yershova, Swati Jain, Steven M. LaValle, Julie C. Mitchell
        As in appendix (or in Eq 4 if one reorders it).
        """
        quaterions = empty((θ.shape[0], 4))

        quaterions[:, 0] = cos(θ / 2) * cos(ψ / 2)
        quaterions[:, 1] = cos(θ / 2) * sin(ψ / 2)
        quaterions[:, 2] = sin(θ / 2) * cos(ϕ + ψ / 2)
        quaterions[:, 3] = sin(θ / 2) * sin(ϕ + ψ / 2)
        return quaterions

    @staticmethod
    def quaternion_to_hopf_yershova(q):
        θ = 2 * arccos(sqrt(q[:, 0] ** 2 + q[:, 1] ** 2))
        ϕ = arctan2(q[:, 3], q[:, 2]) - arctan2(q[:, 1], q[:, 0])
        ψ = 2 * arctan2(q[:, 1], q[:, 0])
        return θ, ϕ, ψ


# pylint: disable=too-many-locals
class HealpixHopfSampler(AbstractHopfBasedS3Sampler):
    def get_grid(self, grid_density_parameter: int | list[int]):
        """
        Hopf coordinates are (θ, ϕ, ψ) where θ and ϕ are the angles for the sphere and ψ is the angle on the circle
        First parameter is the number of points on the sphere, second parameter is the number of points on the circle.
        """
        import healpy as hp

        if isinstance(grid_density_parameter, int):
            grid_density_parameter = [grid_density_parameter]

        s3_points_list = []

        for i in range(grid_density_parameter[0] + 1):
            if grid_density_parameter.shape[0] == 2:
                n_sample_circle = grid_density_parameter[1]
            else:
                n_sample_circle = 2**i * 6

            psi_points = CircularUniformSampler().get_grid(n_sample_circle)

            assert len(psi_points) != 0

            nside = 2**i
            numpixels = hp.nside2npix(nside)

            healpix_points = empty((numpixels, 2))
            for j in range(numpixels):
                theta, phi = hp.pix2ang(nside, j, nest=True)
                healpix_points[j] = [theta, phi]

            for j in range(len(healpix_points)):
                for k in range(len(psi_points)):
                    temp = array(
                        [healpix_points[j, 0], healpix_points[j, 1], psi_points[k]]
                    )
                    s3_points_list.append(temp)

        s3_points = vstack(s3_points_list)  # Need to stack like this and unpack
        grid = AbstractHopfBasedS3Sampler.hopf_coordinates_to_quaterion_yershova(
            s3_points[:, 0], s3_points[:, 1], s3_points[:, 2]
        )

        grid_specific_description = {
            "scheme": "healpix_hopf",
            "layer-parameter": grid_density_parameter,
        }
        return grid, grid_specific_description


class FibonacciHopfSampler(AbstractHopfBasedS3Sampler):
    def get_grid(self, grid_density_parameter: int | list[int]):
        """
        Hopf coordinates are (θ, ϕ, ψ) where θ and ϕ are the angles for the sphere and ψ is the angle on the circle
        First parameter is the number of points on the sphere, second parameter is the number of points on the circle.
        """
        if isinstance(grid_density_parameter, int):
            grid_density_parameter = [grid_density_parameter]

        s3_points_list = []

        # Step 1: Discretize the sphere using the Fibonacci grid
        spherical_sampler = SphericalFibonacciSampler()
        phi, theta, _ = spherical_sampler.get_grid_spherical_coordinates(
            grid_density_parameter[0]
        )
        spherical_points = column_stack((theta, phi))  # stack to match expected shape

        # Step 2: Discretize the unit circle using the circular grid
        circular_sampler = CircularUniformSampler()
        if len(grid_density_parameter) == 2:
            n_sample_circle = grid_density_parameter[1]
        else:
            n_sample_circle = sqrt(grid_density_parameter[0])
        psi_points = circular_sampler.get_grid(n_sample_circle)

        # Step 3: Combine the two grids to generate a grid for S3
        for spherical_point in spherical_points:
            for psi in psi_points:
                s3_point = array([spherical_point[0], spherical_point[1], psi])
                s3_points_list.append(s3_point)

        s3_points = vstack(s3_points_list)
        grid = AbstractHopfBasedS3Sampler.hopf_coordinates_to_quaterion_yershova(
            s3_points[:, 0], s3_points[:, 1], s3_points[:, 2]
        )

        grid_specific_description = {
            "scheme": "fibonacci_hopf",
            "layer-parameter": grid_density_parameter,
        }
        return grid, grid_specific_description
