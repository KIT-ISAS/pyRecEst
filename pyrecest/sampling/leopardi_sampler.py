""" Implementation of the equal area partition of the sphere S^dim into N regions by Paul Leopardi. See
"A partition of the unit sphere into regions of equal area and small diameter", Paul Leopardi, 2006.
The code follows Leopardi's Matlab implementation of the algorithm. The coverage of the "lower" halves of
the sphere is different than in his original code. Note that the order of hyperspherical coordinates
output by get_partition_points_polar follow Leopardi's convention and not that used in pyrecest.
For obtaning Cartesian coordinates, see LeopardiSampler in hyperspherical_sampler.py.
"""

from math import pi

# pylint: disable=redefined-builtin,no-name-in-module,no-member
from pyrecest.backend import (
    abs,
    arange,
    array,
    int32,
    linspace,
    max,
    ones,
    reshape,
    round,
    sin,
    zeros,
    zeros_like,
)
from scipy.optimize import root_scalar
from scipy.special import betainc  # pylint: disable=E0611

from ..distributions import AbstractHypersphereSubsetDistribution


def get_cap_area(dim, colatitude):
    """
    Compute the surface area of a spherical cap on S^dim, given its colatitude.

    Parameters:
    - dim (int): Dimension of the sphere.
    - colatitude: Colatitude angle in radians, in [0, pi].

    Returns:
    - area: Area of the spherical cap.
    """
    if dim == 1:
        return 2 * colatitude
    if dim == 2:
        return 4 * pi * sin(colatitude / 2) ** 2

    area_sphere = (
        AbstractHypersphereSubsetDistribution.compute_unit_hypersphere_surface(dim)
    )
    x = sin(colatitude / 2) ** 2
    a = dim / 2
    b = dim / 2
    area = area_sphere * betainc(a, b, x)
    return area


def get_cap_colatitude(dim, area):
    """
    Compute the colatitude of a spherical cap on S^dim, given its area.

    Parameters:
    - dim : Dimension of the sphere.
    - area: Area of the spherical cap.

    Returns:
    - colatitude: Colatitude angle in radians, in [0, pi].
    """
    colatitude = zeros_like(area)

    area_sphere = (
        AbstractHypersphereSubsetDistribution.compute_unit_hypersphere_surface(dim)
    )

    def f(s, dim, area):
        result = get_cap_area(dim, s) - area
        if result < 0 and abs(result) < 1e-12:
            # Adjust for numerical errors near zero
            result = 0
        return result

    if area >= area_sphere:
        colatitude = pi
    elif area <= 0:
        colatitude = 0
    else:
        if area > area_sphere / 2:
            flipped = True
            area = area_sphere - area
        else:
            flipped = False
        res = root_scalar(f, args=(dim, area), bracket=[0, pi / 2], method="bisect")
        s = res.root
        if flipped:
            colatitude = pi - s
        else:
            colatitude = s
    return colatitude


def get_region_area(dim, N):
    """
    Compute the area of one region in an equal area partition of S^dim into N regions.

    Parameters:
    - dim (int): Dimension of the sphere.
    - N (int): Number of regions.

    Returns:
    - area: Area of one region.
    """
    return array(
        AbstractHypersphereSubsetDistribution.compute_unit_hypersphere_surface(dim) / N
    )


def get_polar_cap_colatitude(dim, N):
    """
    Compute the colatitude of the polar cap for the EQ partitioning.

    Parameters:
    - dim (int): Dimension of the sphere.
    - N (int): Number of regions.

    Returns:
    - c_polar: Colatitude angle in radians.
    """
    if N == 1:
        return array(pi)

    if N == 2:
        return array(pi / 2)

    area = get_region_area(dim, N)
    return get_cap_colatitude(dim, area)


def get_ideal_collar_angle(dim, N):
    """
    Compute the ideal collar angle for EQ partitioning.

    Parameters:
    - dim (int): Dimension of the sphere.
    - N (int): Number of regions.

    Returns:
    - angle: Ideal collar angle in radians.
    """
    area = get_region_area(dim, N)
    angle = area ** (1 / dim)
    return angle


def get_number_of_collars(N, c_polar, ideal_angle):
    """
    Determine the number of collars between the polar caps.

    Parameters:
    - N: Number of regions.
    - c_polar: Colatitude of polar cap.
    - ideal_angle: Ideal collar angle.

    Returns:
    - n_collars: Number of collars.
    """
    if not ((N > 2) & (ideal_angle > 0)):
        return 0

    return max(array((1, round((pi - 2 * c_polar) / ideal_angle)), dtype=int32))


def get_ideal_region_counts(dim, N, c_polar, n_collars):
    """
    Get the ideal (real) number of regions in each zone (collars and caps).

    Parameters:
    - dim (int): Dimension of the sphere.
    - N (int): Total number of regions.
    - c_polar: Colatitude of the polar cap.
    - n_collars (int): Number of collars.

    Returns:
    - r_regions (ndarray): Ideal number of regions in each zone.
    """
    r_regions = zeros(n_collars + 2)
    r_regions[0] = 1  # North polar cap
    ideal_region_area = get_region_area(dim, N)
    if n_collars > 0:
        a_fitting = (pi - 2 * c_polar) / n_collars
        for collar_n in range(1, n_collars + 1):
            a_top = c_polar + (collar_n - 1) * a_fitting
            a_bot = c_polar + collar_n * a_fitting
            collar_area = get_cap_area(dim, a_bot) - get_cap_area(dim, a_top)
            r_regions[collar_n] = collar_area / ideal_region_area
    r_regions[-1] = 1  # South polar cap
    return r_regions


def round_region_counts(region_counts):
    """
    Round region counts to integers summing to N, minimizing total discrepancy.

    Parameters:
    - N (int): Total number of regions.
    - region_counts (array_like): Ideal (real) region counts.

    Returns:
    - n_regions (ndarray): Rounded region counts summing to N.
    """
    n_regions = zeros_like(region_counts, dtype=int)
    discrepancy = 0
    for zone_n in range(len(region_counts)):
        n_regions[zone_n] = int(round(region_counts[zone_n] + discrepancy))
        discrepancy += region_counts[zone_n] - n_regions[zone_n]
    return n_regions


def get_cap_colatitudes(dim, N, c_polar, n_regions):
    """
    Compute the colatitudes of caps that enclose cumulative sums of regions.

    Parameters:
    - dim (int): Dimension of the sphere.
    - N (int): Total number of regions.
    - c_polar (float): Colatitude of the polar cap.
    - n_regions (array_like): Number of regions in each zone.

    Returns:
    - c_caps (ndarray): Colatitudes of caps.
    """
    n_collars = len(n_regions) - 2
    c_caps = zeros(n_collars + 2)
    c_caps[0] = c_polar
    ideal_region_area = get_region_area(dim, N)
    subtotal_n_regions = n_regions[0]
    for collar_n in range(1, n_collars + 1):
        subtotal_n_regions += n_regions[collar_n]
        area = subtotal_n_regions * ideal_region_area
        c_caps[collar_n] = get_cap_colatitude(dim, area)
    c_caps[-1] = pi
    return c_caps


def get_equal_area_caps(dim, N):
    """
    Partition the sphere into nested spherical caps and get the number of regions in each zone.

    Parameters:
    - dim (int): Dimension of the sphere.
    - N (int): Total number of regions.

    Returns:
    - cap_colatitudes (ndarray): Colatitudes of caps (in increasing order).
    - n_regions (ndarray): Number of regions in each zone.
    """
    if dim == 1:
        sector = arange(1, N + 1)
        cap_colatitudes = sector * 2 * pi / N
        n_regions = ones(N, dtype=int)
        return cap_colatitudes, n_regions

    if N == 1:
        cap_colatitudes = array([pi])
        n_regions = array([1])
        return cap_colatitudes, n_regions

    c_polar = get_polar_cap_colatitude(dim, N)
    ideal_angle = get_ideal_collar_angle(dim, N)
    n_collars = get_number_of_collars(N, c_polar, ideal_angle)
    region_counts = get_ideal_region_counts(dim, N, c_polar, n_collars)
    n_regions = round_region_counts(region_counts)
    cap_colatitudes = get_cap_colatitudes(dim, N, c_polar, n_regions)
    return cap_colatitudes, n_regions


# pylint: disable=R0914
def get_partition_points_polar(dim, N, extra_offset=False):
    """
    Get the center points of the regions in the EQ partition, in spherical polar coordinates.

    Parameters:
    - dim (int): Dimension of the sphere.
    - N (int): Total number of regions.
    - extra_offset (bool): Whether to use extra offsets (experimental).

    Returns:
    - points_s (ndarray): Spherical polar coordinates of the points (shape: [dim, N]).
    """
    if N == 1:
        return zeros((dim, 1))

    if dim == 1:
        points_s = linspace(0, 2 * pi, N, endpoint=False) + pi / N
        points_s = reshape(points_s, (1, N))
        return points_s

    cap_colatitudes, n_regions = get_equal_area_caps(dim, N)
    n_collars = len(n_regions) - 2  # Excluding the two polar caps
    points_s = zeros((dim, N))
    point_n = 0
    # North polar cap center point
    points_s[:, point_n] = zeros(dim)
    point_n += 1
    # For each collar
    for collar_n in range(n_collars):
        a_top = cap_colatitudes[collar_n]
        a_bot = cap_colatitudes[collar_n + 1]
        n_in_collar = n_regions[collar_n + 1]
        # Recursively partition the (dim-1)-sphere
        points_1 = get_partition_points_polar(dim - 1, n_in_collar, extra_offset)
        num_points = points_1.shape[1]
        # For each point, construct a new point in dim dimensions
        for i in range(num_points):
            point = zeros(dim)
            point[:-1] = points_1[:, i]
            point[-1] = (a_top + a_bot) / 2
            points_s[:, point_n] = point
            point_n += 1
    # South polar cap center point
    points_s[:, point_n] = zeros(dim)
    points_s[-1, point_n] = pi
    point_n += 1
    return points_s
