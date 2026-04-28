"""Quaternion helpers shared by SO(3) distributions."""

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import all, array, linalg, ndim, reshape, where


def as_quaternion_batch(quaternions):
    """Return scalar-last quaternions as a flattened ``(n, 4)`` array."""
    quaternions = array(quaternions, dtype=float)
    if ndim(quaternions) == 1:
        assert quaternions.shape[0] == 4, "SO(3) quaternions must have length 4."
        quaternions = reshape(quaternions, (1, 4))
    else:
        assert quaternions.shape[-1] == 4, "SO(3) quaternions must have length 4."
        quaternions = reshape(quaternions, (-1, 4))
    return quaternions


def normalize_quaternions(quaternions):
    """Return canonical scalar-last unit quaternions."""
    quaternions = as_quaternion_batch(quaternions)
    norms = linalg.norm(quaternions, None, -1)
    assert all(norms > 0.0), "SO(3) quaternions must be nonzero."

    normalized = quaternions / reshape(norms, (-1, 1))
    sign = where(normalized[:, -1:] < 0.0, -1.0, 1.0)
    return sign * normalized
