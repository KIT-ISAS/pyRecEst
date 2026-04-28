"""Shared helper functions for SO(3) distributions."""

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import (
    abs,
    all,
    arccos,
    array,
    clip,
    linalg,
    ndim,
    reshape,
    sum,
    where,
)


def as_batch(values, width, name):
    """Return values with a fixed trailing width as a two-dimensional batch."""
    values = array(values, dtype=float)
    if ndim(values) == 1:
        assert values.shape[0] == width, f"{name} must have length {width}."
        values = reshape(values, (1, width))
    else:
        assert values.shape[-1] == width, f"{name} must have length {width}."
        values = reshape(values, (-1, width))
    return values


def normalize_quaternions(quaternions):
    """Return canonical scalar-last unit quaternions shaped ``(n, 4)``."""
    quaternions = as_batch(quaternions, 4, "SO(3) quaternions")
    norms = linalg.norm(quaternions, None, -1)
    assert all(norms > 0.0), "SO(3) quaternions must be nonzero."

    normalized = quaternions / reshape(norms, (-1, 1))
    sign = where(normalized[:, -1:] < 0.0, -1.0, 1.0)
    return sign * normalized


def geodesic_distance(rotation_a, rotation_b):
    """Return the SO(3) geodesic distance between quaternions in radians."""
    quat_a = normalize_quaternions(rotation_a)
    quat_b = normalize_quaternions(rotation_b)
    inner = abs(sum(quat_a * quat_b, axis=-1))
    return 2.0 * arccos(clip(inner, 0.0, 1.0))
