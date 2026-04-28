"""Relaxed S3F prediction helpers for S1 x R2.

The helpers in this module implement the WP1 pilot relaxations for uniform
circular cells:

* R1: replace representative-orientation displacement by a cell average.
* R2: add the unresolved within-cell displacement covariance.

They intentionally call the existing :class:`StateSpaceSubdivisionFilter`
``predict_linear`` method, so the baseline prediction implementation remains
unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pyrecest.backend import asarray as backend_asarray

from .state_space_subdivision_filter import StateSpaceSubdivisionFilter


SUPPORTED_RELAXED_S3F_VARIANTS = ("baseline", "r1", "r1_r2")


@dataclass(frozen=True)
class CircularCellStatistics:
    """Closed-form statistics for uniform circular cells."""

    grid: np.ndarray
    cell_width: float
    body_increment: np.ndarray
    representative_displacements: np.ndarray
    mean_displacements: np.ndarray
    covariance_inflations: np.ndarray


def rotation_matrix(angle: float) -> np.ndarray:
    """Return the 2-D rotation matrix for ``angle``."""

    c = np.cos(angle)
    s = np.sin(angle)
    return np.array([[c, -s], [s, c]], dtype=float)


def rotate_body_increment(angles: np.ndarray, body_increment: np.ndarray) -> np.ndarray:
    """Rotate a body-frame increment for one or more angles.

    Parameters
    ----------
    angles
        Array of angles in radians.
    body_increment
        Vector ``(u_x, u_y)`` in the body frame.
    """

    angles = np.asarray(angles, dtype=float).reshape(-1)
    u = _as_body_increment(body_increment)
    c = np.cos(angles)
    s = np.sin(angles)
    return np.column_stack((c * u[0] - s * u[1], s * u[0] + c * u[1]))


def uniform_circular_cell_statistics(  # pylint: disable=too-many-locals
    n_cells: int,
    body_increment: np.ndarray,
    grid: np.ndarray | None = None,
) -> CircularCellStatistics:
    """Compute R1/R2 statistics for equal-width cells on S1.

    Each cell is centered at the supplied grid point, with width ``2*pi/n``.
    The within-cell orientation is treated as uniform. This matches the
    piecewise-constant marginal used by the S3F pilot.
    """

    if n_cells <= 0:
        raise ValueError("n_cells must be positive.")

    u = _as_body_increment(body_increment)
    centers = (
        np.linspace(0.0, 2.0 * np.pi, n_cells, endpoint=False)
        if grid is None
        else np.asarray(grid, dtype=float).reshape(-1)
    )
    if centers.shape[0] != n_cells:
        raise ValueError("grid must contain exactly n_cells entries.")

    width = 2.0 * np.pi / n_cells
    half_width = 0.5 * width
    first_factor = _safe_sinc(half_width)
    second_factor = _safe_sinc(2.0 * half_width)

    representative_displacements = rotate_body_increment(centers, u)
    mean_displacements = first_factor * representative_displacements

    covariance_inflations = np.empty((n_cells, 2, 2), dtype=float)
    for idx, center in enumerate(centers):
        cos_2 = np.cos(2.0 * center)
        sin_2 = np.sin(2.0 * center)

        e_cos2 = 0.5 * (1.0 + second_factor * cos_2)
        e_sin2 = 0.5 * (1.0 - second_factor * cos_2)
        e_cossin = 0.5 * second_factor * sin_2

        ux, uy = u
        second_moment = np.array(
            [
                [
                    ux * ux * e_cos2 - 2.0 * ux * uy * e_cossin + uy * uy * e_sin2,
                    (ux * ux - uy * uy) * e_cossin + ux * uy * second_factor * cos_2,
                ],
                [
                    (ux * ux - uy * uy) * e_cossin + ux * uy * second_factor * cos_2,
                    ux * ux * e_sin2 + 2.0 * ux * uy * e_cossin + uy * uy * e_cos2,
                ],
            ],
            dtype=float,
        )
        mean = mean_displacements[idx]
        cov = second_moment - np.outer(mean, mean)
        covariance_inflations[idx] = 0.5 * (cov + cov.T)

    return CircularCellStatistics(
        grid=centers,
        cell_width=width,
        body_increment=u,
        representative_displacements=representative_displacements,
        mean_displacements=mean_displacements,
        covariance_inflations=covariance_inflations,
    )


def predict_circular_relaxed(
    filter_: StateSpaceSubdivisionFilter,
    body_increment: np.ndarray,
    variant: str = "r1_r2",
    process_noise_cov: np.ndarray | None = None,
) -> CircularCellStatistics:
    """Predict an ``S1 x R2`` S3F with the selected relaxed variant.

    Parameters
    ----------
    filter_
        State-space subdivision filter whose grid component is an equal-width
        circular grid and whose linear conditional is 2-D.
    body_increment
        Body-frame translation increment applied during this prediction.
    variant
        One of ``"baseline"``, ``"r1"``, or ``"r1_r2"``.
    process_noise_cov
        Optional additive 2x2 process noise shared by all cells.
    """

    if variant not in SUPPORTED_RELAXED_S3F_VARIANTS:
        raise ValueError(
            f"Unknown variant {variant!r}; expected one of "
            f"{SUPPORTED_RELAXED_S3F_VARIANTS}."
        )

    state = filter_.filter_state
    n_cells = len(state.linear_distributions)
    if state.lin_dim != 2:
        raise ValueError("predict_circular_relaxed requires a 2-D linear state.")

    grid = np.asarray(state.gd.get_grid(), dtype=float).reshape(-1)
    stats = uniform_circular_cell_statistics(n_cells, body_increment, grid=grid)

    if variant == "baseline":
        displacements = stats.representative_displacements
        covariance_inflations = np.zeros_like(stats.covariance_inflations)
    elif variant == "r1":
        displacements = stats.mean_displacements
        covariance_inflations = np.zeros_like(stats.covariance_inflations)
    else:
        displacements = stats.mean_displacements
        covariance_inflations = stats.covariance_inflations

    q_base = (
        np.zeros((2, 2), dtype=float)
        if process_noise_cov is None
        else np.asarray(process_noise_cov, dtype=float)
    )
    if q_base.shape != (2, 2):
        raise ValueError("process_noise_cov must have shape (2, 2).")

    covariance_matrices = np.empty((2, 2, n_cells), dtype=float)
    for idx in range(n_cells):
        covariance_matrices[:, :, idx] = q_base + covariance_inflations[idx]

    filter_.predict_linear(
        covariance_matrices=backend_asarray(covariance_matrices),
        linear_input_vectors=backend_asarray(displacements.T),
    )
    return stats


def grid_probability_masses(grid_values: np.ndarray) -> np.ndarray:
    """Convert equal-cell grid density values to normalized cell masses."""

    values = np.asarray(grid_values, dtype=float).reshape(-1)
    total = np.sum(values)
    if total <= 0.0:
        raise ValueError("grid values must have positive total mass.")
    return values / total


def circular_error(angle_a: float, angle_b: float) -> float:
    """Return the wrapped absolute angular error in radians."""

    return float(np.abs((angle_a - angle_b + np.pi) % (2.0 * np.pi) - np.pi))


def circular_weighted_mean(angles: np.ndarray, weights: np.ndarray) -> float:
    """Return the circular mean angle for weighted grid values."""

    angles = np.asarray(angles, dtype=float).reshape(-1)
    weights = grid_probability_masses(weights)
    moment = np.sum(weights * np.exp(1j * angles))
    return float(np.mod(np.angle(moment), 2.0 * np.pi))


def _safe_sinc(x: float) -> float:
    if abs(x) < 1e-14:
        return 1.0
    return float(np.sin(x) / x)


def _as_body_increment(body_increment: np.ndarray) -> np.ndarray:
    u = np.asarray(body_increment, dtype=float).reshape(-1)
    if u.shape != (2,):
        raise ValueError("body_increment must be a 2-D vector.")
    return u
