"""Fixed-lag smoother for random-matrix extended object trackers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from pyrecest.backend import asarray
from pyrecest.backend import copy as backend_copy
from pyrecest.backend import eye, linalg, maximum
from pyrecest.filters.factorized_giw_random_matrix_tracker import (
    FactorizedGIWRandomMatrixTracker,
)
from pyrecest.filters.random_matrix_tracker import RandomMatrixTracker

from .abstract_smoother import AbstractSmoother


@dataclass
class RandomMatrixTrackerState:
    """Detached snapshot of a :class:`RandomMatrixTracker` state."""

    kinematic_state: Any
    covariance: Any
    extent: Any
    alpha: float = 0.0
    kinematic_state_to_pos_matrix: Any | None = None

    @classmethod
    def from_tracker(cls, tracker: RandomMatrixTracker) -> "RandomMatrixTrackerState":
        """Create a detached state snapshot from ``tracker``."""

        return cls(
            backend_copy(tracker.kinematic_state),
            backend_copy(tracker.covariance),
            backend_copy(tracker.extent),
            float(tracker.alpha),
            (
                None
                if tracker.kinematic_state_to_pos_matrix is None
                else backend_copy(tracker.kinematic_state_to_pos_matrix)
            ),
        )

    def copy(self) -> "RandomMatrixTrackerState":
        """Return a detached copy of this state."""

        return RandomMatrixTrackerState(
            backend_copy(self.kinematic_state),
            backend_copy(self.covariance),
            backend_copy(self.extent),
            float(self.alpha),
            (
                None
                if self.kinematic_state_to_pos_matrix is None
                else backend_copy(self.kinematic_state_to_pos_matrix)
            ),
        )

    def to_tracker(self) -> RandomMatrixTracker:
        """Convert this snapshot back to a mutable tracker instance."""

        tracker = RandomMatrixTracker(
            backend_copy(self.kinematic_state),
            backend_copy(self.covariance),
            backend_copy(self.extent),
            (
                None
                if self.kinematic_state_to_pos_matrix is None
                else backend_copy(self.kinematic_state_to_pos_matrix)
            ),
        )
        tracker.alpha = float(self.alpha)
        return tracker


@dataclass
class FactorizedGIWRandomMatrixTrackerState:
    """Detached snapshot of a factorized GIW random-matrix tracker state."""

    kinematic_state: Any
    covariance: Any
    extent_dof: float
    extent_scale: Any
    kinematic_state_to_pos_matrix: Any | None = None
    extent_transition_dof: float | None = None
    extent_transition_matrix: Any | None = None
    measurement_spread_factor: float = 1.0
    minimum_extent_eigenvalue: float = 1e-12

    @property
    def extent_dimension(self) -> int:
        return int(asarray(self.extent_scale).shape[0])

    @property
    def extent_mean_denominator(self) -> float:
        return max(
            float(self.extent_dof) - 2.0 * float(self.extent_dimension) - 2.0,
            float(self.minimum_extent_eigenvalue),
        )

    @property
    def extent(self):
        return self.extent_scale / self.extent_mean_denominator

    @classmethod
    def from_tracker(
        cls, tracker: FactorizedGIWRandomMatrixTracker
    ) -> "FactorizedGIWRandomMatrixTrackerState":
        """Create a detached state snapshot from ``tracker``."""

        return cls(
            backend_copy(tracker.kinematic_state),
            backend_copy(tracker.covariance),
            float(tracker.extent_dof),
            backend_copy(tracker.extent_scale),
            (
                None
                if tracker.kinematic_state_to_pos_matrix is None
                else backend_copy(tracker.kinematic_state_to_pos_matrix)
            ),
            float(tracker.extent_transition_dof),
            (
                None
                if tracker.extent_transition_matrix is None
                else backend_copy(tracker.extent_transition_matrix)
            ),
            float(tracker.measurement_spread_factor),
            float(tracker.minimum_extent_eigenvalue),
        )

    def copy(self) -> "FactorizedGIWRandomMatrixTrackerState":
        """Return a detached copy of this state."""

        return FactorizedGIWRandomMatrixTrackerState(
            backend_copy(self.kinematic_state),
            backend_copy(self.covariance),
            float(self.extent_dof),
            backend_copy(self.extent_scale),
            (
                None
                if self.kinematic_state_to_pos_matrix is None
                else backend_copy(self.kinematic_state_to_pos_matrix)
            ),
            (
                None
                if self.extent_transition_dof is None
                else float(self.extent_transition_dof)
            ),
            (
                None
                if self.extent_transition_matrix is None
                else backend_copy(self.extent_transition_matrix)
            ),
            float(self.measurement_spread_factor),
            float(self.minimum_extent_eigenvalue),
        )

    def to_tracker(self) -> FactorizedGIWRandomMatrixTracker:
        """Convert this snapshot back to a mutable tracker instance."""

        return FactorizedGIWRandomMatrixTracker(
            backend_copy(self.kinematic_state),
            backend_copy(self.covariance),
            float(self.extent_dof),
            backend_copy(self.extent_scale),
            (
                None
                if self.kinematic_state_to_pos_matrix is None
                else backend_copy(self.kinematic_state_to_pos_matrix)
            ),
            extent_transition_dof=(
                100.0
                if self.extent_transition_dof is None
                else float(self.extent_transition_dof)
            ),
            extent_transition_matrix=(
                None
                if self.extent_transition_matrix is None
                else backend_copy(self.extent_transition_matrix)
            ),
            measurement_spread_factor=float(self.measurement_spread_factor),
            minimum_extent_eigenvalue=float(self.minimum_extent_eigenvalue),
        )


class FixedLagRandomMatrixSmoother(AbstractSmoother):
    """Fixed-lag smoother for ``RandomMatrixTracker`` posterior sequences.

    The kinematic state is smoothed with a finite-window RTS recursion. The
    default random-matrix extent smoother follows the constant-extent-dynamics
    factorized GIW backward recursion of Granstrom and Bramstang, "Bayesian
    Smoothing for the Extended Object Random Matrix Model", IEEE TSP 2019,
    Table VII. Since :class:`RandomMatrixTrackerState` stores only the extent
    mean and an effective information scalar ``alpha``, the implementation
    reconstructs the inverse-Wishart natural parameter as ``V = alpha * X`` and
    treats ``alpha`` as ``v - 2d - 2``. Set ``extent_smoothing='information'``
    for the earlier SPD-preserving alpha-weighted average, or
    ``extent_smoothing='none'`` to smooth only the kinematic component. Provide
    ``extent_transition_dof`` to include the paper's finite-transition-dof
    correction terms; the default omits those corrections.
    """

    _EXTENT_SMOOTHING_MODES = ("granstrom", "information", "none")

    def __init__(
        self,
        lag: int = 1,
        extent_smoothing: str = "granstrom",
        extent_smoothing_factor: float = 1.0,
        minimum_extent_weight: float = 1e-12,
        extent_transition_dof: float | None = None,
    ):
        lag = int(lag)
        if lag < 0:
            raise ValueError("lag must be a non-negative integer.")
        if extent_smoothing not in self._EXTENT_SMOOTHING_MODES:
            raise ValueError(
                f"extent_smoothing must be one of {', '.join(self._EXTENT_SMOOTHING_MODES)}."
            )
        if extent_smoothing_factor < 0:
            raise ValueError("extent_smoothing_factor must be non-negative.")
        if minimum_extent_weight <= 0:
            raise ValueError("minimum_extent_weight must be positive.")
        if extent_transition_dof is not None and extent_transition_dof <= 0:
            raise ValueError("extent_transition_dof must be positive or None.")

        self.lag = lag
        self.extent_smoothing = extent_smoothing
        self.extent_smoothing_factor = float(extent_smoothing_factor)
        self.minimum_extent_weight = float(minimum_extent_weight)
        self.extent_transition_dof = (
            None if extent_transition_dof is None else float(extent_transition_dof)
        )
        self._filtered_buffer: list[RandomMatrixTrackerState] = []
        self._predicted_buffer: list[RandomMatrixTrackerState] = []
        self._system_matrix_buffer: list[Any] = []

    @staticmethod
    def _as_state(state) -> RandomMatrixTrackerState:
        if isinstance(state, RandomMatrixTrackerState):
            return state.copy()
        if isinstance(state, RandomMatrixTracker):
            return RandomMatrixTrackerState.from_tracker(state)
        if isinstance(state, tuple) and len(state) in (3, 4, 5):
            alpha = 0.0 if len(state) < 4 else float(state[3])
            pos_matrix = None if len(state) < 5 else state[4]
            return RandomMatrixTrackerState(
                asarray(state[0]).reshape(-1),
                asarray(state[1]),
                asarray(state[2]),
                alpha,
                None if pos_matrix is None else asarray(pos_matrix),
            )
        raise ValueError(
            "State must be a RandomMatrixTracker, RandomMatrixTrackerState, or a tuple (kinematic_state, covariance, extent[, alpha[, H_pos]])."
        )

    @classmethod
    def _normalize_state_sequence(
        cls, states: Sequence
    ) -> list[RandomMatrixTrackerState]:
        return [cls._as_state(state) for state in states]

    def _positive_extent_weight(self, alpha) -> float:
        return max(float(alpha), self.minimum_extent_weight)

    @classmethod
    def _project_symmetric_extent(cls, extent, minimum_eigenvalue=1e-12):
        extent = cls._symmetrize(asarray(extent))
        eigenvalues, eigenvectors = linalg.eigh(extent)
        if float(eigenvalues[0]) >= minimum_eigenvalue:
            return extent
        eigenvalues = maximum(eigenvalues, minimum_eigenvalue)
        return cls._symmetrize((eigenvectors * eigenvalues) @ eigenvectors.T)

    def _smooth_extent_information(
        self,
        filtered_state: RandomMatrixTrackerState,
        predicted_state: RandomMatrixTrackerState,
        next_smoothed_state: RandomMatrixTrackerState,
    ) -> tuple[Any, float]:
        if self.extent_smoothing == "none" or self.extent_smoothing_factor == 0.0:
            return backend_copy(filtered_state.extent), float(filtered_state.alpha)

        current_weight = self._positive_extent_weight(filtered_state.alpha)
        future_information = max(
            float(next_smoothed_state.alpha) - float(predicted_state.alpha), 0.0
        )
        future_weight = self.extent_smoothing_factor * future_information
        if future_weight <= 0.0:
            return backend_copy(filtered_state.extent), float(filtered_state.alpha)

        weight_sum = current_weight + future_weight
        smoothed_extent = (
            current_weight * filtered_state.extent
            + future_weight * next_smoothed_state.extent
        ) / weight_sum
        return (
            self._project_symmetric_extent(smoothed_extent, self.minimum_extent_weight),
            weight_sum,
        )

    def _smooth_extent_granstrom(
        self,
        filtered_state: RandomMatrixTrackerState,
        predicted_state: RandomMatrixTrackerState,
        next_smoothed_state: RandomMatrixTrackerState,
    ) -> tuple[Any, float]:
        extent_dimension = int(asarray(filtered_state.extent).shape[0])
        alpha_delta = float(next_smoothed_state.alpha) - float(predicted_state.alpha)
        gain_denominator = 1.0
        dof_correction = 0.0
        if self.extent_transition_dof is not None:
            # Factorized GIW Table VII with A=I, translated from (v, V) to the
            # RandomMatrixTrackerState representation (alpha, X).
            dof_correction = (
                2.0 * float(extent_dimension + 1) ** 2 / self.extent_transition_dof
            )
            gain_denominator = (
                1.0
                + (alpha_delta - 3.0 * float(extent_dimension + 1))
                / self.extent_transition_dof
            )
            gain_denominator = max(gain_denominator, self.minimum_extent_weight)

        granstrom_gain = self.extent_smoothing_factor / gain_denominator
        filtered_weight = self._positive_extent_weight(filtered_state.alpha)
        predicted_weight = self._positive_extent_weight(predicted_state.alpha)
        next_weight = self._positive_extent_weight(next_smoothed_state.alpha)

        filtered_scale = filtered_weight * filtered_state.extent
        predicted_scale = predicted_weight * predicted_state.extent
        next_scale = next_weight * next_smoothed_state.extent

        smoothed_weight = max(
            filtered_weight + granstrom_gain * (alpha_delta - dof_correction),
            self.minimum_extent_weight,
        )
        smoothed_scale = filtered_scale + granstrom_gain * (
            next_scale - predicted_scale
        )
        smoothed_extent = smoothed_scale / smoothed_weight
        return (
            self._project_symmetric_extent(smoothed_extent, self.minimum_extent_weight),
            smoothed_weight,
        )

    def _smooth_extent(
        self,
        filtered_state: RandomMatrixTrackerState,
        predicted_state: RandomMatrixTrackerState,
        next_smoothed_state: RandomMatrixTrackerState,
    ) -> tuple[Any, float]:
        if self.extent_smoothing == "none" or self.extent_smoothing_factor == 0.0:
            return backend_copy(filtered_state.extent), float(filtered_state.alpha)

        if self.extent_smoothing == "information":
            return self._smooth_extent_information(
                filtered_state,
                predicted_state,
                next_smoothed_state,
            )

        return self._smooth_extent_granstrom(
            filtered_state,
            predicted_state,
            next_smoothed_state,
        )

    def _smooth_window(
        self,
        filtered_states: Sequence[RandomMatrixTrackerState],
        predicted_states: Sequence[RandomMatrixTrackerState],
        system_matrices: Sequence,
    ) -> tuple[list[RandomMatrixTrackerState], list[Any]]:
        n_states = len(filtered_states)
        if n_states == 0:
            return [], []
        if len(predicted_states) != n_states - 1:
            raise ValueError(
                "predicted_states must contain one entry fewer than filtered_states."
            )
        if len(system_matrices) != n_states - 1:
            raise ValueError(
                "system_matrices must contain one entry fewer than filtered_states."
            )

        smoothed: list[RandomMatrixTrackerState | None] = [None] * n_states
        gains: list[Any] = [None] * max(n_states - 1, 0)
        smoothed[-1] = filtered_states[-1].copy()

        for time_idx in range(n_states - 2, -1, -1):
            filtered_state = filtered_states[time_idx]
            predicted_state = predicted_states[time_idx]
            system_matrix = system_matrices[time_idx]
            next_smoothed = smoothed[time_idx + 1]
            assert next_smoothed is not None

            smoother_gain = linalg.solve(
                predicted_state.covariance.T,
                (filtered_state.covariance @ system_matrix.T).T,
            ).T
            gains[time_idx] = smoother_gain

            smoothed_kinematic_state = (
                filtered_state.kinematic_state
                + smoother_gain
                @ (next_smoothed.kinematic_state - predicted_state.kinematic_state)
            )
            smoothed_covariance = (
                filtered_state.covariance
                + smoother_gain
                @ (next_smoothed.covariance - predicted_state.covariance)
                @ smoother_gain.T
            )
            smoothed_extent, smoothed_alpha = self._smooth_extent(
                filtered_state,
                predicted_state,
                next_smoothed,
            )
            smoothed[time_idx] = RandomMatrixTrackerState(
                smoothed_kinematic_state,
                self._symmetrize(smoothed_covariance),
                smoothed_extent,
                smoothed_alpha,
                filtered_state.kinematic_state_to_pos_matrix,
            )

        return [state for state in smoothed if state is not None], gains

    def smooth(
        self,
        filtered_states: Sequence,
        predicted_states: Sequence | None = None,
        system_matrices=None,
        lag: int | None = None,
    ) -> tuple[list[RandomMatrixTrackerState], list[list[Any]]]:
        """Return fixed-lag smoothed random-matrix tracker states."""

        lag_value = self.lag if lag is None else int(lag)
        if lag_value < 0:
            raise ValueError("lag must be a non-negative integer.")

        filt_list = self._normalize_state_sequence(filtered_states)
        if len(filt_list) == 0:
            raise ValueError("At least one filtered state is required.")
        if lag_value == 0 or len(filt_list) == 1:
            return [state.copy() for state in filt_list], [[] for _ in filt_list]

        if predicted_states is None:
            raise ValueError(
                "predicted_states must be provided for non-zero lag smoothing."
            )
        pred_list = self._normalize_state_sequence(predicted_states)
        if len(pred_list) != len(filt_list) - 1:
            raise ValueError(
                "predicted_states must contain one entry fewer than filtered_states."
            )

        state_dim = filt_list[0].kinematic_state.shape[0]
        sys_matrices_list = self._normalize_matrix_sequence(
            system_matrices,
            len(filt_list) - 1,
            "system_matrices",
            state_dim,
            default=eye(state_dim),
        )

        smoothed_states: list[RandomMatrixTrackerState] = []
        smoother_gains: list[list[Any]] = []
        for time_idx in range(len(filt_list)):
            window_end = min(time_idx + lag_value, len(filt_list) - 1)
            if window_end == time_idx:
                smoothed_states.append(filt_list[time_idx].copy())
                smoother_gains.append([])
                continue

            window_smoothed, window_gains = self._smooth_window(
                filt_list[time_idx : window_end + 1],
                pred_list[time_idx:window_end],
                sys_matrices_list[time_idx:window_end],
            )
            smoothed_states.append(window_smoothed[0])
            smoother_gains.append(window_gains)

        return smoothed_states, smoother_gains

    def append(
        self,
        filtered_state,
        predicted_state=None,
        system_matrix=None,
    ) -> RandomMatrixTrackerState | None:
        """Append a filtered state and emit the oldest fixed-lag state if ready."""

        new_filtered_state = self._as_state(filtered_state)
        if self.lag == 0:
            return new_filtered_state

        if self._filtered_buffer:
            if predicted_state is None:
                raise ValueError(
                    "predicted_state is required for the second and later filtered states."
                )
            self._predicted_buffer.append(self._as_state(predicted_state))
            state_dim = self._filtered_buffer[-1].kinematic_state.shape[0]
            self._system_matrix_buffer.append(
                eye(state_dim) if system_matrix is None else asarray(system_matrix)
            )
        elif predicted_state is not None:
            raise ValueError(
                "predicted_state must not be provided for the first filtered state."
            )

        self._filtered_buffer.append(new_filtered_state)
        if len(self._filtered_buffer) <= self.lag:
            return None
        return self._emit_oldest()

    def _emit_oldest(self) -> RandomMatrixTrackerState:
        smoothed_states, _ = self._smooth_window(
            self._filtered_buffer,
            self._predicted_buffer,
            self._system_matrix_buffer,
        )
        emitted = smoothed_states[0]
        self._filtered_buffer.pop(0)
        if self._predicted_buffer:
            self._predicted_buffer.pop(0)
        if self._system_matrix_buffer:
            self._system_matrix_buffer.pop(0)
        return emitted

    def flush(self) -> list[RandomMatrixTrackerState]:
        """Return all still-buffered states with truncated look-ahead windows."""

        if self.lag == 0:
            return []

        remaining: list[RandomMatrixTrackerState] = []
        while self._filtered_buffer:
            if len(self._filtered_buffer) == 1:
                remaining.append(self._filtered_buffer.pop(0).copy())
                self._predicted_buffer.clear()
                self._system_matrix_buffer.clear()
            else:
                remaining.append(self._emit_oldest())
        return remaining


class FixedLagFactorizedGIWRandomMatrixSmoother(AbstractSmoother):
    """Fixed-lag smoother for factorized GIW random-matrix tracker states.

    This class applies the factorized Gaussian inverse-Wishart backward
    recursion of Granstrom and Bramstang, IEEE TSP 2019, Table VII. Unlike
    :class:`FixedLagRandomMatrixSmoother`, it keeps the inverse-Wishart
    parameters ``v`` and ``V`` explicit and therefore implements the paper's
    finite-transition-dof correction terms directly.
    """

    _EXTENT_SMOOTHING_MODES = ("granstrom", "none")

    def __init__(
        self,
        lag: int = 1,
        extent_smoothing: str = "granstrom",
        extent_transition_dof: float = 100.0,
        minimum_extent_weight: float = 1e-12,
        minimum_extent_eigenvalue: float = 1e-12,
    ):
        lag = int(lag)
        if lag < 0:
            raise ValueError("lag must be a non-negative integer.")
        if extent_smoothing not in self._EXTENT_SMOOTHING_MODES:
            raise ValueError(
                f"extent_smoothing must be one of {', '.join(self._EXTENT_SMOOTHING_MODES)}."
            )
        if extent_transition_dof <= 0:
            raise ValueError("extent_transition_dof must be positive.")
        if minimum_extent_weight <= 0:
            raise ValueError("minimum_extent_weight must be positive.")
        if minimum_extent_eigenvalue <= 0:
            raise ValueError("minimum_extent_eigenvalue must be positive.")

        self.lag = lag
        self.extent_smoothing = extent_smoothing
        self.extent_transition_dof = float(extent_transition_dof)
        self.minimum_extent_weight = float(minimum_extent_weight)
        self.minimum_extent_eigenvalue = float(minimum_extent_eigenvalue)
        self._filtered_buffer: list[FactorizedGIWRandomMatrixTrackerState] = []
        self._predicted_buffer: list[FactorizedGIWRandomMatrixTrackerState] = []
        self._system_matrix_buffer: list[Any] = []
        self._extent_transition_matrix_buffer: list[Any] = []

    @staticmethod
    def _as_state(state) -> FactorizedGIWRandomMatrixTrackerState:
        if isinstance(state, FactorizedGIWRandomMatrixTrackerState):
            return state.copy()
        if isinstance(state, FactorizedGIWRandomMatrixTracker):
            return FactorizedGIWRandomMatrixTrackerState.from_tracker(state)
        if isinstance(state, tuple) and len(state) in (4, 5, 6):
            pos_matrix = None if len(state) < 5 else state[4]
            transition_matrix = None if len(state) < 6 else state[5]
            return FactorizedGIWRandomMatrixTrackerState(
                asarray(state[0]).reshape(-1),
                asarray(state[1]),
                float(state[2]),
                asarray(state[3]),
                None if pos_matrix is None else asarray(pos_matrix),
                extent_transition_matrix=(
                    None if transition_matrix is None else asarray(transition_matrix)
                ),
            )
        raise ValueError(
            "State must be a FactorizedGIWRandomMatrixTracker, "
            "FactorizedGIWRandomMatrixTrackerState, or a tuple "
            "(kinematic_state, covariance, extent_dof, extent_scale"
            "[, H_pos[, extent_transition_matrix]])."
        )

    @classmethod
    def _normalize_state_sequence(
        cls, states: Sequence
    ) -> list[FactorizedGIWRandomMatrixTrackerState]:
        return [cls._as_state(state) for state in states]

    @classmethod
    def _project_symmetric_positive(cls, matrix, minimum_eigenvalue=1e-12):
        matrix = cls._symmetrize(asarray(matrix))
        eigenvalues, eigenvectors = linalg.eigh(matrix)
        if float(eigenvalues[0]) >= minimum_eigenvalue:
            return matrix
        eigenvalues = maximum(eigenvalues, minimum_eigenvalue)
        return cls._symmetrize((eigenvectors * eigenvalues) @ eigenvectors.T)

    def _extent_transition_dof_for_state(
        self, state: FactorizedGIWRandomMatrixTrackerState
    ) -> float:
        transition_dof = (
            float(state.extent_transition_dof)
            if state.extent_transition_dof is not None
            else self.extent_transition_dof
        )
        if transition_dof <= state.extent_dimension + 1.0:
            raise ValueError(
                "extent_transition_dof must be larger than extent_dimension + 1."
            )
        return transition_dof

    def _smooth_extent_granstrom(
        self,
        filtered_state: FactorizedGIWRandomMatrixTrackerState,
        predicted_state: FactorizedGIWRandomMatrixTrackerState,
        next_smoothed_state: FactorizedGIWRandomMatrixTrackerState,
        extent_transition_matrix,
    ) -> tuple[float, Any]:
        extent_dimension = filtered_state.extent_dimension
        transition_dof = self._extent_transition_dof_for_state(filtered_state)
        if self.extent_smoothing == "none":
            return float(filtered_state.extent_dof), backend_copy(
                filtered_state.extent_scale
            )

        dof_delta = float(next_smoothed_state.extent_dof) - float(
            predicted_state.extent_dof
        )
        eta = 1.0 + (dof_delta - 3.0 * float(extent_dimension + 1)) / transition_dof
        eta = max(eta, self.minimum_extent_weight)
        dof_correction = 2.0 * float(extent_dimension + 1) ** 2 / transition_dof

        smoothed_dof = (
            float(filtered_state.extent_dof) + (dof_delta - dof_correction) / eta
        )
        min_dof = 2.0 * float(extent_dimension) + 2.0 + self.minimum_extent_weight
        smoothed_dof = max(smoothed_dof, min_dof)

        A_inv = linalg.inv(asarray(extent_transition_matrix))
        smoothed_scale = (
            filtered_state.extent_scale
            + (
                A_inv
                @ (next_smoothed_state.extent_scale - predicted_state.extent_scale)
                @ A_inv.T
            )
            / eta
        )
        smoothed_scale = self._project_symmetric_positive(
            smoothed_scale, self.minimum_extent_eigenvalue
        )
        return smoothed_dof, smoothed_scale

    def _smooth_window(
        self,
        filtered_states: Sequence[FactorizedGIWRandomMatrixTrackerState],
        predicted_states: Sequence[FactorizedGIWRandomMatrixTrackerState],
        system_matrices: Sequence,
        extent_transition_matrices: Sequence,
    ) -> tuple[list[FactorizedGIWRandomMatrixTrackerState], list[Any]]:
        n_states = len(filtered_states)
        if n_states == 0:
            return [], []
        if len(predicted_states) != n_states - 1:
            raise ValueError(
                "predicted_states must contain one entry fewer than filtered_states."
            )
        if len(system_matrices) != n_states - 1:
            raise ValueError(
                "system_matrices must contain one entry fewer than filtered_states."
            )
        if len(extent_transition_matrices) != n_states - 1:
            raise ValueError(
                "extent_transition_matrices must contain one entry fewer than filtered_states."
            )

        smoothed: list[FactorizedGIWRandomMatrixTrackerState | None] = [None] * n_states
        gains: list[Any] = [None] * max(n_states - 1, 0)
        smoothed[-1] = filtered_states[-1].copy()

        for time_idx in range(n_states - 2, -1, -1):
            filtered_state = filtered_states[time_idx]
            predicted_state = predicted_states[time_idx]
            system_matrix = system_matrices[time_idx]
            next_smoothed = smoothed[time_idx + 1]
            assert next_smoothed is not None

            smoother_gain = linalg.solve(
                predicted_state.covariance.T,
                (filtered_state.covariance @ system_matrix.T).T,
            ).T
            gains[time_idx] = smoother_gain

            smoothed_kinematic_state = (
                filtered_state.kinematic_state
                + smoother_gain
                @ (next_smoothed.kinematic_state - predicted_state.kinematic_state)
            )
            smoothed_covariance = (
                filtered_state.covariance
                + smoother_gain
                @ (next_smoothed.covariance - predicted_state.covariance)
                @ smoother_gain.T
            )
            smoothed_dof, smoothed_scale = self._smooth_extent_granstrom(
                filtered_state,
                predicted_state,
                next_smoothed,
                extent_transition_matrices[time_idx],
            )
            smoothed[time_idx] = FactorizedGIWRandomMatrixTrackerState(
                smoothed_kinematic_state,
                self._symmetrize(smoothed_covariance),
                smoothed_dof,
                smoothed_scale,
                filtered_state.kinematic_state_to_pos_matrix,
                filtered_state.extent_transition_dof,
                filtered_state.extent_transition_matrix,
                filtered_state.measurement_spread_factor,
                filtered_state.minimum_extent_eigenvalue,
            )

        return [state for state in smoothed if state is not None], gains

    def smooth(
        self,
        filtered_states: Sequence,
        predicted_states: Sequence | None = None,
        system_matrices=None,
        extent_transition_matrices=None,
        lag: int | None = None,
    ) -> tuple[list[FactorizedGIWRandomMatrixTrackerState], list[list[Any]]]:
        """Return fixed-lag smoothed factorized GIW tracker states."""

        lag_value = self.lag if lag is None else int(lag)
        if lag_value < 0:
            raise ValueError("lag must be a non-negative integer.")

        filt_list = self._normalize_state_sequence(filtered_states)
        if len(filt_list) == 0:
            raise ValueError("At least one filtered state is required.")
        if lag_value == 0 or len(filt_list) == 1:
            return [state.copy() for state in filt_list], [[] for _ in filt_list]

        if predicted_states is None:
            raise ValueError(
                "predicted_states must be provided for non-zero lag smoothing."
            )
        pred_list = self._normalize_state_sequence(predicted_states)
        if len(pred_list) != len(filt_list) - 1:
            raise ValueError(
                "predicted_states must contain one entry fewer than filtered_states."
            )

        state_dim = filt_list[0].kinematic_state.shape[0]
        extent_dim = filt_list[0].extent_dimension
        sys_matrices_list = self._normalize_matrix_sequence(
            system_matrices,
            len(filt_list) - 1,
            "system_matrices",
            state_dim,
            default=eye(state_dim),
        )
        extent_transition_matrix_default = (
            eye(extent_dim)
            if filt_list[0].extent_transition_matrix is None
            else filt_list[0].extent_transition_matrix
        )
        extent_transition_matrices_list = self._normalize_matrix_sequence(
            extent_transition_matrices,
            len(filt_list) - 1,
            "extent_transition_matrices",
            extent_dim,
            default=extent_transition_matrix_default,
        )

        smoothed_states: list[FactorizedGIWRandomMatrixTrackerState] = []
        smoother_gains: list[list[Any]] = []
        for time_idx in range(len(filt_list)):
            window_end = min(time_idx + lag_value, len(filt_list) - 1)
            if window_end == time_idx:
                smoothed_states.append(filt_list[time_idx].copy())
                smoother_gains.append([])
                continue

            window_smoothed, window_gains = self._smooth_window(
                filt_list[time_idx : window_end + 1],
                pred_list[time_idx:window_end],
                sys_matrices_list[time_idx:window_end],
                extent_transition_matrices_list[time_idx:window_end],
            )
            smoothed_states.append(window_smoothed[0])
            smoother_gains.append(window_gains)

        return smoothed_states, smoother_gains

    def append(
        self,
        filtered_state,
        predicted_state=None,
        system_matrix=None,
        extent_transition_matrix=None,
    ) -> FactorizedGIWRandomMatrixTrackerState | None:
        """Append a filtered state and emit the oldest fixed-lag state if ready."""

        new_filtered_state = self._as_state(filtered_state)
        if self.lag == 0:
            return new_filtered_state

        if self._filtered_buffer:
            if predicted_state is None:
                raise ValueError(
                    "predicted_state is required for the second and later filtered states."
                )
            self._predicted_buffer.append(self._as_state(predicted_state))
            state_dim = self._filtered_buffer[-1].kinematic_state.shape[0]
            extent_dim = self._filtered_buffer[-1].extent_dimension
            self._system_matrix_buffer.append(
                eye(state_dim) if system_matrix is None else asarray(system_matrix)
            )
            default_extent_matrix = (
                eye(extent_dim)
                if new_filtered_state.extent_transition_matrix is None
                else new_filtered_state.extent_transition_matrix
            )
            self._extent_transition_matrix_buffer.append(
                default_extent_matrix
                if extent_transition_matrix is None
                else asarray(extent_transition_matrix)
            )
        elif predicted_state is not None:
            raise ValueError(
                "predicted_state must not be provided for the first filtered state."
            )

        self._filtered_buffer.append(new_filtered_state)
        if len(self._filtered_buffer) <= self.lag:
            return None
        return self._emit_oldest()

    def _emit_oldest(self) -> FactorizedGIWRandomMatrixTrackerState:
        smoothed_states, _ = self._smooth_window(
            self._filtered_buffer,
            self._predicted_buffer,
            self._system_matrix_buffer,
            self._extent_transition_matrix_buffer,
        )
        emitted = smoothed_states[0]
        self._filtered_buffer.pop(0)
        if self._predicted_buffer:
            self._predicted_buffer.pop(0)
        if self._system_matrix_buffer:
            self._system_matrix_buffer.pop(0)
        if self._extent_transition_matrix_buffer:
            self._extent_transition_matrix_buffer.pop(0)
        return emitted

    def flush(self) -> list[FactorizedGIWRandomMatrixTrackerState]:
        """Return all still-buffered states with truncated look-ahead windows."""

        if self.lag == 0:
            return []

        remaining: list[FactorizedGIWRandomMatrixTrackerState] = []
        while self._filtered_buffer:
            if len(self._filtered_buffer) == 1:
                remaining.append(self._filtered_buffer.pop(0).copy())
                self._predicted_buffer.clear()
                self._system_matrix_buffer.clear()
                self._extent_transition_matrix_buffer.clear()
            else:
                remaining.append(self._emit_oldest())
        return remaining


FixedLagRMTSmoother = FixedLagRandomMatrixSmoother
FLRMSmoother = FixedLagRandomMatrixSmoother
FixedLagFactorizedGIWRMSmoother = FixedLagFactorizedGIWRandomMatrixSmoother
FLGIWRMSmoother = FixedLagFactorizedGIWRandomMatrixSmoother
