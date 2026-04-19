"""Rauch--Tung--Striebel smoother for linear Gaussian state-space models."""

from __future__ import annotations

from copy import copy

from typing import Sequence

from pyrecest.backend import asarray, eye, linalg, ndim, zeros
from pyrecest.distributions import GaussianDistribution

from .abstract_smoother import AbstractSmoother


class RauchTungStriebelSmoother(AbstractSmoother):
    """Rauch--Tung--Striebel smoother for linear Gaussian models.

    This class intentionally does not depend on ``pyrecest.filters.KalmanFilter``.
    The current filter classes do not retain the full forward-pass history that an
    RTS smoother requires, so this implementation provides its own forward pass for
    sequences of linear Gaussian models.

    The accepted model inputs can each either be a single matrix/vector reused for
    the whole sequence or a sequence with one entry per time step.

    Parameters
    ----------
    initial_state
        Initial prior state as ``GaussianDistribution`` or ``(mean, covariance)``.
    measurements
        Sequence of measurements. A one-dimensional array is interpreted as a
        sequence of scalar measurements. For vector measurements, pass a list/tuple
        of one-dimensional arrays or a two-dimensional array of shape ``(T, dim_z)``.
    measurement_matrices
        Measurement matrix ``H`` or sequence ``H_t``. Defaults to identity.
    meas_noise_covariances
        Measurement noise covariance ``R`` or sequence ``R_t``.
    system_matrices
        Transition matrix ``F`` or sequence ``F_t`` used between consecutive time
        steps. Defaults to identity.
    sys_noise_covariances
        Process noise covariance ``Q`` or sequence ``Q_t``. Defaults to zero.
    sys_inputs
        Optional control/input vector ``u`` or sequence ``u_t`` added in the
        prediction step. Defaults to zero.
    """

    @staticmethod
    def _as_gaussian(state: "GaussianDistribution | tuple") -> GaussianDistribution:
        if isinstance(state, GaussianDistribution):
            return state
        if isinstance(state, tuple) and len(state) == 2:
            return GaussianDistribution(state[0], state[1], check_validity=False)
        raise ValueError(
            "State must be a GaussianDistribution or a tuple of (mean, covariance)."
        )

    @staticmethod
    def _symmetrize(matrix):
        return 0.5 * (matrix + matrix.T)

    @staticmethod
    def _normalize_measurements(measurements) -> list:
        if isinstance(measurements, (list, tuple)):
            return [asarray(measurement).reshape(-1) for measurement in measurements]

        measurements_array = asarray(measurements)
        if ndim(measurements_array) == 0:
            return [measurements_array.reshape((1,))]
        if ndim(measurements_array) == 1:
            return [asarray([measurement]) for measurement in measurements_array]
        if ndim(measurements_array) == 2:
            return [measurements_array[idx] for idx in range(measurements_array.shape[0])]
        raise ValueError("Measurements must be a 1-D or 2-D array, or a Python sequence.")

    @staticmethod
    def _normalize_matrix_sequence(values, length: int, name: str, matrix_dim: int, default=None) -> list:
        if length == 0:
            return []

        if values is None:
            if default is None:
                raise ValueError(f"{name} must be provided.")
            default_arr = asarray(default)
            return [copy(default_arr) for _ in range(length)]

        values_arr = asarray(values)
        if ndim(values_arr) == 0:
            if matrix_dim != 1:
                raise ValueError(
                    f"Scalar input for {name} is only supported in one-dimensional models."
                )
            scalar_matrix = asarray([[values_arr]])
            return [copy(scalar_matrix) for _ in range(length)]
        if ndim(values_arr) == 1 and matrix_dim == 1 and values_arr.shape[0] == length:
            return [asarray([[values_arr[idx]]]) for idx in range(length)]
        if ndim(values_arr) == 2:
            return [copy(values_arr) for _ in range(length)]
        if ndim(values_arr) == 3 and values_arr.shape[0] == length:
            return [copy(values_arr[idx]) for idx in range(length)]

        if isinstance(values, (list, tuple)) and len(values) == length:
            normalized_values = []
            for value in values:
                value_arr = asarray(value)
                if ndim(value_arr) == 0:
                    if matrix_dim != 1:
                        raise ValueError(
                            f"Scalar entries in {name} are only supported in one-dimensional models."
                        )
                    normalized_values.append(asarray([[value_arr]]))
                else:
                    normalized_values.append(value_arr)
            return normalized_values

        raise ValueError(
            f"{name} must be a single matrix or a sequence with length {length}."
        )

    @staticmethod
    def _normalize_vector_sequence(values, length: int, name: str, vector_dim: int) -> list:
        if length == 0:
            return []

        if values is None:
            return [None] * length

        values_arr = asarray(values)
        if ndim(values_arr) == 0:
            if vector_dim != 1:
                raise ValueError(
                    f"Scalar input for {name} is only supported in one-dimensional models."
                )
            scalar_vector = asarray([values_arr])
            return [copy(scalar_vector) for _ in range(length)]
        if ndim(values_arr) == 1:
            if vector_dim == 1 and values_arr.shape[0] == length:
                return [asarray([values_arr[idx]]) for idx in range(length)]
            return [copy(values_arr) for _ in range(length)]
        if ndim(values_arr) == 2 and values_arr.shape[0] == length:
            return [copy(values_arr[idx]) for idx in range(length)]

        if isinstance(values, (list, tuple)) and len(values) == length:
            normalized_values = []
            for value in values:
                value_arr = asarray(value)
                if ndim(value_arr) == 0:
                    if vector_dim != 1:
                        raise ValueError(
                            f"Scalar entries in {name} are only supported in one-dimensional models."
                        )
                    normalized_values.append(asarray([value_arr]))
                else:
                    normalized_values.append(value_arr)
            return normalized_values

        raise ValueError(
            f"{name} must be a single vector or a sequence with length {length}."
        )

    def filter(
        self,
        initial_state: "GaussianDistribution | tuple",
        measurements,
        measurement_matrices=None,
        meas_noise_covariances=None,
        system_matrices=None,
        sys_noise_covariances=None,
        sys_inputs=None,
    ) -> tuple[list[GaussianDistribution], list[GaussianDistribution]]:
        """Run the forward Kalman filtering pass for a sequence.

        Returns
        -------
        filtered_states
            List of posterior states ``x_{t|t}``.
        predicted_states
            List of one-step predictions ``x_{t+1|t}``. Its length is one less than
            the number of measurements.
        """

        initial_state = self._as_gaussian(initial_state)
        measurement_list = self._normalize_measurements(measurements)

        if len(measurement_list) == 0:
            raise ValueError("At least one measurement is required.")

        state_dim = initial_state.dim
        identity_matrix = eye(state_dim)

        measurement_matrices_list = self._normalize_matrix_sequence(
            measurement_matrices,
            len(measurement_list),
            "measurement_matrices",
            state_dim,
            default=identity_matrix,
        )
        meas_noise_covariances_list = self._normalize_matrix_sequence(
            meas_noise_covariances,
            len(measurement_list),
            "meas_noise_covariances",
            state_dim,
        )
        system_matrices_list = self._normalize_matrix_sequence(
            system_matrices,
            max(len(measurement_list) - 1, 0),
            "system_matrices",
            state_dim,
            default=identity_matrix,
        )
        sys_noise_covariances_list = self._normalize_matrix_sequence(
            sys_noise_covariances,
            max(len(measurement_list) - 1, 0),
            "sys_noise_covariances",
            state_dim,
            default=zeros((state_dim, state_dim)),
        )
        sys_inputs_list = self._normalize_vector_sequence(
            sys_inputs,
            max(len(measurement_list) - 1, 0),
            "sys_inputs",
            state_dim,
        )

        filtered_states: list[GaussianDistribution] = []
        predicted_states: list[GaussianDistribution] = []

        predicted_mean = initial_state.mu
        predicted_covariance = initial_state.C

        for time_idx, measurement in enumerate(measurement_list):
            measurement_matrix = measurement_matrices_list[time_idx]
            meas_noise_covariance = meas_noise_covariances_list[time_idx]

            innovation = measurement - measurement_matrix @ predicted_mean
            innovation_covariance = (
                measurement_matrix
                @ predicted_covariance
                @ measurement_matrix.T
                + meas_noise_covariance
            )
            kalman_gain = linalg.solve(
                innovation_covariance.T,
                (predicted_covariance @ measurement_matrix.T).T,
            ).T

            filtered_mean = predicted_mean + kalman_gain @ innovation
            measurement_update = identity_matrix - kalman_gain @ measurement_matrix
            filtered_covariance = (
                measurement_update @ predicted_covariance @ measurement_update.T
                + kalman_gain @ meas_noise_covariance @ kalman_gain.T
            )
            filtered_covariance = self._symmetrize(filtered_covariance)

            filtered_states.append(
                GaussianDistribution(
                    filtered_mean,
                    filtered_covariance,
                    check_validity=False,
                )
            )

            if time_idx == len(measurement_list) - 1:
                continue

            system_matrix = system_matrices_list[time_idx]
            sys_noise_covariance = sys_noise_covariances_list[time_idx]
            sys_input = (
                zeros(state_dim)
                if sys_inputs_list[time_idx] is None
                else sys_inputs_list[time_idx]
            )

            predicted_mean = system_matrix @ filtered_mean + sys_input
            predicted_covariance = (
                system_matrix @ filtered_covariance @ system_matrix.T
                + sys_noise_covariance
            )
            predicted_covariance = self._symmetrize(predicted_covariance)

            predicted_states.append(
                GaussianDistribution(
                    predicted_mean,
                    predicted_covariance,
                    check_validity=False,
                )
            )

        return filtered_states, predicted_states

    def smooth(
        self,
        filtered_states: Sequence["GaussianDistribution | tuple"],
        predicted_states: Sequence["GaussianDistribution | tuple"],
        system_matrices=None,
    ) -> tuple[list[GaussianDistribution], list]:
        """Run the RTS backward pass.

        Parameters
        ----------
        filtered_states
            Sequence of posterior states ``x_{t|t}``.
        predicted_states
            Sequence of one-step predictions ``x_{t+1|t}``.
        system_matrices
            Transition matrix ``F`` or sequence ``F_t`` used in the forward pass.

        Returns
        -------
        smoothed_states
            List of smoothed posterior states ``x_{t|T}``.
        smoother_gains
            List of RTS smoother gains, one per backward recursion step.
        """

        filtered_state_list = [self._as_gaussian(state) for state in filtered_states]
        predicted_state_list = [self._as_gaussian(state) for state in predicted_states]

        if len(filtered_state_list) == 0:
            raise ValueError("At least one filtered state is required.")
        if len(predicted_state_list) != max(len(filtered_state_list) - 1, 0):
            raise ValueError(
                "predicted_states must contain one entry fewer than filtered_states."
            )

        state_dim = filtered_state_list[0].dim
        system_matrices_list = self._normalize_matrix_sequence(
            system_matrices,
            max(len(filtered_state_list) - 1, 0),
            "system_matrices",
            state_dim,
            default=eye(state_dim),
        )

        smoothed_states: list[GaussianDistribution | None] = [None] * len(
            filtered_state_list
        )
        smoother_gains: list = [None] * max(len(filtered_state_list) - 1, 0)

        last_state = filtered_state_list[-1]
        smoothed_states[-1] = GaussianDistribution(
            last_state.mu,
            self._symmetrize(last_state.C),
            check_validity=False,
        )

        for time_idx in range(len(filtered_state_list) - 2, -1, -1):
            filtered_state = filtered_state_list[time_idx]
            predicted_next_state = predicted_state_list[time_idx]
            system_matrix = system_matrices_list[time_idx]

            smoother_gain = linalg.solve(
                predicted_next_state.C.T,
                (filtered_state.C @ system_matrix.T).T,
            ).T
            smoother_gains[time_idx] = smoother_gain

            next_smoothed_state = smoothed_states[time_idx + 1]
            assert next_smoothed_state is not None

            smoothed_mean = filtered_state.mu + smoother_gain @ (
                next_smoothed_state.mu - predicted_next_state.mu
            )
            smoothed_covariance = filtered_state.C + smoother_gain @ (
                next_smoothed_state.C - predicted_next_state.C
            ) @ smoother_gain.T
            smoothed_covariance = self._symmetrize(smoothed_covariance)

            smoothed_states[time_idx] = GaussianDistribution(
                smoothed_mean,
                smoothed_covariance,
                check_validity=False,
            )

        return [state for state in smoothed_states if state is not None], smoother_gains

    def filter_and_smooth(
        self,
        initial_state: "GaussianDistribution | tuple",
        measurements,
        measurement_matrices=None,
        meas_noise_covariances=None,
        system_matrices=None,
        sys_noise_covariances=None,
        sys_inputs=None,
    ) -> tuple[
        list[GaussianDistribution],
        list[GaussianDistribution],
        list[GaussianDistribution],
        list,
    ]:
        """Run a full forward-backward pass for a linear Gaussian sequence."""

        filtered_states, predicted_states = self.filter(
            initial_state=initial_state,
            measurements=measurements,
            measurement_matrices=measurement_matrices,
            meas_noise_covariances=meas_noise_covariances,
            system_matrices=system_matrices,
            sys_noise_covariances=sys_noise_covariances,
            sys_inputs=sys_inputs,
        )
        smoothed_states, smoother_gains = self.smooth(
            filtered_states=filtered_states,
            predicted_states=predicted_states,
            system_matrices=system_matrices,
        )
        return filtered_states, predicted_states, smoothed_states, smoother_gains


RTSSmoother = RauchTungStriebelSmoother
