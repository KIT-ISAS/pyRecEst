"""Distributed linear-Gaussian Kalman filtering helpers."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

from pyrecest.backend import eye, linalg, stack, transpose
from pyrecest.distributions import GaussianDistribution

from ._linear_gaussian import _as_matrix, _as_vector
from .kalman_filter import KalmanFilter, _get_required_model_attribute


@dataclass(frozen=True)
class LinearGaussianInformationContribution:
    """Information-form contribution of one linear Gaussian measurement.

    The contribution represents ``H.T @ inv(R) @ H`` and
    ``H.T @ inv(R) @ z`` for ``z = H x + v`` with ``v ~ N(0, R)``.
    Independent local contributions can be summed and applied to any node prior.
    """

    information_matrix: object
    information_vector: object


def _symmetrize(matrix):
    return 0.5 * (matrix + transpose(matrix))


def _is_single_state(state):
    return isinstance(state, GaussianDistribution) or (
        isinstance(state, tuple) and len(state) == 2
    )


def _as_sequence(value):
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _broadcast_to_length(value, length, name):
    values = _as_sequence(value)
    if len(values) == 1:
        return values * length
    if len(values) != length:
        raise ValueError(f"{name} must contain one item or {length} items")
    return values


def _as_information_contribution(contribution):
    if isinstance(contribution, LinearGaussianInformationContribution):
        information_matrix = contribution.information_matrix
        information_vector = contribution.information_vector
    elif isinstance(contribution, (list, tuple)) and len(contribution) == 2:
        information_matrix, information_vector = contribution
    else:
        raise ValueError(
            "contribution must be a LinearGaussianInformationContribution "
            "or a tuple of (information_matrix, information_vector)"
        )

    information_matrix = _as_matrix(information_matrix, "information_matrix")
    information_vector = _as_vector(information_vector, "information_vector")
    state_dim = information_vector.shape[0]
    if information_matrix.shape != (state_dim, state_dim):
        raise ValueError("information_matrix must have shape (state_dim, state_dim)")
    return LinearGaussianInformationContribution(
        _symmetrize(information_matrix),
        information_vector,
    )


class DistributedKalmanFilter:
    """Exact distributed Kalman fusion for nodes estimating the same state.

    The class keeps one local :class:`KalmanFilter` per node. Each node stores a
    Gaussian belief over the same Euclidean state vector. Local linear-Gaussian
    measurements are converted to additive information-form contributions,
    summed, and applied to the selected node priors.

    This class intentionally does not model communication topology, consensus
    iterations, packet loss, or common-information bookkeeping. Callers provide
    the local measurements or already aggregated information contributions that
    are available at a filtering step.
    """

    def __init__(self, initial_states, num_nodes=None):
        """Create a bank of local Kalman filters.

        ``initial_states`` may be one Gaussian state, one ``(mean, covariance)``
        tuple, or a sequence with one state per node. When ``num_nodes`` is
        provided together with one state, that state is replicated.
        """
        if num_nodes is not None:
            num_nodes = int(num_nodes)
            if num_nodes < 1:
                raise ValueError("num_nodes must be at least 1")
            if _is_single_state(initial_states):
                states = [initial_states] * num_nodes
            else:
                states = list(initial_states)
                if len(states) != num_nodes:
                    raise ValueError("initial_states must contain num_nodes entries")
        elif _is_single_state(initial_states):
            states = [initial_states]
        else:
            states = list(initial_states)

        if not states:
            raise ValueError("at least one node state is required")

        self._node_filters = [KalmanFilter(state) for state in states]
        first_dim = self._node_filters[0].dim
        for node_filter in self._node_filters[1:]:
            if node_filter.dim != first_dim:
                raise ValueError("all node states must have the same dimension")

    @property
    def num_nodes(self):
        """Return the number of managed nodes."""
        return len(self._node_filters)

    @property
    def dim(self):
        """Return the common Euclidean state dimension."""
        return self._node_filters[0].dim

    @property
    def node_filters(self):
        """Return the managed local Kalman filters as a tuple."""
        return tuple(self._node_filters)

    @property
    def node_states(self):
        """Return Gaussian state copies for all nodes."""
        return tuple(node_filter.filter_state for node_filter in self._node_filters)

    @property
    def means(self):
        """Return local posterior means with shape ``(num_nodes, dim)``."""
        return stack([state.mu for state in self.node_states])

    @property
    def covariances(self):
        """Return local posterior covariances with shape ``(num_nodes, dim, dim)``."""
        return stack([state.C for state in self.node_states])

    def get_point_estimates(self):
        """Return local point estimates with shape ``(num_nodes, dim)``."""
        return stack([node_filter.get_point_estimate() for node_filter in self._node_filters])

    def get_point_estimate(self, node_index=0):
        """Return the point estimate of one node."""
        return self._node_filters[node_index].get_point_estimate()

    def _selected_indices(self, node_indices):
        if node_indices is None:
            return range(self.num_nodes)
        if isinstance(node_indices, Integral):
            node_indices = [int(node_indices)]
        indices = list(node_indices)
        for index in indices:
            if index < 0 or index >= self.num_nodes:
                raise IndexError("node index out of range")
        return indices

    def _iter_node_filters(self, node_indices=None):
        for index in self._selected_indices(node_indices):
            yield self._node_filters[index]

    def predict_identity(self, sys_noise_cov, sys_input=None, node_indices=None):
        """Predict selected nodes with an identity transition model."""
        for node_filter in self._iter_node_filters(node_indices):
            node_filter.predict_identity(sys_noise_cov, sys_input)

    def predict_linear(self, system_matrix, sys_noise_cov, sys_input=None, node_indices=None):
        """Predict selected nodes with a shared linear Gaussian model."""
        for node_filter in self._iter_node_filters(node_indices):
            node_filter.predict_linear(system_matrix, sys_noise_cov, sys_input)

    def predict_model(self, transition_model, node_indices=None):
        """Predict selected nodes with a shared structural transition model."""
        for node_filter in self._iter_node_filters(node_indices):
            node_filter.predict_model(transition_model)

    @staticmethod
    def local_information_contribution(measurement, measurement_matrix, meas_noise):
        """Return the information contribution of one local measurement."""
        measurement = _as_vector(measurement, "measurement")
        measurement_matrix = _as_matrix(measurement_matrix, "measurement_matrix")
        meas_noise = _as_matrix(meas_noise, "meas_noise")

        meas_dim = measurement_matrix.shape[0]
        if measurement.shape[0] != meas_dim:
            raise ValueError("measurement has incompatible shape")
        if meas_noise.shape != (meas_dim, meas_dim):
            raise ValueError("meas_noise must have shape (meas_dim, meas_dim)")

        inv_noise_times_matrix = linalg.solve(meas_noise, measurement_matrix)
        inv_noise_times_measurement = linalg.solve(meas_noise, measurement)
        information_matrix = transpose(measurement_matrix) @ inv_noise_times_matrix
        information_vector = transpose(measurement_matrix) @ inv_noise_times_measurement
        return LinearGaussianInformationContribution(
            _symmetrize(information_matrix),
            information_vector,
        )

    @staticmethod
    def aggregate_information_contributions(contributions, *, state_dim=None):
        """Sum independent information contributions."""
        contributions = list(contributions)
        if not contributions:
            raise ValueError("at least one information contribution is required")

        total_matrix = None
        total_vector = None
        for contribution in contributions:
            contribution = _as_information_contribution(contribution)
            contribution_dim = contribution.information_vector.shape[0]
            if state_dim is None:
                state_dim = contribution_dim
            if contribution_dim != state_dim:
                raise ValueError("all information vectors must have state_dim entries")

            if total_matrix is None:
                total_matrix = contribution.information_matrix
                total_vector = contribution.information_vector
            else:
                total_matrix = total_matrix + contribution.information_matrix
                total_vector = total_vector + contribution.information_vector

        return LinearGaussianInformationContribution(_symmetrize(total_matrix), total_vector)

    @staticmethod
    def _posterior_from_information_contribution(prior_state, contribution):
        contribution = _as_information_contribution(contribution)
        state_dim = prior_state.dim
        if contribution.information_vector.shape[0] != state_dim:
            raise ValueError("information contribution dimension must match the node state dimension")

        prior_precision = linalg.solve(prior_state.C, eye(state_dim))
        prior_information_vector = linalg.solve(prior_state.C, prior_state.mu)
        posterior_precision = prior_precision + contribution.information_matrix
        posterior_information_vector = prior_information_vector + contribution.information_vector
        posterior_mean = linalg.solve(posterior_precision, posterior_information_vector)
        posterior_covariance = linalg.solve(posterior_precision, eye(state_dim))
        posterior_covariance = _symmetrize(posterior_covariance)

        return GaussianDistribution(posterior_mean, posterior_covariance, check_validity=False)

    def update_from_information_contribution(self, contribution, *, node_indices=None):
        """Apply one already aggregated information contribution to selected nodes."""
        contribution = _as_information_contribution(contribution)
        if contribution.information_vector.shape[0] != self.dim:
            raise ValueError("information contribution dimension must match the common state dimension")

        for node_filter in self._iter_node_filters(node_indices):
            node_filter.filter_state = self._posterior_from_information_contribution(
                node_filter.filter_state,
                contribution,
            )

    def update_from_information_contributions(self, contributions, *, node_indices=None, return_contribution=False):
        """Aggregate information contributions and apply them to selected nodes."""
        contribution = self.aggregate_information_contributions(contributions, state_dim=self.dim)
        self.update_from_information_contribution(contribution, node_indices=node_indices)
        if return_contribution:
            return contribution
        return None

    def update_distributed_linear(self, local_measurements, measurement_matrices, meas_noises, *, node_indices=None, return_contribution=False):
        """Fuse local linear Gaussian measurements and update selected nodes.

        ``measurement_matrices`` and ``meas_noises`` may either contain one
        entry per local measurement or a single shared entry that is reused.
        """
        local_measurements = _as_sequence(local_measurements)
        measurement_matrices = _broadcast_to_length(
            measurement_matrices,
            len(local_measurements),
            "measurement_matrices",
        )
        meas_noises = _broadcast_to_length(meas_noises, len(local_measurements), "meas_noises")
        contributions = [
            self.local_information_contribution(measurement, measurement_matrix, meas_noise)
            for measurement, measurement_matrix, meas_noise in zip(
                local_measurements,
                measurement_matrices,
                meas_noises,
            )
        ]
        return self.update_from_information_contributions(
            contributions,
            node_indices=node_indices,
            return_contribution=return_contribution,
        )

    def update_distributed_model(self, measurement_models, local_measurements, *, node_indices=None, return_contribution=False):
        """Fuse local measurements described by structural measurement models."""
        local_measurements = _as_sequence(local_measurements)
        measurement_models = _broadcast_to_length(
            measurement_models,
            len(local_measurements),
            "measurement_models",
        )

        contributions = []
        for measurement_model, measurement in zip(measurement_models, local_measurements):
            measurement_matrix = _get_required_model_attribute(measurement_model, "measurement_matrix")
            meas_noise = _get_required_model_attribute(
                measurement_model,
                "meas_noise",
                "measurement_noise_cov",
            )
            contributions.append(
                self.local_information_contribution(measurement, measurement_matrix, meas_noise)
            )

        return self.update_from_information_contributions(
            contributions,
            node_indices=node_indices,
            return_contribution=return_contribution,
        )
