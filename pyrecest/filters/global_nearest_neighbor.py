# pylint: disable=redefined-builtin,no-name-in-module,no-member
import warnings

import numpy as _np
from pyrecest.backend import all, empty, repeat, squeeze, stack
from scipy.spatial.distance import cdist
from scipy.stats import chi2

from .abstract_nearest_neighbor_tracker import AbstractNearestNeighborTracker
from .track_manager import solve_global_nearest_neighbor


class GlobalNearestNeighbor(AbstractNearestNeighborTracker):
    def __init__(
        self,
        initial_prior=None,
        association_param=None,
        log_prior_estimates=True,
        log_posterior_estimates=True,
    ):
        if association_param is None:
            association_param = {
                "distance_metric_pos": "Mahalanobis",
                "square_dist": True,
                "max_new_tracks": 10,
                "gating_distance_threshold": chi2.ppf(0.999, 2) ** 2,
            }

        super().__init__(
            initial_prior,
            association_param,
            log_prior_estimates=log_prior_estimates,
            log_posterior_estimates=log_posterior_estimates,
        )

    @staticmethod
    def _get_measurement_covariance(cov_mats_meas, measurement_index=None):
        if cov_mats_meas.ndim == 2:
            return cov_mats_meas
        if measurement_index is None:
            return cov_mats_meas[:, :, 0]
        return cov_mats_meas[:, :, measurement_index]

    # pylint: disable=too-many-locals
    def build_association_cost_matrix(self, measurements, measurement_matrix, cov_mats_meas):
        """Build the track-to-measurement cost matrix used by GNN."""

        n_targets = len(self.filter_bank)
        n_meas = measurements.shape[1]

        assert cov_mats_meas.ndim == 2 or cov_mats_meas.shape[2] == n_meas
        all_gaussians = [filter.filter_state for filter in self.filter_bank]
        all_means_prior = stack([gaussian.mu for gaussian in all_gaussians], axis=1)
        all_cov_mats_prior = stack([gaussian.C for gaussian in all_gaussians], axis=2)

        if self.association_param["distance_metric_pos"].lower() == "euclidean":
            dists = cdist(
                measurements.T, (measurement_matrix @ all_means_prior).T, "euclidean"
            ).T
        elif self.association_param["distance_metric_pos"].lower() == "mahalanobis":
            dists = empty((n_targets, n_meas))

            all_cov_mat_state_equal = all(
                all_cov_mats_prior
                == repeat(
                    all_cov_mats_prior[:, :, 0][:, :, None],
                    all_cov_mats_prior.shape[2],
                    axis=2,
                )
            )
            all_cov_mat_meas_equal = cov_mats_meas.ndim == 2 or all(
                cov_mats_meas
                == repeat(
                    cov_mats_meas[:, :, 0][:, :, None],
                    cov_mats_meas.shape[2],
                    axis=2,
                )
            )

            if all_cov_mat_meas_equal and all_cov_mat_state_equal:
                curr_cov_mahalanobis = (
                    measurement_matrix
                    @ all_cov_mats_prior[:, :, 0]
                    @ measurement_matrix.T
                    + self._get_measurement_covariance(cov_mats_meas)
                )
                dists = cdist(
                    (measurement_matrix @ all_means_prior).T,
                    measurements.T,
                    "mahalanobis",
                    VI=curr_cov_mahalanobis,
                )
            elif all_cov_mat_meas_equal:
                meas_covariance = self._get_measurement_covariance(cov_mats_meas)
                all_mats_mahalanobis = empty(
                    (
                        measurements.shape[0],
                        measurements.shape[0],
                        all_cov_mats_prior.shape[2],
                    )
                )
                for i in range(all_cov_mats_prior.shape[2]):
                    all_mats_mahalanobis[:, :, i] = (
                        measurement_matrix
                        @ all_cov_mats_prior[:, :, i]
                        @ measurement_matrix.T
                        + meas_covariance
                    )
                for i in range(n_targets):
                    dists[i, :] = cdist(
                        (measurement_matrix @ all_means_prior[:, i]).T[None],
                        measurements.T,
                        "mahalanobis",
                        VI=all_mats_mahalanobis[:, :, i],
                    )
            else:
                for i in range(n_targets):
                    for j in range(n_meas):
                        curr_cov_mahalanobis = (
                            measurement_matrix
                            @ all_cov_mats_prior[:, :, i]
                            @ measurement_matrix.T
                            + self._get_measurement_covariance(cov_mats_meas, j)
                        )
                        dists[i, j] = squeeze(
                            cdist(
                                (measurement_matrix @ all_means_prior[:, i]).T[None],
                                measurements[:, j].T[None],
                                "mahalanobis",
                                VI=curr_cov_mahalanobis,
                            )
                        )
        else:
            raise ValueError("Association scheme not recognized")

        return dists

    def find_association_result(
        self,
        measurements,
        measurement_matrix,
        cov_mats_meas,
        warn_on_no_meas_for_track=True,
    ):
        """Return the richer association result used by :mod:`track_manager`."""

        dists = self.build_association_cost_matrix(
            measurements,
            measurement_matrix,
            cov_mats_meas,
        )
        gating_distance_threshold = self.association_param["gating_distance_threshold"]
        association_result = solve_global_nearest_neighbor(
            dists,
            unassigned_track_cost=gating_distance_threshold,
            unassigned_measurement_cost=gating_distance_threshold,
            invalid_cost=gating_distance_threshold,
            dummy_dummy_cost=gating_distance_threshold,
        )

        if warn_on_no_meas_for_track and association_result.unmatched_track_indices:
            warnings.warn(
                "GNN: No measurement was within gating threshold for at least one target."
            )

        return association_result

    def find_association(
        self,
        measurements,
        measurement_matrix,
        cov_mats_meas,
        warn_on_no_meas_for_track=True,
    ):
        association_result = self.find_association_result(
            measurements,
            measurement_matrix,
            cov_mats_meas,
            warn_on_no_meas_for_track=warn_on_no_meas_for_track,
        )

        n_targets = len(self.filter_bank)
        n_meas = measurements.shape[1]
        association = _np.full(n_targets, n_meas, dtype=int)
        for track_index, measurement_index in association_result.matches:
            association[track_index] = measurement_index
        return association
