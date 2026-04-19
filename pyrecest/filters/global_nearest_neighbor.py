# pylint: disable=redefined-builtin,no-name-in-module,no-member
import warnings

import numpy as _np
from pyrecest.backend import all, empty, full, repeat, squeeze, stack
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from scipy.stats import chi2

from .abstract_nearest_neighbor_tracker import AbstractNearestNeighborTracker


class GlobalNearestNeighbor(AbstractNearestNeighborTracker):
    def __init__(
        self,
        initial_prior=None,
        association_param=None,
        log_prior_estimates=True,
        log_posterior_estimates=True,
    ):
        default_association_param = {
            "distance_metric_pos": "Mahalanobis",
            "square_dist": True,
            "max_new_tracks": 10,
            "gating_distance_threshold": chi2.ppf(0.999, 2) ** 2,
            "infeasible_assignment_cost": None,
        }
        if association_param is None:
            association_param = default_association_param
        else:
            association_param = {
                **default_association_param,
                **association_param,
            }

        super().__init__(
            initial_prior,
            association_param,
            log_prior_estimates=log_prior_estimates,
            log_posterior_estimates=log_posterior_estimates,
        )

    @staticmethod
    def _coerce_cost_matrix(cost_matrix):
        pairwise_costs = _np.asarray(cost_matrix, dtype=float)
        if pairwise_costs.ndim != 2:
            raise ValueError("pairwise_cost_matrix must be two-dimensional")
        return pairwise_costs

    def find_association_from_cost_matrix(
        self,
        pairwise_cost_matrix,
        warn_on_no_meas_for_track=True,
    ):
        pairwise_cost_matrix = self._coerce_cost_matrix(pairwise_cost_matrix)
        n_targets, n_meas = pairwise_cost_matrix.shape

        gating_cost = float(self.association_param["gating_distance_threshold"])
        infeasible_assignment_cost = self.association_param[
            "infeasible_assignment_cost"
        ]
        if infeasible_assignment_cost is None:
            infeasible_assignment_cost = gating_cost + 1.0
        infeasible_assignment_cost = float(infeasible_assignment_cost)
        if infeasible_assignment_cost <= gating_cost:
            raise ValueError(
                "infeasible_assignment_cost must be greater than gating_distance_threshold"
            )

        pad_to = max(n_targets, n_meas) + int(self.association_param["max_new_tracks"])
        association_matrix = full((pad_to, pad_to), gating_cost)

        finite_mask = _np.isfinite(pairwise_cost_matrix)
        association_matrix[:n_targets, :n_meas] = _np.where(
            finite_mask,
            pairwise_cost_matrix,
            infeasible_assignment_cost,
        )

        _, col_ind = linear_sum_assignment(association_matrix)
        association = col_ind[:n_targets]

        if warn_on_no_meas_for_track and _np.any(association >= n_meas):
            warnings.warn(
                "GNN: No measurement was within the gating threshold for at least one target.",
                stacklevel=2,
            )

        return association

    # pylint: disable=too-many-locals
    def find_association(
        self,
        measurements,
        measurement_matrix,
        cov_mats_meas,
        warn_on_no_meas_for_track=True,
    ):
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
                if cov_mats_meas.ndim == 2:
                    meas_cov = cov_mats_meas
                else:
                    meas_cov = cov_mats_meas[:, :, 0]
                curr_cov_mahalanobis = (
                    measurement_matrix
                    @ all_cov_mats_prior[:, :, 0]
                    @ measurement_matrix.T
                    + meas_cov
                )
                dists = cdist(
                    (measurement_matrix @ all_means_prior).T,
                    measurements.T,
                    "mahalanobis",
                    VI=curr_cov_mahalanobis,
                )
            elif all_cov_mat_meas_equal:
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
                        + cov_mats_meas
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
                            + cov_mats_meas[:, :, j]
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

        return self.find_association_from_cost_matrix(
            dists,
            warn_on_no_meas_for_track=warn_on_no_meas_for_track,
        )
