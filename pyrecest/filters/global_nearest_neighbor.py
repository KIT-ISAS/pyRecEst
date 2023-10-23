# pylint: disable=redefined-builtin,no-name-in-module,no-member
# pylint: disable=no-name-in-module,no-member
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
                curr_cov_mahalanobis = (
                    measurement_matrix
                    @ all_cov_mats_prior[:, :, 0]
                    @ measurement_matrix.T
                    + cov_mats_meas[:, :, 0]
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

        # Pad to square and add max_new_tracks rows and columns
        pad_to = max(n_targets, n_meas) + self.association_param["max_new_tracks"]
        association_matrix = full(
            (pad_to, pad_to), self.association_param["gating_distance_threshold"]
        )
        association_matrix[: dists.shape[0], : dists.shape[1]] = dists

        # Use the Hungarian algorithm to find the optimal assignment
        _, col_ind = linear_sum_assignment(association_matrix)

        association = col_ind[:n_targets]

        if warn_on_no_meas_for_track and backend.any(association > n_meas):
            print(
                "GNN: No measurement was within gating threshold for at least one target."
            )

        return association
