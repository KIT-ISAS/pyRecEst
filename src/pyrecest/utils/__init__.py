"""Utility helpers for :mod:`pyrecest`."""

# pylint: disable=duplicate-code
from .assignment import murty_k_best_assignments
from .association_models import LogisticPairwiseAssociationModel
from .history_recorder import HistoryRecorder
from .multisession_assignment import (
    MultiSessionAssignmentResult,
    solve_multisession_assignment,
    tracks_to_session_labels,
)
from .multisession_assignment_observation_costs import (
    solve_multisession_assignment_with_observation_costs,
)
from .multisession_assignment_score import (
    solve_multisession_assignment_from_similarity,
    stitch_tracks_from_pairwise_scores,
    tracks_to_index_matrix,
)
from .nonrigid_point_set_registration import (
    ThinPlateSplineRegistrationResult,
    ThinPlateSplineTransform,
    estimate_thin_plate_spline,
    joint_tps_registration_assignment,
)
from .roi_similarity import (
    build_roi_cost_matrix,
    pairwise_centroid_distances,
    pairwise_roi_similarity,
    roi_centroid,
    weighted_roi_cosine_similarity,
    weighted_roi_jaccard,
)

__all__ = [
    "MultiSessionAssignmentResult",
    "solve_multisession_assignment",
    "solve_multisession_assignment_from_similarity",
    "solve_multisession_assignment_with_observation_costs",
    "stitch_tracks_from_pairwise_scores",
    "tracks_to_index_matrix",
    "tracks_to_session_labels",
    "LogisticPairwiseAssociationModel",
    "HistoryRecorder",
    "ThinPlateSplineRegistrationResult",
    "ThinPlateSplineTransform",
    "build_roi_cost_matrix",
    "estimate_thin_plate_spline",
    "joint_tps_registration_assignment",
    "murty_k_best_assignments",
    "pairwise_centroid_distances",
    "pairwise_roi_similarity",
    "roi_centroid",
    "weighted_roi_cosine_similarity",
    "weighted_roi_jaccard",
]
