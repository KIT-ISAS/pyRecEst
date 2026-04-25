import unittest

import numpy.testing as npt

# pylint: disable=no-name-in-module,no-member
from pyrecest import backend
from pyrecest.backend import array, concatenate
from pyrecest.utils.roi_assignment import (
    assign_by_similarity_matrix,
    associate_rois_by_iou,
    minimum_similarity_threshold,
    otsu_similarity_threshold,
    pairwise_centroid_distances,
    pairwise_iou_masks,
    roi_centroid,
    roi_iou,
)


class TestRoiIoU(unittest.TestCase):
    def test_roi_iou_dense_masks(self):
        roi_a = array(
            [
                [1, 1, 0],
                [0, 0, 0],
                [0, 0, 0],
            ],
            dtype=bool,
        )
        roi_b = array(
            [
                [1, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
            ],
            dtype=bool,
        )

        self.assertEqual(roi_iou(roi_a, roi_a), 1.0)
        self.assertAlmostEqual(roi_iou(roi_a, roi_b), 0.5)

    @unittest.skipIf(
        backend.__backend_name__ == "jax",
        reason="Not supported on the jax backend",
    )
    def test_pairwise_iou_masks_supports_suite2p_sparse_dicts(self):
        reference_rois = [
            {"ypix": array([0, 0, 1]), "xpix": array([0, 1, 1])},
            {"ypix": array([4, 4]), "xpix": array([4, 5])},
        ]
        query_rois = [
            {"ypix": array([4, 4]), "xpix": array([4, 5])},
            {"ypix": array([0, 0, 1]), "xpix": array([0, 1, 1])},
        ]

        iou_matrix = pairwise_iou_masks(reference_rois, query_rois)

        expected = array(
            [
                [0.0, 1.0],
                [1.0, 0.0],
            ]
        )
        npt.assert_allclose(iou_matrix, expected)

    def test_dense_two_row_list_is_not_misinterpreted_as_sparse_coordinates(self):
        roi_a = [[1, 0], [0, 0]]
        roi_b = [[0, 1], [0, 0]]

        self.assertEqual(roi_iou(roi_a, roi_b), 0.0)

    @unittest.skipIf(
        backend.__backend_name__ == "jax",
        reason="Not supported on the jax backend",
    )
    def test_roi_centroid_and_pairwise_centroid_distances(self):
        roi_a = array(
            [
                [1, 1, 0],
                [0, 0, 0],
                [0, 0, 0],
            ],
            dtype=bool,
        )
        roi_b = array(
            [
                [0, 0, 0],
                [0, 0, 0],
                [0, 1, 1],
            ],
            dtype=bool,
        )

        centroid_a = roi_centroid(roi_a)
        npt.assert_allclose(centroid_a, array([0.0, 0.5]))

        distances = pairwise_centroid_distances([roi_a], [roi_b])
        self.assertGreater(distances[0, 0], 2.0)


class TestThresholds(unittest.TestCase):
    def test_otsu_threshold_separates_low_and_high_scores(self):
        scores = array([0.05, 0.08, 0.1, 0.81, 0.84, 0.9])
        threshold = otsu_similarity_threshold(scores)
        self.assertGreater(threshold, 0.1)
        self.assertLess(threshold, 0.81)

    def test_minimum_threshold_falls_between_two_modes(self):
        scores = concatenate(
            [
                array([0.05, 0.06, 0.07, 0.08, 0.09]),
                array([0.8, 0.82, 0.84, 0.86, 0.88]),
            ]
        )
        threshold = minimum_similarity_threshold(scores)
        self.assertGreater(threshold, 0.1)
        self.assertLess(threshold, 0.8)


class TestSimilarityAssignment(unittest.TestCase):
    @unittest.skipIf(
        backend.__backend_name__ == "jax",
        reason="Not supported on the jax backend",
    )
    def test_assignment_maximizes_global_similarity(self):
        similarity_matrix = array(
            [
                [0.90, 0.80],
                [0.85, 0.10],
            ]
        )

        assignment = assign_by_similarity_matrix(similarity_matrix)
        npt.assert_array_equal(assignment, array([1, 0]))

    @unittest.skipIf(
        backend.__backend_name__ == "jax",
        reason="Not supported on the jax backend",
    )
    def test_assignment_keeps_match_at_exact_threshold(self):
        similarity_matrix = array([[0.5]])
        assignment = assign_by_similarity_matrix(similarity_matrix, min_similarity=0.5)
        npt.assert_array_equal(assignment, array([0]))

    @unittest.skipIf(
        backend.__backend_name__ == "jax",
        reason="Not supported on the jax backend",
    )
    def test_assignment_result_contains_bookkeeping(self):
        similarity_matrix = array(
            [
                [1.0, 0.0],
                [0.0, 0.0],
            ]
        )
        result = assign_by_similarity_matrix(
            similarity_matrix,
            min_similarity=0.1,
            return_result=True,
        )
        npt.assert_array_equal(result.assignment, array([0, -1]))
        npt.assert_array_equal(result.matched_row_indices, array([0]))
        npt.assert_array_equal(result.unmatched_row_indices, array([1]))


class TestRoiAssociation(unittest.TestCase):
    @unittest.skipIf(
        backend.__backend_name__ == "jax",
        reason="Not supported on the jax backend",
    )
    def test_associate_rois_by_iou_recovers_crossed_order(self):
        reference_rois = [
            array(
                [
                    [1, 1, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                ],
                dtype=bool,
            ),
            array(
                [
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 1, 1],
                    [0, 0, 0, 0],
                ],
                dtype=bool,
            ),
        ]
        query_rois = [reference_rois[1], reference_rois[0]]

        assignment, iou_matrix = associate_rois_by_iou(
            reference_rois,
            query_rois,
            return_iou_matrix=True,
        )

        npt.assert_array_equal(assignment, array([1, 0]))
        npt.assert_allclose(iou_matrix, array([[0.0, 1.0], [1.0, 0.0]]))

    @unittest.skipIf(
        backend.__backend_name__ == "jax",
        reason="Not supported on the jax backend",
    )
    def test_associate_rois_by_iou_rejects_zero_overlap_by_default(self):
        reference_rois = [
            array(
                [
                    [1, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                ],
                dtype=bool,
            )
        ]
        query_rois = [
            array(
                [
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 1],
                ],
                dtype=bool,
            )
        ]

        assignment = associate_rois_by_iou(reference_rois, query_rois)
        npt.assert_array_equal(assignment, array([-1]))

    @unittest.skipIf(
        backend.__backend_name__ == "jax",
        reason="Not supported on the jax backend",
    )
    def test_associate_rois_by_iou_supports_centroid_gating(self):
        reference_rois = [
            array(
                [
                    [1, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                ],
                dtype=bool,
            )
        ]
        query_rois = [
            array(
                [
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 1],
                ],
                dtype=bool,
            )
        ]

        result = associate_rois_by_iou(
            reference_rois,
            query_rois,
            centroid_distance_threshold=1.0,
            return_result=True,
        )
        npt.assert_array_equal(result.assignment, array([-1]))
        self.assertGreater(result.centroid_distance_matrix[0, 0], 1.0)

    @unittest.skipIf(
        backend.__backend_name__ == "jax",
        reason="Not supported on the jax backend",
    )
    def test_associate_rois_by_iou_supports_otsu_post_filtering(self):
        reference_rois = [
            array([[1, 1, 0, 0]], dtype=bool),
            array([[0, 0, 1, 1]], dtype=bool),
            array([[0, 0, 0, 0, 1, 1]], dtype=bool),
        ]
        query_rois = [
            array([[1, 1, 0, 0]], dtype=bool),
            array([[0, 0, 1, 0]], dtype=bool),
            array([[0, 0, 0, 0, 1, 1]], dtype=bool),
        ]

        result = associate_rois_by_iou(
            reference_rois,
            query_rois,
            threshold_method="otsu",
            return_result=True,
        )
        npt.assert_array_equal(result.assignment, array([0, -1, 2]))
        self.assertEqual(result.threshold_method, "otsu")
        self.assertIsNotNone(result.acceptance_threshold)
        npt.assert_array_equal(result.matched_reference_indices, array([0, 2]))
        npt.assert_array_equal(result.unmatched_reference_indices, array([1]))


if __name__ == "__main__":
    unittest.main()
