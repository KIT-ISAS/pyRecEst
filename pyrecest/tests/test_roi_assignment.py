import unittest

import numpy as np
import numpy.testing as npt

import pyrecest.backend
from pyrecest.utils.roi_assignment import (
    assign_by_similarity_matrix,
    associate_rois_by_iou,
    pairwise_iou_masks,
    roi_iou,
)


class TestRoiIoU(unittest.TestCase):
    def test_roi_iou_dense_masks(self):
        roi_a = np.array(
            [
                [1, 1, 0],
                [0, 0, 0],
                [0, 0, 0],
            ],
            dtype=bool,
        )
        roi_b = np.array(
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
        pyrecest.backend.__backend_name__ == "jax",
        reason="Not supported on the jax backend",
    )
    def test_pairwise_iou_masks_supports_suite2p_sparse_dicts(self):
        reference_rois = [
            {"ypix": np.array([0, 0, 1]), "xpix": np.array([0, 1, 1])},
            {"ypix": np.array([4, 4]), "xpix": np.array([4, 5])},
        ]
        query_rois = [
            {"ypix": np.array([4, 4]), "xpix": np.array([4, 5])},
            {"ypix": np.array([0, 0, 1]), "xpix": np.array([0, 1, 1])},
        ]

        iou_matrix = pairwise_iou_masks(reference_rois, query_rois)

        expected = np.array(
            [
                [0.0, 1.0],
                [1.0, 0.0],
            ]
        )
        npt.assert_allclose(iou_matrix, expected)


class TestSimilarityAssignment(unittest.TestCase):
    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        reason="Not supported on the jax backend",
    )
    def test_assignment_maximizes_global_similarity(self):
        similarity_matrix = np.array(
            [
                [0.90, 0.80],
                [0.85, 0.10],
            ]
        )

        assignment = assign_by_similarity_matrix(similarity_matrix)
        npt.assert_array_equal(assignment, np.array([1, 0]))

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        reason="Not supported on the jax backend",
    )
    def test_assignment_keeps_match_at_exact_threshold(self):
        similarity_matrix = np.array([[0.5]])
        assignment = assign_by_similarity_matrix(similarity_matrix, min_similarity=0.5)
        npt.assert_array_equal(assignment, np.array([0]))


class TestRoiAssociation(unittest.TestCase):
    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        reason="Not supported on the jax backend",
    )
    def test_associate_rois_by_iou_recovers_crossed_order(self):
        reference_rois = [
            np.array(
                [
                    [1, 1, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                ],
                dtype=bool,
            ),
            np.array(
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

        npt.assert_array_equal(assignment, np.array([1, 0]))
        npt.assert_allclose(iou_matrix, np.array([[0.0, 1.0], [1.0, 0.0]]))

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        reason="Not supported on the jax backend",
    )
    def test_associate_rois_by_iou_rejects_low_overlap(self):
        reference_rois = [
            np.array(
                [
                    [1, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                ],
                dtype=bool,
            )
        ]
        query_rois = [
            np.array(
                [
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 1],
                ],
                dtype=bool,
            )
        ]

        assignment = associate_rois_by_iou(reference_rois, query_rois, min_iou=0.1)
        npt.assert_array_equal(assignment, np.array([-1]))


if __name__ == "__main__":
    unittest.main()
