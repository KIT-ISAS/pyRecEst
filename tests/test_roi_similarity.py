import math
import unittest

import numpy.testing as npt

from pyrecest.backend import array  # pylint: disable=no-name-in-module
from pyrecest.utils.roi_similarity import (
    build_roi_cost_matrix,
    pairwise_centroid_distances,
    pairwise_roi_similarity,
    roi_centroid,
    weighted_roi_cosine_similarity,
    weighted_roi_jaccard,
)


class TestWeightedRoiSimilarity(unittest.TestCase):
    def test_weighted_jaccard_uses_suite2p_lam(self):
        roi_a = {
            "ypix": array([0, 0]),
            "xpix": array([0, 1]),
            "lam": array([1.0, 1.0]),
        }
        roi_b = {
            "ypix": array([0, 0]),
            "xpix": array([0, 1]),
            "lam": array([1.0, 0.5]),
        }

        self.assertAlmostEqual(weighted_roi_jaccard(roi_a, roi_b), 0.75)

    def test_cosine_similarity_supports_dense_weighted_masks(self):
        roi_a = array([[0.0, 2.0], [0.0, 1.0]])
        roi_b = array([[0.0, 1.0], [0.0, 1.0]])

        expected = 3.0 / math.sqrt(10.0)
        self.assertAlmostEqual(weighted_roi_cosine_similarity(roi_a, roi_b), expected)

    def test_exclude_overlap_drops_suite2p_overlapping_pixels(self):
        roi_a = {
            "ypix": array([0, 0, 1]),
            "xpix": array([0, 1, 1]),
            "lam": array([1.0, 1.0, 1.0]),
            "overlap": array([False, True, False], dtype=bool),
        }
        roi_b = {
            "ypix": array([0, 1]),
            "xpix": array([0, 1]),
            "lam": array([1.0, 1.0]),
        }

        self.assertAlmostEqual(weighted_roi_jaccard(roi_a, roi_b), 2.0 / 3.0)
        self.assertAlmostEqual(
            weighted_roi_jaccard(roi_a, roi_b, exclude_overlap=True), 1.0
        )

    def test_pairwise_similarity_recovers_crossed_order(self):
        reference_rois = [
            {
                "ypix": array([0, 0]),
                "xpix": array([0, 1]),
                "lam": array([1.0, 0.8]),
            },
            {
                "ypix": array([5, 5]),
                "xpix": array([5, 6]),
                "lam": array([1.0, 1.0]),
            },
        ]
        query_rois = [reference_rois[1], reference_rois[0]]

        similarity_matrix = pairwise_roi_similarity(reference_rois, query_rois)

        npt.assert_allclose(similarity_matrix, array([[0.0, 1.0], [1.0, 0.0]]))


class TestRoiCentroids(unittest.TestCase):
    def test_centroid_prefers_med_when_present(self):
        roi = {
            "ypix": array([0, 2]),
            "xpix": array([0, 2]),
            "lam": array([1.0, 3.0]),
            "med": array([10.0, 20.0]),
        }

        npt.assert_allclose(roi_centroid(roi), array([10.0, 20.0]))
        npt.assert_allclose(
            roi_centroid(roi, prefer_med=False), array([1.5, 1.5])
        )

    def test_pairwise_centroid_distances_use_med_if_available(self):
        reference_rois = [
            {"ypix": array([0]), "xpix": array([0]), "med": array([1.0, 1.0])}
        ]
        query_rois = [
            {"ypix": array([0]), "xpix": array([0]), "med": array([4.0, 5.0])}
        ]

        distances = pairwise_centroid_distances(reference_rois, query_rois)
        npt.assert_allclose(distances, array([[5.0]]))

    def test_pairwise_centroid_distances_handles_empty_queries(self):
        reference_rois = [
            {"ypix": array([0]), "xpix": array([0]), "med": array([1.0, 1.0])}
        ]

        distances = pairwise_centroid_distances(reference_rois, [])
        self.assertEqual(distances.shape, (1, 0))


class TestRoiCostMatrix(unittest.TestCase):
    def test_build_roi_cost_matrix_combines_similarity_and_centroid_gating(self):
        reference_rois = [
            {
                "ypix": array([0, 0]),
                "xpix": array([0, 1]),
                "lam": array([1.0, 1.0]),
                "med": array([0.0, 0.5]),
            },
            {
                "ypix": array([5, 5]),
                "xpix": array([5, 6]),
                "lam": array([1.0, 1.0]),
                "med": array([5.0, 5.5]),
            },
        ]
        query_rois = [reference_rois[1], reference_rois[0]]

        cost_matrix, similarity_matrix, centroid_distances = build_roi_cost_matrix(
            reference_rois,
            query_rois,
            centroid_weight=0.25,
            centroid_scale=1.0,
            max_centroid_distance=2.0,
            return_components=True,
        )

        self.assertTrue(math.isinf(float(cost_matrix[0, 0])))
        self.assertTrue(math.isinf(float(cost_matrix[1, 1])))
        self.assertEqual(cost_matrix[0, 1], 0.0)
        self.assertEqual(cost_matrix[1, 0], 0.0)
        npt.assert_allclose(similarity_matrix, array([[0.0, 1.0], [1.0, 0.0]]))
        npt.assert_allclose(
            centroid_distances,
            array(
                [
                    [math.sqrt(50.0), 0.0],
                    [0.0, math.sqrt(50.0)],
                ]
            ),
        )

    def test_build_roi_cost_matrix_rejects_low_similarity(self):
        reference_rois = [
            {"ypix": array([0]), "xpix": array([0]), "lam": array([1.0])}
        ]
        query_rois = [
            {"ypix": array([3]), "xpix": array([3]), "lam": array([1.0])}
        ]

        cost_matrix = build_roi_cost_matrix(
            reference_rois,
            query_rois,
            min_similarity=0.1,
        )

        self.assertTrue(math.isinf(float(cost_matrix[0, 0])))


if __name__ == "__main__":
    unittest.main()
