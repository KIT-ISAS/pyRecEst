"""Tests for global multi-session association."""

import unittest
from unittest.mock import patch

import numpy as np

import pyrecest.utils.multisession_assignment as multisession_assignment_module
from pyrecest.backend import (  # pylint: disable=no-name-in-module
    __backend_name__,
    array,
    array_equal,
)
from pyrecest.utils import solve_multisession_assignment, tracks_to_session_labels


class TestMultiSessionAssignment(unittest.TestCase):
    @staticmethod
    def _canonical_tracks(tracks):
        return sorted(tuple(sorted(track.items())) for track in tracks)

    @staticmethod
    def _assert_valid_matching(mask, left_nodes, right_nodes):
        selected_indices = np.flatnonzero(mask)
        if selected_indices.size == 0:
            return
        assert np.unique(left_nodes[selected_indices]).size == selected_indices.size
        assert np.unique(right_nodes[selected_indices]).size == selected_indices.size

    @unittest.skipIf(
        __backend_name__ == "jax",
        reason="Not supported on this backend",
    )
    def test_singletons_without_edges(self):
        result = solve_multisession_assignment(
            {},
            session_sizes=[2, 1, 2],
            start_cost=1.0,
            end_cost=1.0,
        )

        expected_tracks = [
            ((0, 0),),
            ((0, 1),),
            ((1, 0),),
            ((2, 0),),
            ((2, 1),),
        ]
        self.assertEqual(self._canonical_tracks(result.tracks), expected_tracks)
        self.assertAlmostEqual(result.total_cost, 10.0)
        self.assertEqual(result.matched_edges, [])

    @unittest.skipIf(
        __backend_name__ == "jax",
        reason="Not supported on this backend",
    )
    def test_consecutive_costs_form_two_long_tracks(self):
        result = solve_multisession_assignment(
            [
                array([[0.1, 8.0], [8.0, 0.2]], dtype=float),
                array([[0.1, 8.0], [8.0, 0.2]], dtype=float),
            ],
            start_cost=5.0,
            end_cost=5.0,
            cost_threshold=10.0,
        )

        expected_tracks = [
            ((0, 0), (1, 0), (2, 0)),
            ((0, 1), (1, 1), (2, 1)),
        ]
        self.assertEqual(self._canonical_tracks(result.tracks), expected_tracks)
        self.assertAlmostEqual(result.total_cost, 20.6)

    @unittest.skipIf(
        __backend_name__ == "jax",
        reason="Not supported on this backend",
    )
    def test_cross_gap_linking_is_supported(self):
        result = solve_multisession_assignment(
            {(0, 2): array([[0.3]], dtype=float)},
            session_sizes=[1, 0, 1],
            start_cost=4.0,
            end_cost=4.0,
            gap_penalty=0.5,
        )

        expected_tracks = [((0, 0), (2, 0))]
        self.assertEqual(self._canonical_tracks(result.tracks), expected_tracks)
        self.assertAlmostEqual(result.total_cost, 8.8)
        self.assertEqual(result.matched_edges, [((0, 0), (2, 0), 0.8)])

    @unittest.skipIf(
        __backend_name__ == "jax",
        reason="Not supported on this backend",
    )
    def test_global_solution_beats_pairwise_greedy_choice(self):
        result = solve_multisession_assignment(
            [
                array([[0.0, 1.0], [1.0, 100.0]], dtype=float),
                array([[100.0, 0.0], [0.0, 100.0]], dtype=float),
            ],
            start_cost=10.0,
            end_cost=10.0,
            cost_threshold=150.0,
        )

        expected_tracks = [
            ((0, 0), (1, 1), (2, 0)),
            ((0, 1), (1, 0), (2, 1)),
        ]
        self.assertEqual(self._canonical_tracks(result.tracks), expected_tracks)
        self.assertAlmostEqual(result.total_cost, 42.0)

    @unittest.skipIf(
        __backend_name__ == "jax",
        reason="Not supported on this backend",
    )
    def test_tracks_to_session_labels(self):
        result = solve_multisession_assignment(
            {(0, 2): array([[0.3]], dtype=float)},
            session_sizes=[1, 0, 1],
            start_cost=4.0,
            end_cost=4.0,
            gap_penalty=0.5,
        )

        labels = result.to_session_labels(session_sizes=[1, 0, 1])
        self.assertTrue(array_equal(labels[0], array([0])))
        self.assertEqual(labels[1].shape[0], 0)
        self.assertTrue(array_equal(labels[2], array([0])))

        labels_from_function = tracks_to_session_labels(result.tracks, session_sizes=[1, 0, 1])
        self.assertTrue(array_equal(labels_from_function[0], array([0])))
        self.assertEqual(labels_from_function[1].shape[0], 0)
        self.assertTrue(array_equal(labels_from_function[2], array([0])))

    @unittest.skipIf(
        __backend_name__ == "jax",
        reason="Not supported on this backend",
    )
    def test_rejects_inconsistent_session_sizes(self):
        pairwise_costs = {
            (0, 1): array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
            (1, 2): array([[0.0], [0.0], [0.0], [0.0]]),
        }

        with self.assertRaises(ValueError):
            solve_multisession_assignment(pairwise_costs)

    @unittest.skipIf(
        __backend_name__ == "jax",
        reason="Not supported on this backend",
    )
    def test_drops_non_beneficial_links_even_without_threshold(self):
        result = solve_multisession_assignment(
            [array([[5.0]], dtype=float)],
            start_cost=1.0,
            end_cost=1.0,
        )

        self.assertEqual(self._canonical_tracks(result.tracks), [((0, 0),), ((1, 0),)])
        self.assertAlmostEqual(result.total_cost, 4.0)

    @unittest.skipIf(
        __backend_name__ == "jax",
        reason="Not supported on this backend",
    )
    def test_sparse_backend_matches_previous_linprog_backend(self):
        rng = np.random.default_rng(42)
        num_nodes = 12
        all_pairs = [(left, right) for left in range(num_nodes) for right in range(num_nodes)]
        selected_pairs = rng.choice(len(all_pairs), size=40, replace=False)

        left_nodes = np.array([all_pairs[index][0] for index in selected_pairs], dtype=int)
        right_nodes = np.array([all_pairs[index][1] for index in selected_pairs], dtype=int)
        edge_gains = rng.uniform(0.1, 10.0, size=selected_pairs.size)

        sparse_mask = multisession_assignment_module._solve_max_weight_matching(
            left_nodes,
            right_nodes,
            edge_gains,
            num_nodes,
        )
        linprog_mask = multisession_assignment_module._solve_max_weight_matching_via_linprog(
            left_nodes,
            right_nodes,
            edge_gains,
            num_nodes,
        )

        self._assert_valid_matching(sparse_mask, left_nodes, right_nodes)
        self._assert_valid_matching(linprog_mask, left_nodes, right_nodes)
        self.assertAlmostEqual(
            float(edge_gains[sparse_mask].sum()),
            float(edge_gains[linprog_mask].sum()),
        )

    @unittest.skipIf(
        __backend_name__ == "jax",
        reason="Not supported on this backend",
    )
    def test_sparse_backend_falls_back_to_linprog_when_requested(self):
        pairwise_costs = [
            np.array([[0.1, 8.0], [8.0, 0.2]], dtype=float),
            np.array([[0.1, 8.0], [8.0, 0.2]], dtype=float),
        ]

        with patch.object(
            multisession_assignment_module,
            "min_weight_full_bipartite_matching",
            side_effect=ValueError("forced sparse-backend failure"),
        ):
            result = solve_multisession_assignment(
                pairwise_costs,
                start_cost=5.0,
                end_cost=5.0,
                cost_threshold=10.0,
            )

        expected_tracks = [
            ((0, 0), (1, 0), (2, 0)),
            ((0, 1), (1, 1), (2, 1)),
        ]
        self.assertEqual(self._canonical_tracks(result.tracks), expected_tracks)
        self.assertAlmostEqual(result.total_cost, 20.6)


if __name__ == "__main__":
    unittest.main()
