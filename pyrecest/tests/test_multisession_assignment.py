"""Tests for global multi-session association."""

import unittest

import numpy as np

from pyrecest.utils import solve_multisession_assignment, tracks_to_session_labels


class TestMultiSessionAssignment(unittest.TestCase):
    @staticmethod
    def _canonical_tracks(tracks):
        return sorted(tuple(sorted(track.items())) for track in tracks)

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

    def test_consecutive_costs_form_two_long_tracks(self):
        result = solve_multisession_assignment(
            [
                np.array([[0.1, 8.0], [8.0, 0.2]], dtype=float),
                np.array([[0.1, 8.0], [8.0, 0.2]], dtype=float),
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

    def test_cross_gap_linking_is_supported(self):
        result = solve_multisession_assignment(
            {(0, 2): np.array([[0.3]], dtype=float)},
            session_sizes=[1, 0, 1],
            start_cost=4.0,
            end_cost=4.0,
            gap_penalty=0.5,
        )

        expected_tracks = [((0, 0), (2, 0))]
        self.assertEqual(self._canonical_tracks(result.tracks), expected_tracks)
        self.assertAlmostEqual(result.total_cost, 8.8)
        self.assertEqual(result.matched_edges, [((0, 0), (2, 0), 0.8)])

    def test_global_solution_beats_pairwise_greedy_choice(self):
        result = solve_multisession_assignment(
            [
                np.array([[0.0, 1.0], [1.0, 100.0]], dtype=float),
                np.array([[100.0, 0.0], [0.0, 100.0]], dtype=float),
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

    def test_tracks_to_session_labels(self):
        result = solve_multisession_assignment(
            {(0, 2): np.array([[0.3]], dtype=float)},
            session_sizes=[1, 0, 1],
            start_cost=4.0,
            end_cost=4.0,
            gap_penalty=0.5,
        )

        labels = result.to_session_labels(session_sizes=[1, 0, 1])
        self.assertTrue(np.array_equal(labels[0], np.array([0])))
        self.assertEqual(labels[1].size, 0)
        self.assertTrue(np.array_equal(labels[2], np.array([0])))

        labels_from_function = tracks_to_session_labels(result.tracks, session_sizes=[1, 0, 1])
        self.assertTrue(np.array_equal(labels_from_function[0], np.array([0])))
        self.assertEqual(labels_from_function[1].size, 0)
        self.assertTrue(np.array_equal(labels_from_function[2], np.array([0])))

    def test_rejects_inconsistent_session_sizes(self):
        pairwise_costs = {
            (0, 1): np.zeros((2, 3)),
            (1, 2): np.zeros((4, 1)),
        }

        with self.assertRaises(ValueError):
            solve_multisession_assignment(pairwise_costs)

    def test_drops_non_beneficial_links_even_without_threshold(self):
        result = solve_multisession_assignment(
            [np.array([[5.0]], dtype=float)],
            start_cost=1.0,
            end_cost=1.0,
        )

        self.assertEqual(self._canonical_tracks(result.tracks), [((0, 0),), ((1, 0),)])
        self.assertAlmostEqual(result.total_cost, 4.0)


if __name__ == "__main__":
    unittest.main()
