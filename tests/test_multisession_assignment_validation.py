"""Input validation tests for multi-session assignment."""

import unittest

import numpy as np

from pyrecest.backend import __backend_name__, array
from pyrecest.utils import solve_multisession_assignment, tracks_to_session_labels


class TestMultiSessionAssignmentValidation(unittest.TestCase):
    @unittest.skipIf(
        __backend_name__ == "jax",
        reason="Not supported on this backend",
    )
    def test_accepts_scalar_integer_like_session_sizes(self):
        result = solve_multisession_assignment(
            {},
            session_sizes=[np.array(1.0), 0],
            start_cost=1.0,
            end_cost=1.0,
        )

        self.assertEqual(result.tracks, [{0: 0}])
        self.assertAlmostEqual(result.total_cost, 2.0)

    @unittest.skipIf(
        __backend_name__ == "jax",
        reason="Not supported on this backend",
    )
    def test_rejects_non_integer_session_sizes(self):
        invalid_session_sizes = (
            [True],
            [1.5],
            [np.nan],
            [np.inf],
            [-1],
            [np.array([1])],
            {True: 1},
            {0: True},
        )

        for session_sizes in invalid_session_sizes:
            with self.subTest(session_sizes=session_sizes):
                with self.assertRaisesRegex(
                    ValueError,
                    "must be a non-negative integer|negative detection count",
                ):
                    solve_multisession_assignment({}, session_sizes=session_sizes)

    @unittest.skipIf(
        __backend_name__ == "jax",
        reason="Not supported on this backend",
    )
    def test_rejects_non_integer_pairwise_session_indices(self):
        invalid_pairwise_costs = (
            {(True, 2): array([[0.1]], dtype=float)},
            {(0, 1.5): array([[0.1]], dtype=float)},
            {(np.nan, 1): array([[0.1]], dtype=float)},
        )

        for pairwise_costs in invalid_pairwise_costs:
            with self.subTest(pairwise_costs=pairwise_costs):
                with self.assertRaisesRegex(
                    ValueError,
                    "Session indices must be a non-negative integer",
                ):
                    solve_multisession_assignment(pairwise_costs)

    @unittest.skipIf(
        __backend_name__ == "jax",
        reason="Not supported on this backend",
    )
    def test_rejects_non_integer_track_indices(self):
        invalid_tracks = (
            [{1.5: 0}],
            [{0: True}],
            [[(0, 1.5)]],
        )

        for tracks in invalid_tracks:
            with self.subTest(tracks=tracks):
                with self.assertRaisesRegex(
                    ValueError,
                    "must be a non-negative integer",
                ):
                    tracks_to_session_labels(tracks)


if __name__ == "__main__":
    unittest.main()
