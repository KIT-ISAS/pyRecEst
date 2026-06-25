"""Regression tests for strict track-evaluation session selectors."""

import unittest

import numpy as np
from pyrecest.utils.track_evaluation import complete_track_set, track_pair_set


class TestTrackEvaluationSessionValidation(unittest.TestCase):
    def test_complete_track_set_rejects_noninteger_session_indices(self):
        invalid_indices = (
            [True],
            [1.5],
            [np.nan],
            [np.inf],
            ["1"],
            [b"1"],
            [1 + 0j],
            [np.array([1])],
        )

        for session_indices in invalid_indices:
            with self.subTest(session_indices=session_indices):
                with self.assertRaisesRegex(
                    ValueError,
                    "session indices must be integers",
                ):
                    complete_track_set([[0, 1, 2]], session_indices=session_indices)

    def test_track_pair_set_rejects_noninteger_session_pair_indices(self):
        invalid_pairs = (
            [(True, 1)],
            [(0, 1.5)],
            [(0, np.nan)],
            [(0, np.inf)],
            [("0", 1)],
            [(b"0", 1)],
            [(0, 1 + 0j)],
            [(np.array([0]), 1)],
        )

        for session_pairs in invalid_pairs:
            with self.subTest(session_pairs=session_pairs):
                with self.assertRaisesRegex(
                    ValueError,
                    "session indices must be integers",
                ):
                    track_pair_set([[0, 1, 2]], session_pairs=session_pairs)

    def test_session_selectors_accept_integer_like_numeric_scalars(self):
        self.assertEqual(
            complete_track_set([[0, 1, 2]], session_indices=[np.array(0), 2.0]),
            {(0, 2)},
        )
        self.assertEqual(
            track_pair_set([[0, 1, 2]], session_pairs=[(np.array(0), 2.0)]),
            {(0, 2, 0, 2)},
        )


if __name__ == "__main__":
    unittest.main()
