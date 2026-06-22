"""Regression tests for multi-session label conversion."""

import unittest

from pyrecest.backend import __backend_name__
from pyrecest.utils import MultiSessionAssignmentResult, tracks_to_session_labels

import pyrecest.utils.multisession_assignment as multisession_assignment_module


class TestMultiSessionAssignmentLabels(unittest.TestCase):
    @unittest.skipIf(
        __backend_name__ == "jax",
        reason="Not supported on this backend",
    )
    def test_duplicate_detection_rejected_when_fill_value_matches_track_label(self):
        tracks = [{0: 0}, {0: 0}]
        converters = (
            ("public", tracks_to_session_labels),
            ("module", multisession_assignment_module.tracks_to_session_labels),
            (
                "result_method",
                lambda track_list, **kwargs: MultiSessionAssignmentResult(
                    tracks=track_list,
                    matched_edges=[],
                    total_cost=0.0,
                ).to_session_labels(**kwargs),
            ),
        )

        for name, converter in converters:
            with self.subTest(converter=name):
                with self.assertRaisesRegex(
                    ValueError,
                    "Each detection can only belong to a single track",
                ):
                    converter(tracks, session_sizes=[1], fill_value=0)


if __name__ == "__main__":
    unittest.main()
