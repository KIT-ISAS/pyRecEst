"""Compatibility wrapper for track-matrix evaluation helpers.

This wrapper keeps the existing implementation in ``track_evaluation.py`` while
patching track-matrix normalization so boolean-like cells are treated as missing
rather than as observation ids 0 and 1.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
_SOURCE_PATH = Path(__file__).resolve().parent.parent / "track_evaluation.py"
_SPEC = importlib.util.spec_from_file_location(f"{__name__}._source", _SOURCE_PATH)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover - importlib guard
    raise ImportError(f"could not load track evaluation source from {_SOURCE_PATH}")
_SOURCE_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_SOURCE_MODULE)

__all__ = (
    "Observation",
    "TrackLink",
    "complete_track_set",
    "normalize_track_matrix",
    "pairwise_track_set",
    "reference_fragment_counts",
    "score_complete_tracks",
    "score_false_continuations",
    "score_fragmentation",
    "score_pairwise_tracks",
    "score_track_fragmentation",
    "score_track_links",
    "score_track_matrices",
    "summarize_track_errors",
    "summarize_tracks",
    "track_error_ledger",
    "track_lengths",
    "track_pair_set",
)

for _name in __all__:
    globals()[_name] = getattr(_SOURCE_MODULE, _name)
