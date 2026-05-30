"""Preferred aliases for the orientation-aware variational-Bayes RM tracker.

The implementation lives in :mod:`pyrecest.filters.vbrm_tracker`.  The ``O``
suffix denotes the Tuncer--Ozkan orientation-aware variant and avoids using the
ambiguous plot label ``VBRM`` when comparing against other random-matrix-family
trackers.
"""

from .vbrm_tracker import VBRMTracker


VBRMOTracker = VBRMTracker
VbrmOTracker = VBRMTracker
OrientationAwareVBRMTracker = VBRMTracker
OrientationAwareVbrmTracker = VBRMTracker
