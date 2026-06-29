"""Validated public sequence-association node export."""

from __future__ import annotations

from .sequence_association import SequenceAssociationNode as _SequenceAssociationNode


class SequenceAssociationNode(_SequenceAssociationNode):
    """Sequence-association node with consistent gap-node bookkeeping."""

    def __post_init__(self) -> None:
        if self.is_missed_detection and self.candidate_index is not None:
            raise ValueError("candidate_index must be None for explicit gap nodes")
        super().__post_init__()
