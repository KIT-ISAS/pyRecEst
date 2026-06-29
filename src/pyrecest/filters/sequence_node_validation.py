"""Validated public sequence-association node export."""

from __future__ import annotations

from .sequence_association import SequenceAssociationNode as _SequenceAssociationNode


class SequenceAssociationNode(_SequenceAssociationNode):
    """Sequence-association node with consistent gap-node bookkeeping."""

    def __post_init__(self) -> None:
        super().__post_init__()
