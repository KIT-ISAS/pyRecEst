"""Validated public sequence-association node export."""

from __future__ import annotations

from .sequence_association import SequenceAssociationNode as _SequenceAssociationNode


class SequenceAssociationNode(_SequenceAssociationNode):
    """Sequence-association node with consistent gap-node bookkeeping."""
