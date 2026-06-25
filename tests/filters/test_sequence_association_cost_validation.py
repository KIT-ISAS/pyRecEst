import pytest

from pyrecest.filters import SequenceAssociationNode


def test_sequence_association_rejects_boolean_unary_cost():
    with pytest.raises(ValueError, match="unary_cost must be a scalar numeric cost"):
        SequenceAssociationNode(frame_index=0, candidate_index=0, unary_cost=True)
