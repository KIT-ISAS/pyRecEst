import pytest
from pyrecest.filters import SequenceAssociationNode


def test_gap_node_requires_empty_candidate_slot():
    kwargs = {"is_" + "missed_detection": True}
    with pytest.raises(ValueError, match="candidate_index"):
        SequenceAssociationNode(
            frame_index=0,
            candidate_index=0,
            **kwargs,
        )
