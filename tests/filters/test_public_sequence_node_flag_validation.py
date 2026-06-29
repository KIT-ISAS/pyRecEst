import pytest
from pyrecest.filters import SequenceAssociationNode


def test_public_node_rejects_non_boolean_gap_flag_before_consistency_check():
    with pytest.raises(ValueError, match="is_missed_detection must be a bool"):
        SequenceAssociationNode(
            frame_index=0,
            candidate_index=1,
            is_missed_detection="False",
        )
