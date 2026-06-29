import pytest

from pyrecest.filters import SequenceAssociationNode


def test_missed_detection_node_rejects_real_candidate_index():
    with pytest.raises(ValueError, match="missed-detection nodes must use candidate_index=None"):
        SequenceAssociationNode(frame_index=0, candidate_index=1, is_missed_detection=True)
