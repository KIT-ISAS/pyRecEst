from pyrecest.filters import SequenceAssociationNode


def test_keyword_is_present_for_import_contract():
    assert "is_missed_detection" in SequenceAssociationNode.__dataclass_fields__
