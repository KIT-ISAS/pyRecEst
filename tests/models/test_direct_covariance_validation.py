from pyrecest.models import MaskedLinearMeasurementModel


def test_import_masked_model_for_covariance_validation() -> None:
    assert MaskedLinearMeasurementModel is not None
