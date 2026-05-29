from pyrecest.backend import array, diag
from pyrecest.diagnostics import FilterDiagnostics, ParticleDiagnostics
from pyrecest.filters import KalmanFilter


def test_kalman_update_returns_mapping_compatible_filter_diagnostics():
    kalman_filter = KalmanFilter((array([0.0, 1.0]), diag(array([1.0, 1.0]))))
    diagnostics = kalman_filter.update_linear(
        array([1.0]),
        array([[1.0, 0.0]]),
        array([[0.25]]),
        return_diagnostics=True,
    )

    assert isinstance(diagnostics, FilterDiagnostics)
    assert diagnostics["nis"] is not None
    assert diagnostics.get("action") == "updated"


def test_particle_diagnostics_from_weights_computes_health_metrics():
    diagnostics = ParticleDiagnostics.from_weights([0.25, 0.25, 0.5], resampled=False)

    assert diagnostics["effective_sample_size"] > 0.0
    assert diagnostics["weight_entropy"] > 0.0
    assert diagnostics["resampled"] is False


def test_particle_diagnostics_clips_negative_weights():
    diagnostics = ParticleDiagnostics.from_weights([-1.0, 1.0, 3.0])

    assert abs(diagnostics.effective_sample_size - 1.6) < 1e-12
    assert diagnostics.weight_entropy > 0.0


def test_particle_diagnostics_handles_empty_effective_support():
    diagnostics = ParticleDiagnostics.from_weights([-1.0, 0.0])

    assert diagnostics.effective_sample_size == 0.0
    assert diagnostics.weight_entropy == 0.0
