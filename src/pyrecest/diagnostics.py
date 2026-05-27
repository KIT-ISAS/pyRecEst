"""Standard diagnostic containers for filters, trackers, and samplers.

These dataclasses are intentionally lightweight. Algorithms can return them
through optional ``return_diagnostics=True`` code paths without introducing a
dependency on pandas, plotting libraries, or backend-specific array types.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from math import isfinite, log
from typing import Any


def _coerce_weight_values(weights: Any) -> list[float]:
    """Return backend-independent Python floats from an array-like weight vector."""
    try:
        from pyrecest.backend import to_numpy

        weights = to_numpy(weights)
    except Exception:  # pragma: no cover - best-effort fallback for foreign arrays
        pass

    if hasattr(weights, "tolist"):
        weights = weights.tolist()
    if isinstance(weights, int | float):
        return [float(weights)]
    return [float(weight) for weight in weights]


def _coerce_numeric_values(values: Any) -> list[float]:
    """Return a flat list of finite-compatible numeric values."""
    if values is None:
        return []
    try:
        from pyrecest.backend import to_numpy

        values = to_numpy(values)
    except Exception:  # pragma: no cover - best-effort fallback for foreign arrays
        pass

    if hasattr(values, "tolist"):
        values = values.tolist()
    if isinstance(values, bool | int | float):
        return [float(values)]

    out: list[float] = []
    for value in values:
        out.extend(_coerce_numeric_values(value))
    return out


def _coerce_bool_values(values: Any) -> list[bool]:
    """Return a flat list of Boolean values."""
    if values is None:
        return []
    try:
        from pyrecest.backend import to_numpy

        values = to_numpy(values)
    except Exception:  # pragma: no cover - best-effort fallback for foreign arrays
        pass

    if hasattr(values, "tolist"):
        values = values.tolist()
    if isinstance(values, bool | int | float):
        return [bool(values)]

    out: list[bool] = []
    for value in values:
        out.extend(_coerce_bool_values(value))
    return out


def _finite_mean(values: list[float]) -> float | None:
    valid = [value for value in values if isfinite(value)]
    if not valid:
        return None
    return sum(valid) / len(valid)


def _finite_min(values: list[float]) -> float | None:
    valid = [value for value in values if isfinite(value)]
    if not valid:
        return None
    return min(valid)


def _finite_last(values: list[float]) -> float | None:
    for value in reversed(values):
        if isfinite(value):
            return value
    return None


class _DiagnosticsMappingMixin:
    """Small mapping compatibility layer for legacy diagnostics dictionaries."""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)  # type: ignore[arg-type]

    def __contains__(self, key: str) -> bool:
        return key in self.to_dict()

    def __getitem__(self, key: str) -> Any:
        try:
            return self.to_dict()[key]
        except KeyError as exc:
            raise KeyError(key) from exc

    def __setitem__(self, key: str, value: Any) -> None:
        if hasattr(self, key):
            setattr(self, key, value)
            return
        metadata = getattr(self, "metadata")
        metadata[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        return self.to_dict().get(key, default)

    def items(self) -> Any:
        return self.to_dict().items()


@dataclass(slots=True)
class FilterDiagnostics(_DiagnosticsMappingMixin):
    """Diagnostics commonly emitted by single-target Bayesian filters."""

    innovation: Any | None = None
    innovation_covariance: Any | None = None
    residual: Any | None = None
    nis: float | None = None
    nees: float | None = None
    log_likelihood: float | None = None
    covariance_trace: float | None = None
    scale: float | None = None
    action: str | None = None
    accepted: bool | None = None
    robust_update: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, mapping: dict[str, Any]) -> "FilterDiagnostics":
        known = {field_name for field_name in cls.__dataclass_fields__}
        values = {key: value for key, value in mapping.items() if key in known}
        metadata = dict(mapping.get("metadata", {}))
        metadata.update(
            {key: value for key, value in mapping.items() if key not in known}
        )
        values["metadata"] = metadata
        return cls(**values)


@dataclass(slots=True)
class ParticleDiagnostics(_DiagnosticsMappingMixin):
    """Diagnostics commonly emitted by particle filters and samplers."""

    effective_sample_size: float | None = None
    resampled: bool | None = None
    resampling_count: int | None = None
    weight_entropy: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_weights(
        cls,
        weights: Any,
        *,
        resampled: bool | None = None,
        resampling_count: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "ParticleDiagnostics":
        """Build particle diagnostics from normalized or unnormalized weights."""
        values = _coerce_weight_values(weights)
        total = sum(values)
        if total <= 0.0:
            normalized = [0.0 for _ in values]
        else:
            normalized = [max(0.0, value / total) for value in values]
        squared_sum = sum(weight * weight for weight in normalized)
        effective_sample_size = 1.0 / squared_sum if squared_sum > 0.0 else 0.0
        entropy = -sum(weight * log(weight) for weight in normalized if weight > 0.0)
        return cls(
            effective_sample_size=effective_sample_size,
            resampled=resampled,
            resampling_count=resampling_count,
            weight_entropy=entropy,
            metadata={} if metadata is None else dict(metadata),
        )


@dataclass(slots=True)
class ParticleFilterResult(_DiagnosticsMappingMixin):
    """Sequence-level particle-filter estimates and diagnostics.

    This container is intentionally generic: algorithms can store estimates,
    effective-sample-size histories, resampling decisions, spread summaries, and
    optional block-wise ESS values without tying the diagnostics module to a
    specific state space.
    """

    estimates: Any
    effective_sample_size: Any
    resampled: Any
    particle_spread: Any | None = None
    block_effective_sample_size: Any | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def ess_history(self) -> Any:
        return self.effective_sample_size

    @property
    def resampling_flags(self) -> Any:
        return self.resampled

    @property
    def resampling_count(self) -> int:
        return sum(1 for value in _coerce_bool_values(self.resampled) if value)

    @property
    def resampling_fraction(self) -> float:
        values = _coerce_bool_values(self.resampled)
        if not values:
            return 0.0
        return self.resampling_count / len(values)

    def summary_statistics(self) -> dict[str, Any]:
        """Return scalar sequence diagnostics for reports and logs."""
        ess = _coerce_numeric_values(self.effective_sample_size)
        spread = _coerce_numeric_values(self.particle_spread)
        block_ess = _coerce_numeric_values(self.block_effective_sample_size)

        summary = {
            "mean_effective_sample_size": _finite_mean(ess),
            "min_effective_sample_size": _finite_min(ess),
            "final_effective_sample_size": _finite_last(ess),
            "resampling_count": self.resampling_count,
            "resampling_fraction": self.resampling_fraction,
        }
        if spread:
            summary.update(
                {
                    "mean_particle_spread": _finite_mean(spread),
                    "final_particle_spread": _finite_last(spread),
                }
            )
        if block_ess:
            summary.update(
                {
                    "mean_block_effective_sample_size": _finite_mean(block_ess),
                    "min_block_effective_sample_size": _finite_min(block_ess),
                }
            )
        return summary


@dataclass(slots=True)
class AssociationDiagnostics(_DiagnosticsMappingMixin):
    """Diagnostics for association and multi-target tracking steps."""

    cost_matrix: Any | None = None
    gated_measurement_indices: list[int] = field(default_factory=list)
    selected_assignments: list[tuple[int, int]] = field(default_factory=list)
    birth_labels: list[Any] = field(default_factory=list)
    death_labels: list[Any] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


__all__ = [
    "AssociationDiagnostics",
    "FilterDiagnostics",
    "ParticleDiagnostics",
    "ParticleFilterResult",
]
