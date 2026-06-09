"""Normal-flow and polarity helpers for DVS/event-camera measurements."""

from __future__ import annotations

import numpy as np

from .active_contour import (
    activity_profile,
    normal_flow_activity,
    signed_normal_flow,
    signed_normal_flow_profile,
)

INFER_POLARITY_CONTRAST_SIGN = "infer"

__all__ = [
    "INFER_POLARITY_CONTRAST_SIGN",
    "activity_profile",
    "event_polarity_sign",
    "infer_polarity_contrast_sign",
    "normal_flow_activity",
    "normalize_polarity_contrast_sign",
    "polarity_consistency_for_signed_flow",
    "polarity_weight_for_signed_flow",
    "polarity_weights_for_signed_flows",
    "signed_normal_flow",
    "signed_normal_flow_profile",
    "signed_scalar_sign",
]


def signed_scalar_sign(value, zero_tolerance=1e-12) -> float:
    """Return the sign of a scalar with a small zero dead-zone."""
    value = float(value)
    if value > float(zero_tolerance):
        return 1.0
    if value < -float(zero_tolerance):
        return -1.0
    return 0.0


def event_polarity_sign(event_polarity) -> float:
    """Return the sign convention used for event polarities.

    Event-camera datasets commonly encode polarities either as ``0/1`` or
    ``-1/+1``. Positive values are treated as ``+1`` and all non-positive
    values as ``-1``.
    """
    return 1.0 if float(event_polarity) > 0.0 else -1.0


def normalize_polarity_contrast_sign(
    polarity_contrast_sign,
    *,
    infer_sentinel=INFER_POLARITY_CONTRAST_SIGN,
):
    """Normalize a polarity-contrast sign specification.

    Returns ``None`` when polarity should be ignored, the infer sentinel when a
    batch-level sign should be inferred, and ``+1`` or ``-1`` for fixed
    contrast assumptions.
    """
    if polarity_contrast_sign is None:
        return None
    if isinstance(polarity_contrast_sign, str):
        normalized = polarity_contrast_sign.lower()
        if normalized != str(infer_sentinel).lower():
            raise ValueError(
                f"polarity_contrast_sign must be {infer_sentinel!r}, None, or non-zero"
            )
        return infer_sentinel
    value = float(polarity_contrast_sign)
    if value == 0.0:
        raise ValueError(
            f"polarity_contrast_sign must be {infer_sentinel!r}, None, or non-zero"
        )
    return 1.0 if value > 0.0 else -1.0


def infer_polarity_contrast_sign(
    signed_normal_flows,
    event_polarities,
    polarity_contrast_sign=INFER_POLARITY_CONTRAST_SIGN,
    *,
    infer_sentinel=INFER_POLARITY_CONTRAST_SIGN,
    zero_tolerance=1e-12,
):
    """Return the fixed or inferred polarity-contrast sign for an event batch."""
    normalized = normalize_polarity_contrast_sign(
        polarity_contrast_sign,
        infer_sentinel=infer_sentinel,
    )
    if normalized is None:
        return None
    if normalized != infer_sentinel:
        return normalized

    flows = np.asarray(signed_normal_flows, dtype=float).reshape((-1,))
    polarities = np.asarray(event_polarities, dtype=float).reshape((-1,))
    if flows.shape != polarities.shape:
        raise ValueError("event_polarities must have one value per signed normal flow")

    score = 0.0
    for signed_flow, event_polarity in zip(flows, polarities, strict=True):
        if signed_scalar_sign(signed_flow, zero_tolerance=zero_tolerance) == 0.0:
            continue
        score += event_polarity_sign(event_polarity) * float(signed_flow)
    if abs(score) <= float(zero_tolerance):
        return 1.0
    return 1.0 if score > 0.0 else -1.0


def polarity_consistency_for_signed_flow(
    signed_normal_flow_value,
    event_polarity,
    polarity_contrast_sign=1.0,
    *,
    infer_sentinel=INFER_POLARITY_CONTRAST_SIGN,
    zero_tolerance=1e-12,
):
    """Return whether event polarity agrees with signed normal flow.

    ``None`` is returned for uninformative zero signed-flow samples or when
    polarity checks are disabled. The infer sentinel must be resolved at batch
    level before this scalar helper is called.
    """
    contrast_sign = normalize_polarity_contrast_sign(
        polarity_contrast_sign,
        infer_sentinel=infer_sentinel,
    )
    if contrast_sign is None:
        return None
    if contrast_sign == infer_sentinel:
        raise ValueError(
            f"polarity_contrast_sign={infer_sentinel!r} must be resolved at batch level"
        )

    flow_sign = signed_scalar_sign(
        signed_normal_flow_value,
        zero_tolerance=zero_tolerance,
    )
    if flow_sign == 0.0:
        return None
    expected_sign = contrast_sign * flow_sign
    observed_sign = event_polarity_sign(event_polarity)
    return bool(observed_sign == expected_sign)


def polarity_weight_for_signed_flow(
    signed_normal_flow_value,
    event_polarity,
    polarity_contrast_sign=1.0,
    *,
    polarity_mismatch_weight=0.25,
    infer_sentinel=INFER_POLARITY_CONTRAST_SIGN,
    zero_tolerance=1e-12,
) -> float:
    """Return a multiplicative reliability weight from polarity consistency."""
    mismatch_weight = float(polarity_mismatch_weight)
    if mismatch_weight < 0.0 or mismatch_weight > 1.0:
        raise ValueError("polarity_mismatch_weight must be in [0, 1]")
    consistency = polarity_consistency_for_signed_flow(
        signed_normal_flow_value,
        event_polarity,
        polarity_contrast_sign=polarity_contrast_sign,
        infer_sentinel=infer_sentinel,
        zero_tolerance=zero_tolerance,
    )
    if consistency is None or consistency:
        return 1.0
    return mismatch_weight


def polarity_weights_for_signed_flows(
    signed_normal_flows,
    event_polarities,
    polarity_contrast_sign=INFER_POLARITY_CONTRAST_SIGN,
    *,
    polarity_mismatch_weight=0.25,
    infer_sentinel=INFER_POLARITY_CONTRAST_SIGN,
    zero_tolerance=1e-12,
) -> np.ndarray:
    """Return polarity reliability weights for a batch of signed-flow samples."""
    flows = np.asarray(signed_normal_flows, dtype=float).reshape((-1,))
    polarities = np.asarray(event_polarities, dtype=float).reshape((-1,))
    resolved_sign = infer_polarity_contrast_sign(
        flows,
        polarities,
        polarity_contrast_sign=polarity_contrast_sign,
        infer_sentinel=infer_sentinel,
        zero_tolerance=zero_tolerance,
    )
    if resolved_sign is None:
        return np.ones(flows.shape, dtype=float)
    return np.asarray(
        [
            polarity_weight_for_signed_flow(
                signed_flow,
                event_polarity,
                polarity_contrast_sign=resolved_sign,
                polarity_mismatch_weight=polarity_mismatch_weight,
                infer_sentinel=infer_sentinel,
                zero_tolerance=zero_tolerance,
            )
            for signed_flow, event_polarity in zip(flows, polarities, strict=True)
        ],
        dtype=float,
    )
