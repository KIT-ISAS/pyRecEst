"""Compatibility wrapper for finite-state filtering utilities."""

from __future__ import annotations

from pathlib import Path
import runpy

import numpy as np

_module_globals = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "discrete_state.py"),
    run_name=__name__,
)
_original_sparse_gaussian_transition_matrix = _module_globals[
    "sparse_gaussian_transition_matrix"
]


def sparse_gaussian_transition_matrix(
    state_vectors,
    sigma,
    max_step_sigma=4.0,
    *,
    valid_state_mask=None,
):
    states = np.asarray(state_vectors, dtype=float)
    if states.ndim == 1:
        finite_check_states = states[:, None]
    elif states.ndim == 2:
        finite_check_states = states
    else:
        finite_check_states = None
    if finite_check_states is not None and np.any(~np.isfinite(finite_check_states)):
        raise ValueError("state_vectors must contain only finite values")
    return _original_sparse_gaussian_transition_matrix(
        state_vectors,
        sigma,
        max_step_sigma=max_step_sigma,
        valid_state_mask=valid_state_mask,
    )


_module_globals["sparse_gaussian_transition_matrix"] = sparse_gaussian_transition_matrix
for name in _module_globals["__all__"]:
    globals()[name] = _module_globals[name]
__all__ = _module_globals["__all__"]
