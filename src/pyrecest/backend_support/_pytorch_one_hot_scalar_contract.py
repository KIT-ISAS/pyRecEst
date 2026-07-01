"""PyTorch ``one_hot`` scalar-label compatibility hook."""

from __future__ import annotations

from operator import index as _operator_index


def _as_one_hot_labels(torch_module, labels):
    """Return integer labels without treating scalar labels as tensor sizes."""

    if torch_module.is_tensor(labels):
        if (
            labels.dtype == torch_module.bool
            or labels.dtype.is_floating_point
            or labels.dtype.is_complex
        ):
            return labels
        return labels.to(dtype=torch_module.long)

    if isinstance(labels, bool):
        # Preserve PyTorch's previous rejection of boolean scalar labels rather
        # than silently interpreting ``True`` as class index 1.
        return torch_module.LongTensor(labels)

    try:
        scalar_label = _operator_index(labels)
    except TypeError:
        return torch_module.LongTensor(labels)
    return torch_module.as_tensor(scalar_label, dtype=torch_module.long)


def patch_pytorch_one_hot_scalar_contract() -> None:
    """Patch raw/public PyTorch ``one_hot`` to treat scalar labels as values."""

    try:
        import pyrecest._backend.pytorch as pytorch_backend  # pylint: disable=import-outside-toplevel
        import pyrecest.backend as backend  # pylint: disable=import-outside-toplevel
        import torch as torch_module  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError:  # pragma: no cover - PyTorch backend may be unavailable
        return

    original_one_hot = getattr(pytorch_backend, "one_hot", None)
    if original_one_hot is None:
        return
    if getattr(original_one_hot, "_pyrecest_scalar_label_contract", False):
        if getattr(backend, "__backend_name__", None) == "pytorch":
            backend.one_hot = original_one_hot
        return

    def one_hot(labels, num_classes):
        labels = _as_one_hot_labels(torch_module, labels)
        result = torch_module.nn.functional.one_hot(
            labels,
            _operator_index(num_classes),
        )
        return result.to(dtype=torch_module.uint8)

    one_hot.__name__ = getattr(original_one_hot, "__name__", "one_hot")
    one_hot.__doc__ = getattr(original_one_hot, "__doc__", None)
    one_hot._pyrecest_scalar_label_contract = True
    pytorch_backend.one_hot = one_hot
    if getattr(backend, "__backend_name__", None) == "pytorch":
        backend.one_hot = one_hot
