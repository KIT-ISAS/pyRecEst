def patch_pytorch_copy_facade():
    try:
        import pyrecest._backend.pytorch as pytorch_backend
    except ModuleNotFoundError:
        return

    original_copy = getattr(pytorch_backend, "copy", None)
    if original_copy is None:
        return

    def copy_arraylike(x):
        if pytorch_backend.is_array(x):
            return original_copy(x)
        return pytorch_backend.array(x)

    pytorch_backend.copy = copy_arraylike
