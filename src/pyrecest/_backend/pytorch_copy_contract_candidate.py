def patch_backend_copy(backend):
    original_copy = backend.copy

    def copy_arraylike(x):
        if backend.is_array(x):
            return original_copy(x)
        return backend.array(x)

    backend.copy = copy_arraylike
