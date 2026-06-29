def patch_pytorch_copy_tensor_contract():
    import pyrecest._backend.pytorch as pytorch_backend

    original_copy = pytorch_backend.copy

    def copy(x):
        if pytorch_backend.is_array(x):
            return original_copy(x)
        return pytorch_backend.array(x)

    pytorch_backend.copy = copy
