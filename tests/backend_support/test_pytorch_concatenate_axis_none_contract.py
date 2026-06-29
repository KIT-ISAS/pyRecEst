import importlib.util

import pytest


@pytest.mark.backend_portable
def test_raw_pytorch_concatenate_axis_none_matches_numpy_contract():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is not installed")

    import pyrecest._backend.pytorch as pytorch_backend

    result = pytorch_backend.concatenate(([[1, 2]], [[3, 4]]), axis=None)
    assert pytorch_backend.to_numpy(result).tolist() == [1, 2, 3, 4]

    out = pytorch_backend.empty((4,), dtype=result.dtype)
    returned = pytorch_backend.concatenate(([[1, 2]], [[3, 4]]), axis=None, out=out)
    assert returned is out
    assert pytorch_backend.to_numpy(out).tolist() == [1, 2, 3, 4]

    axis_result = pytorch_backend.concatenate(([[1], [2]], [[3], [4]]), axis=1)
    assert pytorch_backend.to_numpy(axis_result).tolist() == [[1, 3], [2, 4]]
