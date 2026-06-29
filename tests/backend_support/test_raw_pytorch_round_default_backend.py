import importlib.util

import pytest


@pytest.mark.backend_portable
def test_raw_pytorch_round_accepts_array_like_inputs_under_numpy_backend(monkeypatch):
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is not installed")

    monkeypatch.setenv("PYRECEST_BACKEND", "numpy")

    import pyrecest  # noqa: F401
    import pyrecest._backend.pytorch as raw_backend

    def assert_close_list(actual, expected):
        actual = [float(value) for value in actual]
        assert len(actual) == len(expected)
        assert all(abs(one_actual - one_expected) < 1e-6 for one_actual, one_expected in zip(actual, expected))

    result = raw_backend.round([1.24, 2.76], decimals=1)
    assert_close_list(raw_backend.to_numpy(result).tolist(), [1.2, 2.8])

    out = raw_backend.array([0.0, 0.0])
    returned = raw_backend.round([1.24, 2.76], decimals=1, out=out)
    assert returned is out
    assert_close_list(raw_backend.to_numpy(out).tolist(), [1.2, 2.8])

    with pytest.raises(TypeError):
        raw_backend.round([1.0], decimals=1.5)
