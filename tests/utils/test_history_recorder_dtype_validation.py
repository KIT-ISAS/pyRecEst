import numpy as np
import numpy.testing as npt
import pyrecest.backend as backend
from pyrecest.utils import HistoryRecorder


def test_padded_history_accepts_non_native_unsigned_integer_values():
    recorder = HistoryRecorder()
    base_dtype = np.dtype(getattr(np, "u" + "int16"))
    non_native_dtype = base_dtype.newbyteorder("S")

    history = recorder.record(
        "state",
        np.array([1, 2], dtype=non_native_dtype),
        pad_with_nan=True,
    )

    npt.assert_allclose(
        np.asarray(backend.to_numpy(history), dtype=float),
        np.array([[1.0], [2.0]]),
    )
