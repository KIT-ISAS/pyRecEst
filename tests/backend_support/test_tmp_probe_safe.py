# simple
import numpy as _np
import torch as _torch

def test_tmp_probe_safe():
    assert _torch.as_tensor(_np.asarray([1])).tolist() == [1]
