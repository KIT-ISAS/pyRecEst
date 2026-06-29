"""Pytorch based linear algebra backend."""

import numpy as _np
import scipy as _scipy
import torch as _torch

from .._backend_config import pytorch_atol as atol
from ..numpy import linalg as _gsnplinalg
from ._common import array, cast
from ._dtype import (
    get_default_dtype,
    is_complex,
    is_floating,
)

# The public backend facade exposes NumPy-style helpers. Keep array-like
# coercion local to this module instead of re-exporting raw torch.linalg
# functions, because raw torch.linalg rejects Python lists and integer arrays.


def _as_numpy_no_grad(value):
    """Return a CPU NumPy view/copy for SciPy bridge functions."""
    if isinstance(value, _torch.Tensor):
        return value.detach().resolve_conj().resolve_neg().cpu().numpy()
    return _np.asarray(value)


def _torch_as_like(value, like):
    """Convert a NumPy/SciPy result back to the input tensor's device and dtype."""
    if isinstance(like, _torch.Tensor):
        result = _torch.as_tensor(value, device=like.device)
        if result.dtype.is_floating_point and like.dtype.is_floating_point:
            return result.to(dtype=like.dtype)
        if result.dtype.is_complex and like.dtype.is_complex:
            return result.to(dtype=like.dtype)
        return result
    return _torch.from_numpy(_np.asarray(value))


_COMPLEX_DTYPE_FOR_TENSOR_DTYPE = {
    _torch.float32: _np.complex64,
    _torch.float64: _np.complex128,
    _torch.complex64: _np.complex64,
    _torch.complex128: _np.complex128,
}


def _default_linalg_dtype():
    dtype = get_default_dtype()
    if dtype in (_torch.float32, _torch.float64):
        return dtype
    if dtype == _np.dtype("float32"):
        return _torch.float32
    if dtype == _np.dtype("float64"):
        return _torch.float64
    return _torch.float64


def _as_linalg_tensor(value):
    """Convert array-like values to a floating/complex tensor for torch.linalg."""
    tensor = array(value)
    if not is_floating(tensor) and not is_complex(tensor):
        tensor = cast(tensor, dtype=_default_linalg_dtype())
    return tensor


def _common_linalg_dtype(*tensors):
    """Return a common floating/complex dtype for torch.linalg operations."""
    dtype = tensors[0].dtype
    for tensor in tensors[1:]:
        dtype = _torch.promote_types(dtype, tensor.dtype)
    if dtype.is_floating_point or dtype.is_complex:
        return dtype
    return _default_linalg_dtype()


def _preferred_linalg_device(*values):
    """Return the non-CPU tensor device to preserve, falling back to any tensor."""
    non_cpu_device = next(
        (
            value.device
            for value in values
            if _torch.is_tensor(value) and value.device.type != "cpu"
        ),
        None,
    )
    if non_cpu_device is not None:
        return non_cpu_device
    return next((value.device for value in values if _torch.is_tensor(value)), None)


def _out_kwargs(out):
    return {} if out is None else {"out": out}


def cholesky(a, upper=False, out=None):
    """Compute a Cholesky factor after PyRecEst-style array-like promotion."""
    return _torch.linalg.cholesky(_as_linalg_tensor(a), upper=upper, **_out_kwargs(out))


def det(a, out=None):
    """Compute a determinant after PyRecEst-style array-like promotion."""
    return _torch.linalg.det(_as_linalg_tensor(a), **_out_kwargs(out))


def eig(a, out=None):
    """Compute eigenvalues/eigenvectors after array-like input promotion."""
    return _torch.linalg.eig(_as_linalg_tensor(a), **_out_kwargs(out))


def eigh(a, UPLO="L", out=None):
    """Compute Hermitian eigenpairs after array-like input promotion."""
    return _torch.linalg.eigh(_as_linalg_tensor(a), UPLO=UPLO, **_out_kwargs(out))


def eigvalsh(a, UPLO="L", out=None):
    """Compute Hermitian eigenvalues after array-like input promotion."""
    return _torch.linalg.eigvalsh(_as_linalg_tensor(a), UPLO=UPLO, **_out_kwargs(out))


def inv(a, out=None):
    """Invert a matrix after PyRecEst-style array-like promotion."""
    return _torch.linalg.inv(_as_linalg_tensor(a), **_out_kwargs(out))


def expm(a):
    """Compute the matrix exponential after array-like input promotion."""
    return _torch.linalg.matrix_exp(_as_linalg_tensor(a))


def matrix_power(a, n):
    """Raise a matrix to an integer power after array-like input promotion."""
    return _torch.linalg.matrix_power(_as_linalg_tensor(a), n)


def pinv(a, rcond=None, hermitian=False, *, atol=None, rtol=None, out=None):
    """Compute the Moore-Penrose pseudoinverse after array-like input promotion."""
    if rcond is not None:
        if rtol is not None:
            raise TypeError("pinv() got both 'rcond' and 'rtol'")
        rtol = rcond
    return _torch.linalg.pinv(
        _as_linalg_tensor(a),
        atol=atol,
        rtol=rtol,
        hermitian=hermitian,
        **_out_kwargs(out),
    )


def block_diag(*arrs):
    """Build a block diagonal tensor from PyRecEst-style array-like inputs."""
    return _torch.block_diag(*(array(arr) for arr in arrs))


def qr(a, mode="reduced"):
    """Compute QR decomposition with NumPy-compatible mode semantics."""
    a = _as_linalg_tensor(a)

    if mode in ("full", "f"):
        mode = "reduced"
    elif mode == "e":
        mode = "economic"

    if mode in ("reduced", "complete"):
        return _torch.linalg.qr(a, mode=mode)
    if mode == "r":
        return _torch.linalg.qr(a, mode="r").R
    if mode == "raw":
        geqrf, tau = _torch.geqrf(a)
        return geqrf.transpose(-2, -1), tau
    if mode == "economic":
        geqrf, _ = _torch.geqrf(a)
        return geqrf

    raise ValueError(f"Unrecognized mode {mode!r}")


class _Logm(_torch.autograd.Function):
    """Torch autograd function for matrix logarithm."""

    @staticmethod
    def _logm(x):
        mat_log = _gsnplinalg.logm(_as_numpy_no_grad(x))
        return _torch_as_like(mat_log, x)

    @staticmethod
    def forward(ctx, tensor):
        """Apply matrix logarithm to a tensor."""
        ctx.save_for_backward(tensor)
        return _Logm._logm(tensor)

    @staticmethod
    def backward(ctx, grad):
        """Run gradients backward."""
        (tensor,) = ctx.saved_tensors

        tensor_H = tensor.transpose(-2, -1).conj().to(grad.dtype)
        n = tensor.size(-1)
        bshape = tensor.shape[:-2] + (2 * n, 2 * n)
        backward_tensor = _torch.zeros(*bshape, dtype=grad.dtype, device=grad.device)
        backward_tensor[..., :n, :n] = tensor_H
        backward_tensor[..., n:, n:] = tensor_H
        backward_tensor[..., :n, n:] = grad

        return _Logm._logm(backward_tensor).to(tensor.dtype)[..., :n, n:]


logm = _Logm.apply


def sqrtm(x):
    x = _as_linalg_tensor(x)
    x_np = _as_numpy_no_grad(x)
    np_sqrtm = _np.vectorize(_scipy.linalg.sqrtm, signature="(n,m)->(n,m)")(x_np)
    if np_sqrtm.dtype.kind == "c":
        target_complex_dtype = _COMPLEX_DTYPE_FOR_TENSOR_DTYPE.get(x.dtype)
        if target_complex_dtype is not None:
            np_sqrtm = np_sqrtm.astype(target_complex_dtype, copy=False)

    return _torch_as_like(np_sqrtm, x)


def svd(x, full_matrices=True, compute_uv=True):
    x = _as_linalg_tensor(x)
    if compute_uv:
        return _torch.linalg.svd(x, full_matrices=full_matrices)

    return _torch.linalg.svdvals(x)


def norm(x, ord=None, axis=None, keepdims=False):
    x = _as_linalg_tensor(x)
    return _torch.linalg.norm(x, ord=ord, dim=axis, keepdim=keepdims)


def matrix_rank(a, tol=None, hermitian=False, *, rtol=None, atol=None, **kwargs):
    if kwargs:
        unexpected = ", ".join(sorted(kwargs))
        raise TypeError(
            f"matrix_rank() got unexpected keyword argument(s): {unexpected}"
        )
    if tol is not None:
        if atol is not None:
            raise TypeError("matrix_rank() got both 'tol' and 'atol'")
        atol = tol

    a = _as_linalg_tensor(a)
    return _torch.linalg.matrix_rank(a, atol=atol, rtol=rtol, hermitian=hermitian)


def solve(a, b):
    """Solve a linear system with PyRecEst-compatible array-like inputs."""
    device = _preferred_linalg_device(a, b)
    a = _as_linalg_tensor(a)
    b = _as_linalg_tensor(b)
    if device is not None:
        a = a.to(device=device)
        b = b.to(device=device)
    common_dtype = _common_linalg_dtype(a, b)
    a = a.to(dtype=common_dtype)
    b = b.to(dtype=common_dtype)
    return _torch.linalg.solve(a, b)
