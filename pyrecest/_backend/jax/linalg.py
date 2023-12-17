"""JAX-based linear algebra backend."""

from jax.numpy.linalg import (  # NOQA
    cholesky,
    det,
    eig,
    eigh,
    eigvalsh,
    inv,
    matrix_rank,
    norm,
    solve,
    svd,
    qr,
)

unsupported_functions = [
    'expm',
    'fractional_matrix_power',
    'is_single_matrix_pd',
    'logm',
    'quadratic_assignment',
    'solve_sylvester',
    'sqrtm',
]
for func_name in unsupported_functions:
    exec(f"{func_name} = lambda *args, **kwargs: NotImplementedError('This function is not supported in this JAX backend.')")

