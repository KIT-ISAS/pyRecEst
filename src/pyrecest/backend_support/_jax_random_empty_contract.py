"""JAX random backend compatibility helpers."""

from __future__ import annotations


def patch_jax_randint_empty_size_contract() -> None:
    """Make JAX randint match NumPy for empty invalid-bound draws."""
    try:
        import jax.numpy as jnp  # pylint: disable=import-outside-toplevel
        import pyrecest._backend.jax.random as raw_jax_random  # pylint: disable=import-outside-toplevel
        import pyrecest.backend as backend  # pylint: disable=import-outside-toplevel
    except ModuleNotFoundError:  # pragma: no cover - JAX backend may be unavailable
        return

    original_randint = raw_jax_random.randint
    if getattr(original_randint, "_pyrecest_empty_size_contract", False):
        if getattr(backend, "__backend_name__", None) == "jax":
            backend_random = getattr(backend, "random", None)
            if backend_random is not None:
                backend_random.randint = original_randint
        return

    def _empty_invalid_bound_result(low, high, size, args, kwargs):
        if args:
            return None
        if high is None:
            if low is None:
                return None
            low, high = 0, low

        low = raw_jax_random._validate_randint_bound(low, "low")  # pylint: disable=protected-access
        high = raw_jax_random._validate_randint_bound(high, "high")  # pylint: disable=protected-access
        try:
            low, high = jnp.broadcast_arrays(low, high)
        except ValueError as exc:
            raise ValueError("low and high could not be broadcast together") from exc

        shape = raw_jax_random._bounded_sampler_shape(size, low, high)  # pylint: disable=protected-access
        if not raw_jax_random._shape_has_no_samples(shape):  # pylint: disable=protected-access
            return None
        state, has_state, remaining_kwargs = raw_jax_random._get_state(**kwargs)  # pylint: disable=protected-access
        state, result = raw_jax_random._randint(  # pylint: disable=protected-access
            state,
            shape,
            low,
            high,
            **remaining_kwargs,
        )
        return raw_jax_random.set_state_return(has_state, state, result)

    def randint(low=None, high=None, size=None, *args, **kwargs):
        try:
            return original_randint(low, high, size, *args, **kwargs)
        except ValueError as exc:
            if "high must be greater than low" not in str(exc):
                raise
            result = _empty_invalid_bound_result(low, high, size, args, kwargs)
            if result is None:
                raise
            return result

    randint.__name__ = getattr(original_randint, "__name__", "randint")
    randint.__doc__ = getattr(original_randint, "__doc__", None)
    randint._pyrecest_empty_size_contract = True
    raw_jax_random.randint = randint
    if getattr(backend, "__backend_name__", None) == "jax":
        backend_random = getattr(backend, "random", None)
        if backend_random is not None:
            backend_random.randint = randint
