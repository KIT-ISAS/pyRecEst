"""Additive-noise model objects.

These classes intentionally stay small. They package a deterministic model
function together with an additive noise description so filters can reuse the
same model object instead of receiving ad hoc functions and covariance matrices.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


def _covariance_from_noise(noise):
    """Return a covariance matrix from a noise object or covariance-like value."""
    covariance = getattr(noise, "covariance", None)
    if covariance is not None:
        return covariance() if callable(covariance) else covariance

    direct_covariance = getattr(noise, "C", None)
    if direct_covariance is not None:
        return direct_covariance

    return noise


@dataclass(frozen=True)
class AdditiveNoiseTransitionModel:
    """Deterministic transition function with additive process noise.

    Parameters
    ----------
    transition_function:
        Callable with signature ``transition_function(x, dt, **kwargs)``. This
        matches the current Euclidean UKF nonlinear prediction API.
    noise_distribution:
        Additive process-noise distribution or covariance-like object. If the
        object has ``covariance()`` or ``C``, that covariance is used by filters
        requiring a covariance matrix.
    dt:
        Optional default time step for filters whose predict methods accept
        ``dt``. A method-level ``dt`` argument overrides this value.
    function_args:
        Optional keyword arguments applied to every transition-function call.
        Method-level keyword arguments override entries in this mapping.
    """

    transition_function: Callable[..., Any]
    noise_distribution: Any
    dt: float | None = None
    function_args: dict[str, Any] | None = None

    @property
    def noise_covariance(self):
        """Covariance matrix associated with the additive process noise."""
        return _covariance_from_noise(self.noise_distribution)

    def evaluate(self, state, dt=None, **kwargs):
        """Evaluate the deterministic transition part.

        Parameters
        ----------
        state:
            State vector with shape ``(state_dim,)``.
        dt:
            Time step passed by the consuming filter. If ``None``, this model's
            default ``dt`` is used.
        kwargs:
            Extra transition-function keyword arguments. These override
            ``function_args`` for this call.
        """
        call_args = dict(self.function_args or {})
        call_args.update(kwargs)
        effective_dt = self.dt if dt is None else dt
        return self.transition_function(state, effective_dt, **call_args)


@dataclass(frozen=True)
class AdditiveNoiseMeasurementModel:
    """Deterministic measurement function with additive measurement noise.

    Parameters
    ----------
    measurement_function:
        Callable with signature ``measurement_function(x, **kwargs)``. This
        matches the current Euclidean UKF nonlinear update API.
    noise_distribution:
        Additive measurement-noise distribution or covariance-like object. If
        the object has ``covariance()`` or ``C``, that covariance is used by
        filters requiring a covariance matrix.
    function_args:
        Optional keyword arguments applied to every measurement-function call.
        Method-level keyword arguments override entries in this mapping.
    """

    measurement_function: Callable[..., Any]
    noise_distribution: Any
    function_args: dict[str, Any] | None = None

    @property
    def noise_covariance(self):
        """Covariance matrix associated with the additive measurement noise."""
        return _covariance_from_noise(self.noise_distribution)

    def evaluate(self, state, **kwargs):
        """Evaluate the deterministic measurement part.

        Parameters
        ----------
        state:
            State vector with shape ``(state_dim,)``.
        kwargs:
            Extra measurement-function keyword arguments. These override
            ``function_args`` for this call.
        """
        call_args = dict(self.function_args or {})
        call_args.update(kwargs)
        return self.measurement_function(state, **call_args)
