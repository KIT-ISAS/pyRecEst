"""Systematic conversion between distribution representations.

The conversion layer is intentionally target-centric: a concrete target
representation can implement ``from_distribution(...)`` and the generic
``convert_distribution`` gateway will route to it. This keeps manifold-specific
logic close to the representation that owns it while giving users a single,
discoverable API.
"""

from __future__ import annotations

import copy
import inspect
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


class ConversionError(ValueError):
    """Raised when a distribution representation conversion is unavailable."""


@dataclass
class ConversionResult:
    """Distribution conversion result and metadata.

    Attributes
    ----------
    distribution
        Converted distribution instance.
    source_type
        Concrete source distribution type.
    target_type
        Requested target distribution type.
    method
        Human-readable conversion method used by the gateway.
    exact
        Whether the conversion is mathematically exact as a distribution
        representation. Moment matching, sampling, and grid approximation
        should be reported as approximate.
    parameters
        Conversion parameters that were passed to the conversion routine.
    """

    distribution: Any
    source_type: type
    target_type: type
    method: str
    exact: bool
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass
class _RegisteredConversion:
    source_type: type
    target_type: type
    converter: Callable[..., Any]
    exact: bool
    method: str


_CONVERSION_REGISTRY: list[_RegisteredConversion] = []


def register_conversion(
    source_type: type,
    target_type: type,
    converter: Callable[..., Any] | None = None,
    *,
    exact: bool = False,
    method: str | None = None,
):
    """Register a custom conversion between two representation classes.

    Parameters
    ----------
    source_type
        Source type accepted by the converter. Subclasses are accepted.
    target_type
        Target type produced by the converter. Subclasses of this target may
        also use the registered conversion.
    converter
        Callable with signature ``converter(source, **kwargs)``. When omitted,
        this function acts as a decorator.
    exact
        Whether the conversion is exact.
    method
        Optional human-readable method name used in :class:`ConversionResult`.

    Returns
    -------
    Callable
        The original converter, enabling decorator use.
    """

    def _decorator(func: Callable[..., Any]):
        _CONVERSION_REGISTRY.insert(
            0,
            _RegisteredConversion(
                source_type=source_type,
                target_type=target_type,
                converter=func,
                exact=exact,
                method=method or _callable_name(func),
            ),
        )
        return func

    if converter is not None:
        return _decorator(converter)
    return _decorator


def convert_distribution(
    distribution,
    target_type: type,
    /,
    *,
    return_info: bool = False,
    copy_if_same: bool = False,
    **kwargs,
):
    """Convert or approximate ``distribution`` as ``target_type``.

    The gateway supports three cases, in order:

    1. identity conversion when ``distribution`` already is an instance of
       ``target_type``;
    2. explicitly registered conversions;
    3. target classes exposing ``from_distribution(distribution, ...)``.

    Parameters
    ----------
    distribution
        Source distribution instance.
    target_type
        Concrete target representation class, such as
        ``LinearDiracDistribution``, ``CircularGridDistribution``, or
        ``GaussianDistribution``.
    return_info
        If false, return the converted distribution directly. If true, return a
        :class:`ConversionResult`.
    copy_if_same
        If true, identity conversion returns a deep copy instead of the same
        object.
    **kwargs
        Parameters required by the target conversion.

    Returns
    -------
    object or ConversionResult
        Converted distribution, or metadata wrapper if ``return_info=True``.
    """

    if not isinstance(target_type, type):
        raise TypeError("target_type must be a distribution class.")

    source_type = type(distribution)

    if isinstance(distribution, target_type):
        converted = copy.deepcopy(distribution) if copy_if_same else distribution
        result = ConversionResult(
            distribution=converted,
            source_type=source_type,
            target_type=target_type,
            method="identity",
            exact=True,
            parameters=dict(kwargs),
        )
        return result if return_info else result.distribution

    for entry in _matching_registered_conversions(distribution, target_type):
        converted = _call_conversion_callable(entry.converter, distribution, kwargs)
        result = ConversionResult(
            distribution=converted,
            source_type=source_type,
            target_type=target_type,
            method=entry.method,
            exact=entry.exact,
            parameters=dict(kwargs),
        )
        return result if return_info else result.distribution

    from_distribution = getattr(target_type, "from_distribution", None)
    if callable(from_distribution):
        converted = _call_conversion_callable(
            from_distribution,
            distribution,
            kwargs,
            conversion_name=f"{target_type.__name__}.from_distribution",
        )
        result = ConversionResult(
            distribution=converted,
            source_type=source_type,
            target_type=target_type,
            method=f"{target_type.__name__}.from_distribution",
            exact=False,
            parameters=dict(kwargs),
        )
        return result if return_info else result.distribution

    raise ConversionError(
        f"No conversion registered for {source_type.__name__} -> {target_type.__name__}. "
        f"Implement {target_type.__name__}.from_distribution(...) or register one with register_conversion(...)."
    )


def can_convert(distribution, target_type: type) -> bool:
    """Return whether a conversion route exists.

    This only checks for a route. It does not verify that all conversion
    arguments required by the route have been supplied.
    """

    if not isinstance(target_type, type):
        return False
    if isinstance(distribution, target_type):
        return True
    if any(_matching_registered_conversions(distribution, target_type)):
        return True
    return callable(getattr(target_type, "from_distribution", None))


def registered_conversions() -> tuple[tuple[type, type, str, bool], ...]:
    """Return a compact snapshot of explicitly registered conversions."""

    return tuple(
        (entry.source_type, entry.target_type, entry.method, entry.exact)
        for entry in _CONVERSION_REGISTRY
    )


def _matching_registered_conversions(distribution, target_type: type):
    return (
        entry
        for entry in _CONVERSION_REGISTRY
        if isinstance(distribution, entry.source_type)
        and issubclass(target_type, entry.target_type)
    )


def _callable_name(func: Callable[..., Any]) -> str:
    name = getattr(func, "__name__", None)
    if isinstance(name, str):
        return name
    return repr(func)


def _call_conversion_callable(
    func: Callable[..., Any],
    distribution,
    kwargs: dict[str, Any],
    *,
    conversion_name: str | None = None,
):
    name = conversion_name if conversion_name is not None else _callable_name(func)
    _validate_conversion_arguments(func, kwargs, name)
    try:
        return func(distribution, **kwargs)
    except TypeError as exc:
        raise ConversionError(f"Could not call {name}: {exc}") from exc


def _validate_conversion_arguments(
    func: Callable[..., Any], kwargs: dict[str, Any], conversion_name: str
) -> None:
    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):
        return

    params = list(signature.parameters.values())
    has_var_keyword = any(
        param.kind == inspect.Parameter.VAR_KEYWORD for param in params
    )
    accepted_kwargs: set[str] = set()
    required_kwargs: set[str] = set()
    skipped_source_argument = False

    for param in params:
        if (
            not skipped_source_argument
            and param.kind
            in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
        ):
            skipped_source_argument = True
            continue

        if param.kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        ):
            accepted_kwargs.add(param.name)
            if param.default is inspect.Parameter.empty:
                required_kwargs.add(param.name)

    missing = required_kwargs - set(kwargs)
    if missing:
        raise ConversionError(
            f"{conversion_name} is missing required conversion argument(s): "
            f"{', '.join(sorted(missing))}."
        )

    if has_var_keyword:
        return

    unknown = set(kwargs) - accepted_kwargs
    if unknown:
        accepted = ", ".join(sorted(accepted_kwargs)) or "no keyword arguments"
        raise ConversionError(
            f"{conversion_name} got unsupported conversion argument(s): "
            f"{', '.join(sorted(unknown))}. Accepted arguments: {accepted}."
        )


__all__ = [
    "ConversionError",
    "ConversionResult",
    "can_convert",
    "convert_distribution",
    "register_conversion",
    "registered_conversions",
]
