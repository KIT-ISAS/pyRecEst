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
from functools import lru_cache
from importlib import import_module
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


@dataclass
class _ConversionAlias:
    target_type_or_resolver: type | Callable[[Any], type]
    default_kwargs: dict[str, Any] = field(default_factory=dict)
    description: str | None = None


ClassSpec = tuple[str, str]
AliasDomain = tuple[ClassSpec, ClassSpec]

_CONVERSION_REGISTRY: list[_RegisteredConversion] = []
_CONVERSION_ALIAS_REGISTRY: dict[str, _ConversionAlias] = {}

_ABSTRACT_CIRCULAR: ClassSpec = (
    "pyrecest.distributions.circle.abstract_circular_distribution",
    "AbstractCircularDistribution",
)
_ABSTRACT_HYPERHEMISPHERICAL: ClassSpec = (
    "pyrecest.distributions.hypersphere_subset.abstract_hyperhemispherical_distribution",
    "AbstractHyperhemisphericalDistribution",
)
_ABSTRACT_HYPERSPHERICAL: ClassSpec = (
    "pyrecest.distributions.hypersphere_subset.abstract_hyperspherical_distribution",
    "AbstractHypersphericalDistribution",
)
_ABSTRACT_HYPERTOROIDAL: ClassSpec = (
    "pyrecest.distributions.hypertorus.abstract_hypertoroidal_distribution",
    "AbstractHypertoroidalDistribution",
)
_ABSTRACT_LINEAR: ClassSpec = (
    "pyrecest.distributions.nonperiodic.abstract_linear_distribution",
    "AbstractLinearDistribution",
)
_CIRCULAR_DIRAC: ClassSpec = (
    "pyrecest.distributions.circle.circular_dirac_distribution",
    "CircularDiracDistribution",
)
_CIRCULAR_FOURIER: ClassSpec = (
    "pyrecest.distributions.circle.circular_fourier_distribution",
    "CircularFourierDistribution",
)
_CIRCULAR_GRID: ClassSpec = (
    "pyrecest.distributions.circle.circular_grid_distribution",
    "CircularGridDistribution",
)
_GAUSSIAN: ClassSpec = (
    "pyrecest.distributions.nonperiodic.gaussian_distribution",
    "GaussianDistribution",
)
_HYPERHEMISPHERICAL_DIRAC: ClassSpec = (
    "pyrecest.distributions.hypersphere_subset.hyperhemispherical_dirac_distribution",
    "HyperhemisphericalDiracDistribution",
)
_HYPERHEMISPHERICAL_GRID: ClassSpec = (
    "pyrecest.distributions.hypersphere_subset.hyperhemispherical_grid_distribution",
    "HyperhemisphericalGridDistribution",
)
_HYPERSPHERICAL_DIRAC: ClassSpec = (
    "pyrecest.distributions.hypersphere_subset.hyperspherical_dirac_distribution",
    "HypersphericalDiracDistribution",
)
_HYPERSPHERICAL_GRID: ClassSpec = (
    "pyrecest.distributions.hypersphere_subset.hyperspherical_grid_distribution",
    "HypersphericalGridDistribution",
)
_HYPERTOROIDAL_DIRAC: ClassSpec = (
    "pyrecest.distributions.hypertorus.hypertoroidal_dirac_distribution",
    "HypertoroidalDiracDistribution",
)
_HYPERTOROIDAL_FOURIER: ClassSpec = (
    "pyrecest.distributions.hypertorus.hypertoroidal_fourier_distribution",
    "HypertoroidalFourierDistribution",
)
_HYPERTOROIDAL_GRID: ClassSpec = (
    "pyrecest.distributions.hypertorus.hypertoroidal_grid_distribution",
    "HypertoroidalGridDistribution",
)
_LINEAR_DIRAC: ClassSpec = (
    "pyrecest.distributions.nonperiodic.linear_dirac_distribution",
    "LinearDiracDistribution",
)

_DIRECT_BUILTIN_ALIAS_CLASS_SPECS: dict[str, ClassSpec] = {
    "circular_dirac": _CIRCULAR_DIRAC,
    "circular_fourier": _CIRCULAR_FOURIER,
    "circular_grid": _CIRCULAR_GRID,
    "gaussian": _GAUSSIAN,
    "hyperhemispherical_dirac": _HYPERHEMISPHERICAL_DIRAC,
    "hyperhemispherical_grid": _HYPERHEMISPHERICAL_GRID,
    "hyperspherical_dirac": _HYPERSPHERICAL_DIRAC,
    "hyperspherical_grid": _HYPERSPHERICAL_GRID,
    "hypertoroidal_dirac": _HYPERTOROIDAL_DIRAC,
    "hypertoroidal_fourier": _HYPERTOROIDAL_FOURIER,
    "hypertoroidal_grid": _HYPERTOROIDAL_GRID,
    "linear_dirac": _LINEAR_DIRAC,
    "linear_particles": _LINEAR_DIRAC,
    "moment_matched_gaussian": _GAUSSIAN,
    "normal": _GAUSSIAN,
}

_PARTICLE_ALIAS_DOMAINS: tuple[AliasDomain, ...] = (
    (_ABSTRACT_CIRCULAR, _CIRCULAR_DIRAC),
    (_ABSTRACT_HYPERTOROIDAL, _HYPERTOROIDAL_DIRAC),
    (_ABSTRACT_HYPERHEMISPHERICAL, _HYPERHEMISPHERICAL_DIRAC),
    (_ABSTRACT_HYPERSPHERICAL, _HYPERSPHERICAL_DIRAC),
    (_ABSTRACT_LINEAR, _LINEAR_DIRAC),
)
_GRID_ALIAS_DOMAINS: tuple[AliasDomain, ...] = (
    (_ABSTRACT_CIRCULAR, _CIRCULAR_GRID),
    (_ABSTRACT_HYPERTOROIDAL, _HYPERTOROIDAL_GRID),
    (_ABSTRACT_HYPERHEMISPHERICAL, _HYPERHEMISPHERICAL_GRID),
    (_ABSTRACT_HYPERSPHERICAL, _HYPERSPHERICAL_GRID),
)
_FOURIER_ALIAS_DOMAINS: tuple[AliasDomain, ...] = (
    (_ABSTRACT_CIRCULAR, _CIRCULAR_FOURIER),
    (_ABSTRACT_HYPERTOROIDAL, _HYPERTOROIDAL_FOURIER),
)
_DOMAIN_BUILTIN_ALIAS_CLASS_SPECS: dict[str, tuple[AliasDomain, ...]] = {
    "dirac": _PARTICLE_ALIAS_DOMAINS,
    "fourier": _FOURIER_ALIAS_DOMAINS,
    "grid": _GRID_ALIAS_DOMAINS,
    "particles": _PARTICLE_ALIAS_DOMAINS,
    "samples": _PARTICLE_ALIAS_DOMAINS,
}


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


def register_conversion_alias(
    alias: str,
    target_type_or_resolver: type | Callable[[Any], type],
    *,
    default_kwargs: dict[str, Any] | None = None,
    description: str | None = None,
) -> None:
    """Register a string alias for a target representation.

    Parameters
    ----------
    alias
        Alias accepted by :func:`convert_distribution`, for example
        ``"particles"`` or ``"gaussian"``. Aliases are case-insensitive;
        hyphens and spaces are normalized to underscores.
    target_type_or_resolver
        Concrete target class or callable ``resolver(source_distribution)``
        returning a concrete target class. Resolvers make aliases domain-aware.
    default_kwargs
        Optional conversion arguments supplied by default. User-provided
        keyword arguments override these defaults.
    description
        Optional human-readable description used by
        :func:`registered_conversion_aliases`.
    """

    _CONVERSION_ALIAS_REGISTRY[_normalize_alias(alias)] = _ConversionAlias(
        target_type_or_resolver=target_type_or_resolver,
        default_kwargs=dict(default_kwargs or {}),
        description=description,
    )


def registered_conversion_aliases() -> tuple[tuple[str, str | None], ...]:
    """Return registered custom conversion aliases."""

    return tuple(
        (alias, entry.description) for alias, entry in _CONVERSION_ALIAS_REGISTRY.items()
    )


def convert_distribution(
    distribution,
    target_type,
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
        ``GaussianDistribution``; or a conversion alias such as ``"particles"``,
        ``"gaussian"``, ``"grid"``, or ``"fourier"``.
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

    target_type, alias_kwargs, target_alias = _resolve_target_type(
        distribution, target_type
    )
    conversion_kwargs = {**alias_kwargs, **kwargs}

    if not isinstance(target_type, type):
        raise TypeError("target_type must be a distribution class or conversion alias.")

    source_type = type(distribution)
    method_prefix = f"alias:{target_alias}->" if target_alias is not None else ""

    if isinstance(distribution, target_type):
        converted = copy.deepcopy(distribution) if copy_if_same else distribution
        result = ConversionResult(
            distribution=converted,
            source_type=source_type,
            target_type=target_type,
            method=f"{method_prefix}identity",
            exact=True,
            parameters=dict(conversion_kwargs),
        )
        return result if return_info else result.distribution

    for entry in _matching_registered_conversions(distribution, target_type):
        converted = _call_conversion_callable(
            entry.converter, distribution, conversion_kwargs
        )
        result = ConversionResult(
            distribution=converted,
            source_type=source_type,
            target_type=target_type,
            method=f"{method_prefix}{entry.method}",
            exact=entry.exact,
            parameters=dict(conversion_kwargs),
        )
        return result if return_info else result.distribution

    from_distribution = getattr(target_type, "from_distribution", None)
    if callable(from_distribution):
        converted = _call_conversion_callable(
            from_distribution,
            distribution,
            conversion_kwargs,
            conversion_name=f"{target_type.__name__}.from_distribution",
        )
        result = ConversionResult(
            distribution=converted,
            source_type=source_type,
            target_type=target_type,
            method=f"{method_prefix}{target_type.__name__}.from_distribution",
            exact=False,
            parameters=dict(conversion_kwargs),
        )
        return result if return_info else result.distribution

    raise ConversionError(
        f"No conversion registered for {source_type.__name__} -> {target_type.__name__}. "
        f"Implement {target_type.__name__}.from_distribution(...) or register one with register_conversion(...)."
    )


def can_convert(distribution, target_type) -> bool:
    """Return whether a conversion route exists.

    This only checks for a route. It does not verify that all conversion
    arguments required by the route have been supplied.
    """

    try:
        target_type, _, _ = _resolve_target_type(distribution, target_type)
    except ConversionError:
        return False

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


def _normalize_alias(alias: str) -> str:
    return alias.strip().lower().replace("-", "_").replace(" ", "_")


def _resolve_target_type(distribution, target_type):
    if isinstance(target_type, str):
        return _resolve_conversion_alias(distribution, target_type)
    return target_type, {}, None


def _resolve_conversion_alias(distribution, alias: str):
    alias_normalized = _normalize_alias(alias)
    custom_alias = _CONVERSION_ALIAS_REGISTRY.get(alias_normalized)
    if custom_alias is not None:
        target_type = _resolve_alias_entry(distribution, custom_alias)
        return target_type, custom_alias.default_kwargs, alias_normalized

    target_type = _resolve_builtin_alias(distribution, alias_normalized)
    return target_type, {}, alias_normalized


def _resolve_alias_entry(distribution, alias_entry: _ConversionAlias):
    target_type_or_resolver = alias_entry.target_type_or_resolver
    if isinstance(target_type_or_resolver, type):
        return target_type_or_resolver
    target_type = target_type_or_resolver(distribution)
    if not isinstance(target_type, type):
        raise ConversionError(
            "Conversion alias resolver must return a distribution class."
        )
    return target_type


# The built-in resolver intentionally maps a compact user-facing alias vocabulary
# across several distribution domains while keeping imports lazy to avoid cycles.
# pylint: disable-next=too-many-locals,too-many-return-statements,too-many-branches
def _resolve_builtin_alias(distribution, alias: str):
    target_type = _resolve_direct_builtin_alias(alias)
    if target_type is not None:
        return target_type

    domain_aliases = _DOMAIN_BUILTIN_ALIAS_CLASS_SPECS.get(alias)
    if domain_aliases is not None:
        target_type = _resolve_domain_builtin_alias(distribution, domain_aliases)
        if target_type is not None:
            return target_type

    raise ConversionError(
        f"Unknown conversion alias '{alias}'. Use a concrete target class or register the alias with register_conversion_alias(...)."
    )


def _resolve_direct_builtin_alias(alias: str) -> type | None:
    class_spec = _DIRECT_BUILTIN_ALIAS_CLASS_SPECS.get(alias)
    if class_spec is None:
        return None
    return _resolve_class(class_spec)


def _resolve_domain_builtin_alias(
    distribution, alias_domains: tuple[AliasDomain, ...]
) -> type | None:
    for source_class_spec, target_class_spec in alias_domains:
        if isinstance(distribution, _resolve_class(source_class_spec)):
            return _resolve_class(target_class_spec)
    return None


def _resolve_class(class_spec: ClassSpec) -> type:
    module_name, class_name = class_spec
    return _import_distribution_class(module_name, class_name)


@lru_cache(maxsize=None)
def _import_distribution_class(module_name: str, class_name: str) -> type:
    module = import_module(module_name)
    distribution_class = getattr(module, class_name)
    if not isinstance(distribution_class, type):
        raise ConversionError(f"{module_name}.{class_name} is not a class.")
    return distribution_class


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
    "register_conversion_alias",
    "registered_conversion_aliases",
    "registered_conversions",
]
