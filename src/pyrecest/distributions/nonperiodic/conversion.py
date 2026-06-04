"""Compatibility import path for nonperiodic distribution conversions."""

from pyrecest.distributions.conversion import (
    ConversionError,
    ConversionResult,
    can_convert,
    convert_distribution,
    register_conversion,
    register_conversion_alias,
    registered_conversion_aliases,
    registered_conversions,
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
