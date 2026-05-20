from importlib.metadata import PackageNotFoundError, version

import pyrecest._backend  # noqa
from pyrecest.backend import copy  # noqa: F401

try:
    __version__ = version("pyrecest")
except PackageNotFoundError:  # pragma: no cover - source tree without install metadata
    __version__ = "0+unknown"
