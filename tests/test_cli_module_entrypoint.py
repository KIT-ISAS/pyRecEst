from __future__ import annotations

import runpy
from unittest.mock import patch

import pytest


def test_package_module_entrypoint_delegates_to_cli_main() -> None:
    with patch("pyrecest.cli.main", return_value=0) as main:
        with pytest.raises(SystemExit) as exc_info:
            runpy.run_module("pyrecest", run_name="__main__")

    assert exc_info.value.code == 0
    main.assert_called_once_with()
