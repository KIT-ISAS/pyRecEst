from unittest import mock

import pytest
from pyrecest.optional_dependencies import require_optional_dependency


def test_optional_dependency_preserves_requested_package_child_import_error():
    import_error = ModuleNotFoundError(name="demo_package.child")

    with mock.patch("importlib.import_module", side_effect=import_error):
        with pytest.raises(ModuleNotFoundError) as exc_info:
            require_optional_dependency("demo_package", "plot")

    assert exc_info.value is import_error
