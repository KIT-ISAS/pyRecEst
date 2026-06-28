from unittest import mock

import pytest

from pyrecest.exceptions import OptionalDependencyError
from pyrecest.optional_dependencies import require_optional_dependency


def test_require_optional_dependency_reports_missing_requested_child_module():
    import_error = ModuleNotFoundError(
        "No module named 'optional_parent.child'",
        name="optional_parent.child",
    )

    with mock.patch("importlib.import_module", side_effect=import_error):
        with pytest.raises(OptionalDependencyError) as exc_info:
            require_optional_dependency("optional_parent", "plot")

    assert "optional_parent" in str(exc_info.value)
    assert "pyrecest[plot]" in str(exc_info.value)
