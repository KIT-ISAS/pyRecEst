import pytest
from pyrecest.exceptions import OptionalDependencyError
from pyrecest.optional_dependencies import require_optional_dependency


def test_require_optional_dependency_imports_existing_module():
    assert require_optional_dependency("math", "test").sqrt(4.0) == 2.0


def test_require_optional_dependency_reports_extra_for_missing_module():
    with pytest.raises(OptionalDependencyError) as exc_info:
        require_optional_dependency(
            "definitely_missing_pyrecest_dependency", "plot", feature="plotting"
        )
    assert "pyrecest[plot]" in str(exc_info.value)
