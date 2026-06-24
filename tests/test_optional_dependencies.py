from unittest import mock

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


@pytest.mark.parametrize(
    ("package", "extra"),
    [
        (None, "plot"),
        ("", "plot"),
        ("   ", "plot"),
        ("math", None),
        ("math", ""),
        ("math", b"plot"),
    ],
)
def test_require_optional_dependency_rejects_invalid_names(package, extra):
    with pytest.raises(ValueError, match="must be a non-empty string"):
        require_optional_dependency(package, extra)


def test_require_optional_dependency_reports_missing_parent_for_submodule():
    import_error = ModuleNotFoundError(
        "No module named 'missing_parent'",
        name="missing_parent",
    )

    with mock.patch("importlib.import_module", side_effect=import_error):
        with pytest.raises(OptionalDependencyError):
            require_optional_dependency("missing_parent.child", "plot")


def test_require_optional_dependency_preserves_nested_missing_imports():
    import_error = ModuleNotFoundError(
        "No module named 'missing_nested_dependency'",
        name="missing_nested_dependency",
    )

    with mock.patch("importlib.import_module", side_effect=import_error):
        with pytest.raises(ModuleNotFoundError) as exc_info:
            require_optional_dependency("available_optional_package", "plot")

    assert exc_info.value is import_error


def test_require_optional_dependency_preserves_generic_import_errors():
    import_error = ImportError("optional package failed while importing")

    with mock.patch("importlib.import_module", side_effect=import_error):
        with pytest.raises(ImportError) as exc_info:
            require_optional_dependency("available_optional_package", "plot")

    assert exc_info.value is import_error
