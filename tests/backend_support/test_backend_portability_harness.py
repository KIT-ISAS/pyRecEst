import pytest

import pyrecest.backend as backend
from tests.support.backend_runner import run_backend_code


@pytest.mark.backend_portable
def test_backend_runner_uses_import_time_backend():
    result = run_backend_code(
        backend.__backend_name__,
        "import pyrecest.backend as backend; print(backend.__backend_name__)",
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == backend.__backend_name__
