from pyrecest.stability import get_public_api_status, iter_public_api_status, stability


def test_registered_public_api_status():
    status = get_public_api_status("KalmanFilter")
    assert status is not None
    assert status.level == "stable"
    assert list(iter_public_api_status())


def test_stability_decorator_attaches_metadata():
    @stability("experimental", since="2.3.0", notes="test helper")
    def sample():
        return 1

    assert sample() == 1
    assert sample.__pyrecest_stability__.level == "experimental"
