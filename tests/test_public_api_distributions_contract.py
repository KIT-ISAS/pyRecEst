import importlib


def test_distribution_public_all_has_no_duplicates():
    module = importlib.import_module("pyrecest.distributions")
    exports = tuple(getattr(module, "__all__", ()))

    duplicates = sorted({name for name in exports if exports.count(name) > 1})

    assert duplicates == []


def test_distribution_public_all_exports_resolve():
    module = importlib.import_module("pyrecest.distributions")
    exports = tuple(getattr(module, "__all__", ()))

    missing = sorted(name for name in exports if not hasattr(module, name))

    assert missing == []
