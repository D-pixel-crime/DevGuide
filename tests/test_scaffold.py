"""Scaffold verification tests for package imports."""


def test_packages_import():
    """Verify that all canonical scaffold packages import successfully."""
    import app
    import app.api
    import app.core
    import app.db
    import app.models
    import app.services

    assert app is not None
    assert app.api is not None
    assert app.core is not None
    assert app.db is not None
    assert app.models is not None
    assert app.services is not None
