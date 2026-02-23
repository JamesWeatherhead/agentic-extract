"""Verify project scaffolding is correct."""
import importlib


def test_package_importable():
    """The agentic_extract package must be importable."""
    mod = importlib.import_module("agentic_extract")
    assert mod is not None


def test_version_string_exists():
    """The package must expose a __version__ string."""
    from agentic_extract import __version__
    assert isinstance(__version__, str)
    assert len(__version__) > 0
    parts = __version__.split(".")
    assert len(parts) >= 3, f"Version {__version__} is not semver"


def test_py_typed_marker_exists():
    """py.typed marker must exist for PEP 561 compliance."""
    import pathlib
    import agentic_extract
    pkg_dir = pathlib.Path(agentic_extract.__file__).parent
    assert (pkg_dir / "py.typed").exists()
