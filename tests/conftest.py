"""Shared test fixtures for agentic_extract."""
import pathlib
import tempfile

import pytest


@pytest.fixture
def tmp_dir():
    """Provide a temporary directory that is cleaned up after the test."""
    with tempfile.TemporaryDirectory(prefix="ae_test_") as d:
        yield pathlib.Path(d)


@pytest.fixture
def sample_image_path(tmp_dir: pathlib.Path) -> pathlib.Path:
    """Create a minimal valid PNG file for testing."""
    from PIL import Image
    img = Image.new("RGB", (100, 100), color="white")
    path = tmp_dir / "sample.png"
    img.save(path)
    return path
