# tests/coordinator/test_quality.py
"""Tests for document quality assessment."""
import pathlib

import pytest
from PIL import Image, ImageDraw

from agentic_extract.coordinator.ingestion import PageImage
from agentic_extract.coordinator.quality import (
    QualityAssessment,
    assess_quality,
)


def _make_page_image(
    tmp_path: pathlib.Path,
    width: int = 2550,
    height: int = 3300,
    dpi: int = 300,
    color: str = "white",
    name: str = "page.png",
) -> PageImage:
    """Helper to create a PageImage with an actual image file."""
    img = Image.new("RGB", (width, height), color)
    path = tmp_path / name
    img.save(path, dpi=(dpi, dpi))
    return PageImage(
        page_number=1, image_path=path, width=width, height=height, dpi=dpi,
    )


def test_quality_assessment_dataclass():
    qa = QualityAssessment(
        dpi=300,
        skew_angle=0.5,
        degradation_score=0.2,
        needs_enhancement=False,
    )
    assert qa.dpi == 300
    assert qa.needs_enhancement is False


def test_quality_high_quality_scan(tmp_path: pathlib.Path):
    """A clean, high-DPI white image should score well."""
    page = _make_page_image(tmp_path, dpi=300)
    qa = assess_quality(page)
    assert qa.dpi == 300
    assert qa.degradation_score < 0.5
    assert qa.needs_enhancement is False


def test_quality_low_dpi_flagged(tmp_path: pathlib.Path):
    """Low DPI should increase degradation score."""
    page = _make_page_image(tmp_path, width=612, height=792, dpi=72)
    qa = assess_quality(page)
    assert qa.dpi == 72
    # Low DPI contributes to degradation
    assert qa.degradation_score > 0.0


def test_quality_noisy_image(tmp_path: pathlib.Path):
    """A noisy/dark image should have higher degradation."""
    import random
    random.seed(42)
    img = Image.new("RGB", (500, 500), "white")
    pixels = img.load()
    # Add noise: random dark pixels
    for x in range(500):
        for y in range(500):
            if random.random() < 0.3:
                pixels[x, y] = (50, 50, 50)
    path = tmp_path / "noisy.png"
    img.save(path, dpi=(150, 150))
    page = PageImage(page_number=1, image_path=path, width=500, height=500, dpi=150)

    qa = assess_quality(page)
    assert qa.degradation_score > 0.2


def test_quality_needs_enhancement_threshold(tmp_path: pathlib.Path):
    """needs_enhancement should be True when degradation_score > 0.5."""
    # Create a very degraded image: mostly dark
    img = Image.new("RGB", (200, 200), (40, 40, 40))
    path = tmp_path / "dark.png"
    img.save(path, dpi=(72, 72))
    page = PageImage(page_number=1, image_path=path, width=200, height=200, dpi=72)

    qa = assess_quality(page)
    assert qa.degradation_score > 0.5
    assert qa.needs_enhancement is True


def test_quality_skew_angle_is_float(tmp_path: pathlib.Path):
    """Skew angle should always be a float."""
    page = _make_page_image(tmp_path)
    qa = assess_quality(page)
    assert isinstance(qa.skew_angle, float)
