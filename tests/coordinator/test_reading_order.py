# tests/coordinator/test_reading_order.py
"""Tests for reading order determination via Surya."""
import json
import pathlib
from unittest.mock import MagicMock, patch

import pytest

from agentic_extract.coordinator.layout import LayoutRegion
from agentic_extract.coordinator.reading_order import (
    SuryaReadingOrder,
    determine_reading_order,
    fallback_reading_order,
)
from agentic_extract.models import BoundingBox, RegionType
from agentic_extract.tools.docker_runner import ToolOutput


def _make_region(
    rid: str, x: float, y: float, w: float = 0.8, h: float = 0.1, page: int = 1,
) -> LayoutRegion:
    return LayoutRegion(
        region_id=rid,
        region_type=RegionType.TEXT,
        bbox=BoundingBox(x=x, y=y, w=w, h=h),
        confidence=0.95,
        page=page,
    )


def test_fallback_reading_order_sorts_top_to_bottom():
    """Without Surya, regions should sort by page then y-coordinate."""
    regions = [
        _make_region("r3", x=0.1, y=0.7, page=1),
        _make_region("r1", x=0.1, y=0.1, page=1),
        _make_region("r2", x=0.1, y=0.4, page=1),
    ]
    order = fallback_reading_order(regions)
    assert order == ["r1", "r2", "r3"]


def test_fallback_reading_order_multipage():
    """Multi-page regions: page 1 regions come before page 2."""
    regions = [
        _make_region("r2_p2", x=0.1, y=0.1, page=2),
        _make_region("r1_p1", x=0.1, y=0.5, page=1),
        _make_region("r0_p1", x=0.1, y=0.1, page=1),
    ]
    order = fallback_reading_order(regions)
    assert order == ["r0_p1", "r1_p1", "r2_p2"]


def test_fallback_reading_order_two_column():
    """Two-column layout: left column before right column at same y."""
    regions = [
        _make_region("right", x=0.55, y=0.1, w=0.4),
        _make_region("left", x=0.05, y=0.1, w=0.4),
    ]
    order = fallback_reading_order(regions)
    assert order == ["left", "right"]


@patch.object(SuryaReadingOrder, "_docker_tool")
def test_surya_reading_order(mock_tool: MagicMock, tmp_path: pathlib.Path):
    from PIL import Image
    img = Image.new("RGB", (100, 100), "white")
    img_path = tmp_path / "page.png"
    img.save(img_path)

    surya_output = json.dumps({"reading_order": ["r2", "r1", "r3"]})
    mock_tool.return_value.run.return_value = ToolOutput(
        stdout=surya_output, stderr="", exit_code=0, duration_ms=300,
    )

    tool = SuryaReadingOrder()
    order = tool.get_reading_order(
        img_path,
        region_ids=["r1", "r2", "r3"],
    )
    assert order == ["r2", "r1", "r3"]


@patch.object(SuryaReadingOrder, "_docker_tool")
def test_surya_fallback_on_failure(mock_tool: MagicMock, tmp_path: pathlib.Path):
    """If Surya fails, determine_reading_order should fall back gracefully."""
    from PIL import Image
    img = Image.new("RGB", (100, 100), "white")
    img_path = tmp_path / "page.png"
    img.save(img_path)

    mock_tool.return_value.run.return_value = ToolOutput(
        stdout="", stderr="Surya crashed", exit_code=1, duration_ms=50,
    )

    regions = [
        _make_region("r1", x=0.1, y=0.1),
        _make_region("r2", x=0.1, y=0.5),
    ]
    order = determine_reading_order(img_path, regions)
    # Should fall back to geometric ordering
    assert order == ["r1", "r2"]
