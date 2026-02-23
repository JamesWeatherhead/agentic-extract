# tests/coordinator/test_layout.py
"""Tests for layout detection via DocLayout-YOLO."""
import json
import pathlib
from unittest.mock import MagicMock, patch

import pytest

from agentic_extract.coordinator.layout import (
    DocLayoutYOLO,
    LayoutRegion,
    detect_layout,
)
from agentic_extract.models import BoundingBox, RegionType
from agentic_extract.tools.docker_runner import ToolOutput


def test_layout_region_dataclass():
    lr = LayoutRegion(
        region_id="r1",
        region_type=RegionType.TEXT,
        bbox=BoundingBox(x=0.1, y=0.2, w=0.8, h=0.1),
        confidence=0.95,
        page=1,
    )
    assert lr.region_id == "r1"
    assert lr.region_type == RegionType.TEXT


def test_yolo_class_id_mapping():
    tool = DocLayoutYOLO()
    assert tool._map_class_id(0) == RegionType.TEXT
    assert tool._map_class_id(1) == RegionType.TABLE
    assert tool._map_class_id(2) == RegionType.FIGURE
    assert tool._map_class_id(3) == RegionType.FORMULA
    assert tool._map_class_id(4) == RegionType.TEXT  # caption -> text
    assert tool._map_class_id(999) == RegionType.TEXT  # unknown -> text


@patch.object(DocLayoutYOLO, "_docker_tool")
def test_detect_layout_parses_yolo_output(mock_tool: MagicMock, tmp_path: pathlib.Path):
    """DocLayout-YOLO JSON output should be parsed into LayoutRegion objects."""
    from PIL import Image
    img = Image.new("RGB", (1000, 1400), "white")
    img_path = tmp_path / "page.png"
    img.save(img_path)

    yolo_output = json.dumps([
        {
            "class_id": 0,
            "confidence": 0.96,
            "bbox": [50, 100, 900, 250],
        },
        {
            "class_id": 1,
            "confidence": 0.91,
            "bbox": [50, 300, 900, 700],
        },
        {
            "class_id": 2,
            "confidence": 0.88,
            "bbox": [100, 750, 800, 1200],
        },
    ])

    mock_tool.return_value.run.return_value = ToolOutput(
        stdout=yolo_output, stderr="", exit_code=0, duration_ms=500,
    )

    tool = DocLayoutYOLO()
    regions = tool.detect(img_path, page_number=1)

    assert len(regions) == 3
    assert regions[0].region_type == RegionType.TEXT
    assert regions[0].confidence == 0.96
    assert regions[1].region_type == RegionType.TABLE
    assert regions[2].region_type == RegionType.FIGURE

    # Bounding boxes should be normalized to [0, 1]
    assert 0.0 <= regions[0].bbox.x <= 1.0
    assert 0.0 <= regions[0].bbox.w <= 1.0


@patch.object(DocLayoutYOLO, "_docker_tool")
def test_detect_layout_handles_empty_output(mock_tool: MagicMock, tmp_path: pathlib.Path):
    from PIL import Image
    img = Image.new("RGB", (100, 100), "white")
    img_path = tmp_path / "blank.png"
    img.save(img_path)

    mock_tool.return_value.run.return_value = ToolOutput(
        stdout="[]", stderr="", exit_code=0, duration_ms=100,
    )

    tool = DocLayoutYOLO()
    regions = tool.detect(img_path, page_number=1)
    assert regions == []


@patch.object(DocLayoutYOLO, "_docker_tool")
def test_detect_layout_handles_docker_error(mock_tool: MagicMock, tmp_path: pathlib.Path):
    from PIL import Image
    img = Image.new("RGB", (100, 100), "white")
    img_path = tmp_path / "error.png"
    img.save(img_path)

    mock_tool.return_value.run.return_value = ToolOutput(
        stdout="", stderr="Container crashed", exit_code=1, duration_ms=50,
    )

    tool = DocLayoutYOLO()
    with pytest.raises(RuntimeError, match="Layout detection failed"):
        tool.detect(img_path, page_number=1)


def test_detect_layout_convenience_function(tmp_path: pathlib.Path):
    from PIL import Image
    img = Image.new("RGB", (500, 700), "white")
    img_path = tmp_path / "page.png"
    img.save(img_path)

    mock_regions = [
        LayoutRegion(
            region_id="r1", region_type=RegionType.TEXT,
            bbox=BoundingBox(x=0.1, y=0.1, w=0.8, h=0.2),
            confidence=0.95, page=1,
        ),
    ]

    with patch.object(DocLayoutYOLO, "detect", return_value=mock_regions):
        regions = detect_layout(img_path, page_number=1)
        assert len(regions) == 1
        assert regions[0].region_id == "r1"
