# tests/grounding/test_visual.py
"""Tests for visual grounding: linking extracted values to source bounding boxes."""
import json
import pathlib
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from agentic_extract.grounding.visual import (
    GroundedField,
    GroundedRegion,
    VisualGrounding,
    CellGrounding,
)
from agentic_extract.models import (
    BoundingBox,
    Region,
    RegionType,
    TextContent,
    TableContent,
    FigureContent,
)
from agentic_extract.tools.docker_runner import ToolOutput


def test_grounded_field_model():
    gf = GroundedField(
        field_name="patient_name",
        value="John Smith",
        bbox=BoundingBox(x=0.1, y=0.2, w=0.3, h=0.05),
        confidence=0.95,
        bbox_verified=True,
    )
    assert gf.field_name == "patient_name"
    assert gf.bbox_verified is True
    assert gf.bbox.x == 0.1


def test_grounded_field_unverified():
    gf = GroundedField(
        field_name="dosage",
        value="500mg",
        bbox=BoundingBox(x=0.4, y=0.5, w=0.2, h=0.03),
        confidence=0.78,
        bbox_verified=False,
    )
    assert gf.bbox_verified is False


def test_cell_grounding_model():
    cg = CellGrounding(
        row=0,
        col=1,
        value="3.2",
        bbox=BoundingBox(x=0.3, y=0.25, w=0.15, h=0.04),
        confidence=0.94,
        bbox_verified=True,
    )
    assert cg.row == 0
    assert cg.col == 1
    assert cg.bbox_verified is True


def test_grounded_region_text():
    gr = GroundedRegion(
        region_id="r1",
        region_type=RegionType.TEXT,
        page=1,
        region_bbox=BoundingBox(x=0.05, y=0.1, w=0.9, h=0.15),
        fields=[
            GroundedField(
                field_name="text",
                value="Hello world",
                bbox=BoundingBox(x=0.05, y=0.1, w=0.9, h=0.15),
                confidence=0.97,
                bbox_verified=True,
            ),
        ],
        cells=[],
    )
    assert gr.region_id == "r1"
    assert len(gr.fields) == 1
    assert gr.fields[0].bbox_verified is True


def test_grounded_region_table_with_cells():
    cells = [
        CellGrounding(
            row=0, col=0, value="Gene",
            bbox=BoundingBox(x=0.05, y=0.2, w=0.3, h=0.05),
            confidence=0.96, bbox_verified=True,
        ),
        CellGrounding(
            row=0, col=1, value="Value",
            bbox=BoundingBox(x=0.35, y=0.2, w=0.3, h=0.05),
            confidence=0.95, bbox_verified=True,
        ),
        CellGrounding(
            row=1, col=0, value="BRCA1",
            bbox=BoundingBox(x=0.05, y=0.25, w=0.3, h=0.05),
            confidence=0.93, bbox_verified=True,
        ),
        CellGrounding(
            row=1, col=1, value="3.2",
            bbox=BoundingBox(x=0.35, y=0.25, w=0.3, h=0.05),
            confidence=0.91, bbox_verified=True,
        ),
    ]
    gr = GroundedRegion(
        region_id="t1",
        region_type=RegionType.TABLE,
        page=2,
        region_bbox=BoundingBox(x=0.05, y=0.2, w=0.9, h=0.4),
        fields=[],
        cells=cells,
    )
    assert len(gr.cells) == 4
    assert gr.cells[2].value == "BRCA1"


def test_visual_grounding_init():
    vg = VisualGrounding()
    assert vg is not None


@patch("agentic_extract.grounding.visual._run_quick_ocr")
def test_visual_grounding_verifies_text_bbox(mock_ocr: MagicMock, tmp_path: pathlib.Path):
    """bbox_verified should be True when OCR on the cropped bbox matches extracted text."""
    img = Image.new("RGB", (1000, 200), "white")
    img_path = tmp_path / "page.png"
    img.save(img_path)

    # Mock OCR on the cropped region returns matching text
    mock_ocr.return_value = "Hello world"

    region = Region(
        id="r1",
        type=RegionType.TEXT,
        subtype=None,
        page=1,
        bbox=BoundingBox(x=0.05, y=0.1, w=0.9, h=0.15),
        content=TextContent(text="Hello world", markdown="Hello world"),
        confidence=0.97,
        extraction_method="paddleocr_3.0",
    )

    vg = VisualGrounding()
    grounded = vg.ground_region(region, img_path)

    assert isinstance(grounded, GroundedRegion)
    assert len(grounded.fields) == 1
    assert grounded.fields[0].bbox_verified is True
    assert grounded.fields[0].value == "Hello world"


@patch("agentic_extract.grounding.visual._run_quick_ocr")
def test_visual_grounding_fails_verification_on_mismatch(mock_ocr: MagicMock, tmp_path: pathlib.Path):
    """bbox_verified should be False when OCR on the bbox does not match extracted text."""
    img = Image.new("RGB", (1000, 200), "white")
    img_path = tmp_path / "page.png"
    img.save(img_path)

    # OCR returns completely different text
    mock_ocr.return_value = "Completely different text"

    region = Region(
        id="r2",
        type=RegionType.TEXT,
        subtype=None,
        page=1,
        bbox=BoundingBox(x=0.1, y=0.2, w=0.8, h=0.1),
        content=TextContent(text="Expected text here", markdown="Expected text here"),
        confidence=0.85,
        extraction_method="paddleocr_3.0 + claude-opus-4-20250514",
    )

    vg = VisualGrounding()
    grounded = vg.ground_region(region, img_path)

    assert grounded.fields[0].bbox_verified is False


@patch("agentic_extract.grounding.visual._run_quick_ocr")
def test_visual_grounding_table_cells(mock_ocr: MagicMock, tmp_path: pathlib.Path):
    """Table regions should produce cell-level grounding."""
    img = Image.new("RGB", (1000, 500), "white")
    img_path = tmp_path / "page.png"
    img.save(img_path)

    # OCR returns matching values for each cell
    mock_ocr.side_effect = ["Gene", "Value", "BRCA1", "3.2"]

    region = Region(
        id="t1",
        type=RegionType.TABLE,
        subtype=None,
        page=1,
        bbox=BoundingBox(x=0.05, y=0.2, w=0.9, h=0.4),
        content=TableContent(
            html="<table><tr><th>Gene</th><th>Value</th></tr><tr><td>BRCA1</td><td>3.2</td></tr></table>",
            json_data={"headers": ["Gene", "Value"], "rows": [{"Gene": "BRCA1", "Value": "3.2"}]},
            cell_bboxes=[
                {"row": 0, "col": 0, "bbox": BoundingBox(x=0.05, y=0.2, w=0.3, h=0.05)},
                {"row": 0, "col": 1, "bbox": BoundingBox(x=0.35, y=0.2, w=0.3, h=0.05)},
                {"row": 1, "col": 0, "bbox": BoundingBox(x=0.05, y=0.25, w=0.3, h=0.05)},
                {"row": 1, "col": 1, "bbox": BoundingBox(x=0.35, y=0.25, w=0.3, h=0.05)},
            ],
        ),
        confidence=0.94,
        extraction_method="docling + claude-opus-4-20250514",
    )

    vg = VisualGrounding()
    grounded = vg.ground_region(region, img_path)

    assert len(grounded.cells) == 4
    assert all(c.bbox_verified for c in grounded.cells)
    assert grounded.cells[0].value == "Gene"
    assert grounded.cells[3].value == "3.2"


def test_visual_grounding_normalized_coordinates():
    """All bounding box coordinates must be normalized to [0, 1]."""
    gf = GroundedField(
        field_name="test",
        value="x",
        bbox=BoundingBox(x=0.0, y=0.0, w=1.0, h=1.0),
        confidence=0.9,
        bbox_verified=True,
    )
    assert 0.0 <= gf.bbox.x <= 1.0
    assert 0.0 <= gf.bbox.y <= 1.0
    assert 0.0 <= gf.bbox.w <= 1.0
    assert 0.0 <= gf.bbox.h <= 1.0


@patch("agentic_extract.grounding.visual._run_quick_ocr")
def test_visual_grounding_figure_region(mock_ocr: MagicMock, tmp_path: pathlib.Path):
    """Figure regions should produce region-level grounding (no cell detail)."""
    img = Image.new("RGB", (800, 600), "white")
    img_path = tmp_path / "page.png"
    img.save(img_path)

    mock_ocr.return_value = "Bar chart showing gene expression"

    region = Region(
        id="f1",
        type=RegionType.FIGURE,
        subtype="bar_chart",
        page=3,
        bbox=BoundingBox(x=0.1, y=0.05, w=0.8, h=0.45),
        content=FigureContent(
            description="Bar chart showing gene expression levels",
            figure_type="bar_chart",
            figure_json={"title": "Gene Expression"},
        ),
        confidence=0.86,
        extraction_method="deplot + claude-opus-4-20250514",
    )

    vg = VisualGrounding()
    grounded = vg.ground_region(region, img_path)

    assert grounded.region_type == RegionType.FIGURE
    assert len(grounded.fields) >= 1
    assert grounded.cells == []
