# tests/specialists/test_visual_formula.py
"""Tests for the Visual Specialist formula mode (GOT-OCR + pix2tex + voting)."""
import json
import pathlib
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from agentic_extract.models import (
    BoundingBox,
    FormulaContent,
    Region,
    RegionType,
)
from agentic_extract.specialists.visual_formula import (
    FormulaSpecialist,
    GotOCRResult,
    GotOCRTool,
    Pix2TexResult,
    Pix2TexTool,
)
from agentic_extract.tools.docker_runner import ToolOutput


# --- GOT-OCR Tool Tests ---

def test_gotocr_result_dataclass():
    result = GotOCRResult(latex=r"\frac{a}{b}", confidence=0.91)
    assert result.latex == r"\frac{a}{b}"


@patch.object(GotOCRTool, "_docker_tool")
def test_gotocr_extracts_latex(mock_tool: MagicMock, tmp_path: pathlib.Path):
    img = Image.new("RGB", (300, 100), "white")
    img_path = tmp_path / "formula.png"
    img.save(img_path)

    got_output = json.dumps({
        "latex": r"E = mc^{2}",
        "confidence": 0.94,
    })
    mock_tool.return_value.run.return_value = ToolOutput(
        stdout=got_output, stderr="", exit_code=0, duration_ms=1500,
    )

    tool = GotOCRTool()
    result = tool.extract(img_path)
    assert result.latex == r"E = mc^{2}"
    assert result.confidence == 0.94


@patch.object(GotOCRTool, "_docker_tool")
def test_gotocr_handles_failure(mock_tool: MagicMock, tmp_path: pathlib.Path):
    img = Image.new("RGB", (100, 100), "white")
    img_path = tmp_path / "bad_formula.png"
    img.save(img_path)

    mock_tool.return_value.run.return_value = ToolOutput(
        stdout="", stderr="GOT-OCR error", exit_code=1, duration_ms=50,
    )

    tool = GotOCRTool()
    with pytest.raises(RuntimeError, match="GOT-OCR failed"):
        tool.extract(img_path)


# --- pix2tex Tool Tests ---

def test_pix2tex_result_dataclass():
    result = Pix2TexResult(latex=r"\frac{a}{b}", confidence=0.88)
    assert result.latex == r"\frac{a}{b}"


@patch.object(Pix2TexTool, "_docker_tool")
def test_pix2tex_extracts_latex(mock_tool: MagicMock, tmp_path: pathlib.Path):
    img = Image.new("RGB", (300, 100), "white")
    img_path = tmp_path / "formula_p2t.png"
    img.save(img_path)

    p2t_output = json.dumps({
        "latex": r"\int_{0}^{\infty} e^{-x} dx",
        "confidence": 0.89,
    })
    mock_tool.return_value.run.return_value = ToolOutput(
        stdout=p2t_output, stderr="", exit_code=0, duration_ms=1000,
    )

    tool = Pix2TexTool()
    result = tool.extract(img_path)
    assert r"\int" in result.latex
    assert result.confidence == 0.89


@patch.object(Pix2TexTool, "_docker_tool")
def test_pix2tex_handles_failure(mock_tool: MagicMock, tmp_path: pathlib.Path):
    img = Image.new("RGB", (100, 100), "white")
    img_path = tmp_path / "bad_p2t.png"
    img.save(img_path)

    mock_tool.return_value.run.return_value = ToolOutput(
        stdout="", stderr="pix2tex error", exit_code=1, duration_ms=50,
    )

    tool = Pix2TexTool()
    with pytest.raises(RuntimeError, match="pix2tex failed"):
        tool.extract(img_path)


# --- Formula Specialist Tests ---

def test_formula_specialist_both_agree(tmp_path: pathlib.Path):
    """When both tools produce the same LaTeX, confidence is boosted."""
    img = Image.new("RGB", (300, 100), "white")
    img_path = tmp_path / "formula_agree.png"
    img.save(img_path)

    mock_got = MagicMock()
    mock_got.extract.return_value = GotOCRResult(
        latex=r"E = mc^{2}", confidence=0.92,
    )

    mock_p2t = MagicMock()
    mock_p2t.extract.return_value = Pix2TexResult(
        latex=r"E = mc^{2}", confidence=0.89,
    )

    specialist = FormulaSpecialist(gotocr_tool=mock_got, pix2tex_tool=mock_p2t)
    region = specialist.extract_sync(
        image_path=img_path,
        region_id="eq1",
        page=3,
        bbox=BoundingBox(x=0.1, y=0.4, w=0.8, h=0.1),
    )

    assert isinstance(region, Region)
    assert region.type == RegionType.FORMULA
    assert isinstance(region.content, FormulaContent)
    assert region.content.latex == r"E = mc^{2}"
    # Both agree: confidence should be boosted
    assert region.confidence >= 0.92
    assert "got-ocr" in region.extraction_method
    assert "pix2tex" in region.extraction_method


def test_formula_specialist_disagree_picks_higher(tmp_path: pathlib.Path):
    """When tools disagree, pick the one with higher confidence."""
    img = Image.new("RGB", (300, 100), "white")
    img_path = tmp_path / "formula_disagree.png"
    img.save(img_path)

    mock_got = MagicMock()
    mock_got.extract.return_value = GotOCRResult(
        latex=r"\frac{a}{b}", confidence=0.90,
    )

    mock_p2t = MagicMock()
    mock_p2t.extract.return_value = Pix2TexResult(
        latex=r"\frac{a}{6}", confidence=0.75,
    )

    specialist = FormulaSpecialist(gotocr_tool=mock_got, pix2tex_tool=mock_p2t)
    region = specialist.extract_sync(
        image_path=img_path,
        region_id="eq2",
        page=1,
        bbox=BoundingBox(x=0.0, y=0.0, w=1.0, h=1.0),
    )

    assert region.content.latex == r"\frac{a}{b}"
    assert region.confidence == 0.90
    assert region.needs_review is True  # Disagreement flags review


def test_formula_specialist_one_tool_fails(tmp_path: pathlib.Path):
    """If one tool fails, use the other."""
    img = Image.new("RGB", (300, 100), "white")
    img_path = tmp_path / "formula_one_fail.png"
    img.save(img_path)

    mock_got = MagicMock()
    mock_got.extract.side_effect = RuntimeError("GOT-OCR crashed")

    mock_p2t = MagicMock()
    mock_p2t.extract.return_value = Pix2TexResult(
        latex=r"x^{2} + y^{2} = r^{2}", confidence=0.87,
    )

    specialist = FormulaSpecialist(gotocr_tool=mock_got, pix2tex_tool=mock_p2t)
    region = specialist.extract_sync(
        image_path=img_path,
        region_id="eq3",
        page=1,
        bbox=BoundingBox(x=0.0, y=0.0, w=1.0, h=1.0),
    )

    assert region.content.latex == r"x^{2} + y^{2} = r^{2}"
    assert region.confidence == 0.87
    assert region.extraction_method == "pix2tex"


def test_formula_specialist_both_fail(tmp_path: pathlib.Path):
    """If both tools fail, return empty LaTeX with zero confidence."""
    img = Image.new("RGB", (300, 100), "white")
    img_path = tmp_path / "formula_both_fail.png"
    img.save(img_path)

    mock_got = MagicMock()
    mock_got.extract.side_effect = RuntimeError("GOT-OCR crashed")

    mock_p2t = MagicMock()
    mock_p2t.extract.side_effect = RuntimeError("pix2tex crashed")

    specialist = FormulaSpecialist(gotocr_tool=mock_got, pix2tex_tool=mock_p2t)
    region = specialist.extract_sync(
        image_path=img_path,
        region_id="eq4",
        page=1,
        bbox=BoundingBox(x=0.0, y=0.0, w=1.0, h=1.0),
    )

    assert region.content.latex == ""
    assert region.confidence == 0.0
    assert region.needs_review is True
