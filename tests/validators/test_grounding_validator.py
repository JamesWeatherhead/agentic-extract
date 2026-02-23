# tests/validators/test_grounding_validator.py
"""Tests for Validator Layers 4-5: visual grounding check + confidence calibration."""
import pathlib
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from agentic_extract.models import (
    BoundingBox,
    Region,
    RegionType,
    TextContent,
    TableContent,
)
from agentic_extract.specialists.text import OCRResult, PaddleOCRTool
from agentic_extract.validators.grounding_validator import (
    GroundingResult,
    ValidationDecision,
    calibrate_confidence,
    check_visual_grounding,
    compute_weighted_confidence,
    make_validation_decision,
)


def _make_text_region(
    rid: str, text: str, conf: float, bbox: BoundingBox | None = None,
) -> Region:
    return Region(
        id=rid,
        type=RegionType.TEXT,
        subtype=None,
        page=1,
        bbox=bbox or BoundingBox(x=0.1, y=0.1, w=0.8, h=0.1),
        content=TextContent(text=text, markdown=text),
        confidence=conf,
        extraction_method="paddleocr_3.0",
    )


# --- ValidationDecision enum tests ---

def test_validation_decision_accept():
    assert ValidationDecision.ACCEPT == "accept"


def test_validation_decision_re_extract():
    assert ValidationDecision.RE_EXTRACT == "re_extract"


def test_validation_decision_flag():
    assert ValidationDecision.FLAG == "flag"


# --- Visual Grounding (Layer 4) tests ---

def test_grounding_result_dataclass():
    gr = GroundingResult(
        region_id="r1",
        extracted_text="Hello world",
        ocr_text="Hello world",
        edit_distance=0,
        grounding_score=1.0,
    )
    assert gr.grounding_score == 1.0
    assert gr.edit_distance == 0


@patch.object(PaddleOCRTool, "_docker_tool")
def test_visual_grounding_matching_text(mock_tool: MagicMock, tmp_path: pathlib.Path):
    """When OCR on the bbox crop matches the extracted text, grounding is high."""
    # Create a page image
    page_img = Image.new("RGB", (1000, 1000), "white")
    page_path = tmp_path / "page.png"
    page_img.save(page_path)

    import json
    mock_tool.return_value.run.return_value = MagicMock(
        stdout=json.dumps({"text": "Hello world", "confidence": 0.95, "per_char_confidences": []}),
        stderr="",
        exit_code=0,
        duration_ms=500,
    )

    region = _make_text_region("r1", "Hello world", 0.95)
    results = check_visual_grounding(
        regions=[region],
        page_images={1: page_path},
        ocr_tool=PaddleOCRTool(),
    )

    assert len(results) == 1
    assert results[0].grounding_score >= 0.9
    assert results[0].edit_distance == 0


@patch.object(PaddleOCRTool, "_docker_tool")
def test_visual_grounding_mismatched_text(mock_tool: MagicMock, tmp_path: pathlib.Path):
    """When OCR on the crop diverges from extracted text, grounding is low."""
    page_img = Image.new("RGB", (1000, 1000), "white")
    page_path = tmp_path / "page.png"
    page_img.save(page_path)

    import json
    mock_tool.return_value.run.return_value = MagicMock(
        stdout=json.dumps({"text": "Completely different text", "confidence": 0.90, "per_char_confidences": []}),
        stderr="",
        exit_code=0,
        duration_ms=500,
    )

    region = _make_text_region("r1", "Hello world", 0.95)
    results = check_visual_grounding(
        regions=[region],
        page_images={1: page_path},
        ocr_tool=PaddleOCRTool(),
    )

    assert len(results) == 1
    assert results[0].grounding_score < 0.5
    assert results[0].edit_distance > 0


# --- Confidence Calibration (Layer 5) tests ---

def test_compute_weighted_confidence():
    """Weighted confidence formula: 0.3*ocr + 0.4*vlm + 0.3*validation."""
    result = compute_weighted_confidence(
        ocr_confidence=0.95,
        vlm_confidence=0.90,
        validation_score=0.85,
    )
    expected = 0.95 * 0.3 + 0.90 * 0.4 + 0.85 * 0.3
    assert abs(result - expected) < 0.001


def test_compute_weighted_confidence_clamps():
    """Result must be clamped to [0, 1]."""
    result = compute_weighted_confidence(
        ocr_confidence=1.0,
        vlm_confidence=1.0,
        validation_score=1.0,
    )
    assert result <= 1.0

    result = compute_weighted_confidence(
        ocr_confidence=0.0,
        vlm_confidence=0.0,
        validation_score=0.0,
    )
    assert result >= 0.0


def test_calibrate_confidence_with_temperature():
    """Temperature scaling should adjust raw confidence."""
    raw = 0.85
    calibrated = calibrate_confidence(raw, temperature=1.5)
    # With temperature > 1, confidence should be pulled toward 0.5
    assert calibrated != raw
    assert 0.0 <= calibrated <= 1.0


def test_calibrate_confidence_temperature_one():
    """Temperature=1.0 should leave confidence unchanged (identity)."""
    raw = 0.90
    calibrated = calibrate_confidence(raw, temperature=1.0)
    assert abs(calibrated - raw) < 0.01


# --- Decision Gate tests ---

def test_decision_accept_high_confidence():
    decision = make_validation_decision(confidence=0.95)
    assert decision == ValidationDecision.ACCEPT


def test_decision_re_extract_medium_confidence():
    decision = make_validation_decision(confidence=0.80)
    assert decision == ValidationDecision.RE_EXTRACT


def test_decision_flag_low_confidence():
    decision = make_validation_decision(confidence=0.60)
    assert decision == ValidationDecision.FLAG


def test_decision_boundary_accept():
    """Confidence exactly at 0.90 should be ACCEPT."""
    decision = make_validation_decision(confidence=0.90)
    assert decision == ValidationDecision.ACCEPT


def test_decision_boundary_re_extract():
    """Confidence exactly at 0.70 should be RE_EXTRACT."""
    decision = make_validation_decision(confidence=0.70)
    assert decision == ValidationDecision.RE_EXTRACT


def test_decision_boundary_flag():
    """Confidence just below 0.70 should be FLAG."""
    decision = make_validation_decision(confidence=0.69)
    assert decision == ValidationDecision.FLAG
