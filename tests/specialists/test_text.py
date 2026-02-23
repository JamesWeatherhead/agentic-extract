"""Tests for the Text Specialist (PaddleOCR + Claude enhancement)."""
import json
import pathlib
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from PIL import Image

from agentic_extract.models import BoundingBox, Region, RegionType, TextContent
from agentic_extract.specialists.text import (
    PaddleOCRTool,
    TextSpecialist,
    OCRResult,
)
from agentic_extract.tools.docker_runner import ToolOutput


def test_ocr_result_dataclass():
    result = OCRResult(
        text="Hello world",
        confidence=0.97,
        per_char_confidences=[0.98, 0.99, 0.96, 0.97, 0.98],
    )
    assert result.text == "Hello world"
    assert result.confidence == 0.97


@patch.object(PaddleOCRTool, "_docker_tool")
def test_paddleocr_extracts_text(mock_tool: MagicMock, tmp_path: pathlib.Path):
    img = Image.new("RGB", (400, 100), "white")
    img_path = tmp_path / "text_region.png"
    img.save(img_path)

    paddle_output = json.dumps({
        "text": "The quick brown fox",
        "confidence": 0.96,
        "per_char_confidences": [0.95, 0.97, 0.98, 0.94, 0.96],
    })
    mock_tool.return_value.run.return_value = ToolOutput(
        stdout=paddle_output, stderr="", exit_code=0, duration_ms=800,
    )

    tool = PaddleOCRTool()
    result = tool.extract(img_path)

    assert result.text == "The quick brown fox"
    assert result.confidence == 0.96


@patch.object(PaddleOCRTool, "_docker_tool")
def test_paddleocr_handles_failure(mock_tool: MagicMock, tmp_path: pathlib.Path):
    img = Image.new("RGB", (100, 100), "white")
    img_path = tmp_path / "bad.png"
    img.save(img_path)

    mock_tool.return_value.run.return_value = ToolOutput(
        stdout="", stderr="PaddleOCR error", exit_code=1, duration_ms=100,
    )

    tool = PaddleOCRTool()
    with pytest.raises(RuntimeError, match="PaddleOCR failed"):
        tool.extract(img_path)


@pytest.mark.asyncio
async def test_text_specialist_high_confidence_skips_vlm(tmp_path: pathlib.Path):
    """When PaddleOCR confidence >= 0.95, the VLM call should be skipped."""
    img = Image.new("RGB", (400, 100), "white")
    img_path = tmp_path / "clean.png"
    img.save(img_path)

    mock_ocr = MagicMock()
    mock_ocr.extract.return_value = OCRResult(
        text="Clean text here",
        confidence=0.98,
        per_char_confidences=[0.99, 0.98, 0.97],
    )

    mock_vlm = AsyncMock()

    specialist = TextSpecialist(ocr_tool=mock_ocr, vlm_client=mock_vlm)
    region = await specialist.extract(
        image_path=img_path,
        region_id="r1",
        page=1,
        bbox=BoundingBox(x=0.05, y=0.10, w=0.90, h=0.05),
    )

    assert isinstance(region, Region)
    assert region.type == RegionType.TEXT
    assert isinstance(region.content, TextContent)
    assert region.content.text == "Clean text here"
    assert region.confidence == 0.98
    # VLM should NOT have been called
    mock_vlm.send_vision_request.assert_not_called()


@pytest.mark.asyncio
async def test_text_specialist_low_confidence_calls_vlm(tmp_path: pathlib.Path):
    """When PaddleOCR confidence < 0.95, Claude should enhance the text."""
    img = Image.new("RGB", (400, 100), "white")
    img_path = tmp_path / "degraded.png"
    img.save(img_path)

    mock_ocr = MagicMock()
    mock_ocr.extract.return_value = OCRResult(
        text="Th3 qu1ck br0wn f0x",
        confidence=0.72,
        per_char_confidences=[0.6, 0.8, 0.5],
    )

    from agentic_extract.clients.vlm import VLMResponse
    mock_vlm = AsyncMock()
    mock_vlm.send_vision_request.return_value = VLMResponse(
        content={"corrected_text": "The quick brown fox"},
        confidence=0.92,
        model="claude-opus-4-20250514",
        usage_tokens=200,
        duration_ms=1500,
    )

    specialist = TextSpecialist(ocr_tool=mock_ocr, vlm_client=mock_vlm)
    region = await specialist.extract(
        image_path=img_path,
        region_id="r1",
        page=1,
        bbox=BoundingBox(x=0.05, y=0.10, w=0.90, h=0.05),
    )

    assert region.content.text == "The quick brown fox"
    assert region.confidence == 0.92
    assert "claude" in region.extraction_method
    mock_vlm.send_vision_request.assert_called_once()


@pytest.mark.asyncio
async def test_text_specialist_vlm_failure_falls_back_to_ocr(tmp_path: pathlib.Path):
    """If Claude fails, the specialist should return OCR text as fallback."""
    img = Image.new("RGB", (400, 100), "white")
    img_path = tmp_path / "fallback.png"
    img.save(img_path)

    mock_ocr = MagicMock()
    mock_ocr.extract.return_value = OCRResult(
        text="Fa11back text",
        confidence=0.80,
        per_char_confidences=[0.7, 0.8, 0.9],
    )

    mock_vlm = AsyncMock()
    mock_vlm.send_vision_request.side_effect = RuntimeError("API down")

    specialist = TextSpecialist(ocr_tool=mock_ocr, vlm_client=mock_vlm)
    region = await specialist.extract(
        image_path=img_path,
        region_id="r1",
        page=1,
        bbox=BoundingBox(x=0.0, y=0.0, w=1.0, h=1.0),
    )

    # Should fall back to OCR text
    assert region.content.text == "Fa11back text"
    assert region.confidence == 0.80
    assert "paddleocr" in region.extraction_method
