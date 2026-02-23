# tests/specialists/test_visual_handwriting.py
"""Tests for the Visual Specialist handwriting mode (TrOCR + DocEnTr + dual-model)."""
import json
import pathlib
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from PIL import Image

from agentic_extract.clients.vlm import VLMResponse
from agentic_extract.models import (
    BoundingBox,
    HandwritingContent,
    Region,
    RegionType,
)
from agentic_extract.specialists.visual_handwriting import (
    DocEnTrTool,
    HandwritingSpecialist,
    TrOCRResult,
    TrOCRTool,
)
from agentic_extract.tools.docker_runner import ToolOutput


# --- TrOCR Tool Tests ---

def test_trocr_result_dataclass():
    result = TrOCRResult(
        text="Patient notes here",
        confidence=0.82,
        per_char_confidences=[0.85, 0.80, 0.79, 0.88],
    )
    assert result.text == "Patient notes here"
    assert result.confidence == 0.82


@patch.object(TrOCRTool, "_docker_tool")
def test_trocr_extracts_text(mock_tool: MagicMock, tmp_path: pathlib.Path):
    img = Image.new("RGB", (600, 200), "white")
    img_path = tmp_path / "handwriting.png"
    img.save(img_path)

    trocr_output = json.dumps({
        "text": "Administered 500mg at 14:00",
        "confidence": 0.83,
        "per_char_confidences": [0.85, 0.82, 0.79, 0.88, 0.80],
    })
    mock_tool.return_value.run.return_value = ToolOutput(
        stdout=trocr_output, stderr="", exit_code=0, duration_ms=1200,
    )

    tool = TrOCRTool()
    result = tool.extract(img_path)
    assert "500mg" in result.text
    assert result.confidence == 0.83


@patch.object(TrOCRTool, "_docker_tool")
def test_trocr_handles_failure(mock_tool: MagicMock, tmp_path: pathlib.Path):
    img = Image.new("RGB", (100, 100), "white")
    img_path = tmp_path / "bad_hw.png"
    img.save(img_path)

    mock_tool.return_value.run.return_value = ToolOutput(
        stdout="", stderr="TrOCR error", exit_code=1, duration_ms=50,
    )

    tool = TrOCRTool()
    with pytest.raises(RuntimeError, match="TrOCR failed"):
        tool.extract(img_path)


# --- DocEnTr Tool Tests ---

@patch.object(DocEnTrTool, "_docker_tool")
def test_docentr_enhances_image(mock_tool: MagicMock, tmp_path: pathlib.Path):
    img = Image.new("RGB", (400, 200), (100, 100, 100))
    img_path = tmp_path / "degraded.png"
    img.save(img_path)

    enhanced_path = tmp_path / "enhanced.png"
    Image.new("RGB", (400, 200), "white").save(enhanced_path)

    docentr_output = json.dumps({"enhanced_path": str(enhanced_path)})
    mock_tool.return_value.run.return_value = ToolOutput(
        stdout=docentr_output, stderr="", exit_code=0, duration_ms=2000,
    )

    tool = DocEnTrTool()
    result_path = tool.enhance(img_path, output_dir=tmp_path)
    assert result_path.exists()


@patch.object(DocEnTrTool, "_docker_tool")
def test_docentr_handles_failure(mock_tool: MagicMock, tmp_path: pathlib.Path):
    img = Image.new("RGB", (100, 100), "white")
    img_path = tmp_path / "bad.png"
    img.save(img_path)

    mock_tool.return_value.run.return_value = ToolOutput(
        stdout="", stderr="DocEnTr error", exit_code=1, duration_ms=50,
    )

    tool = DocEnTrTool()
    with pytest.raises(RuntimeError, match="DocEnTr failed"):
        tool.enhance(img_path, output_dir=tmp_path)


# --- Handwriting Specialist Tests ---

@pytest.mark.asyncio
async def test_handwriting_specialist_dual_model_agree(tmp_path: pathlib.Path):
    """When Codex and Claude agree, confidence is boosted."""
    img = Image.new("RGB", (600, 200), "white")
    img_path = tmp_path / "hw_agree.png"
    img.save(img_path)

    mock_trocr = MagicMock()
    mock_trocr.extract.return_value = TrOCRResult(
        text="Patlent admlnlstered 500mg",
        confidence=0.75,
        per_char_confidences=[0.7, 0.8, 0.6],
    )

    mock_codex = AsyncMock()
    mock_codex.send_vision_request.return_value = VLMResponse(
        content={"corrected_text": "Patient administered 500mg"},
        confidence=0.88,
        model="gpt-4o",
        usage_tokens=150,
        duration_ms=1000,
    )

    mock_claude = AsyncMock()
    mock_claude.send_vision_request.return_value = VLMResponse(
        content={"verified_text": "Patient administered 500mg", "hallucination_risk": "low"},
        confidence=0.90,
        model="claude-opus-4-20250514",
        usage_tokens=120,
        duration_ms=900,
    )

    specialist = HandwritingSpecialist(
        trocr_tool=mock_trocr,
        docentr_tool=None,
        codex_client=mock_codex,
        claude_client=mock_claude,
    )
    region = await specialist.extract(
        image_path=img_path,
        region_id="h1",
        page=7,
        bbox=BoundingBox(x=0.05, y=0.50, w=0.90, h=0.30),
        needs_enhancement=False,
    )

    assert region.content.text == "Patient administered 500mg"
    assert region.confidence >= 0.88
    assert "codex" in region.extraction_method.lower() or "gpt" in region.extraction_method.lower()
    mock_codex.send_vision_request.assert_called_once()
    mock_claude.send_vision_request.assert_called_once()


@pytest.mark.asyncio
async def test_handwriting_specialist_dual_model_disagree(tmp_path: pathlib.Path):
    """When models disagree, take the one with higher confidence but flag for review."""
    img = Image.new("RGB", (600, 200), "white")
    img_path = tmp_path / "hw_disagree.png"
    img.save(img_path)

    mock_trocr = MagicMock()
    mock_trocr.extract.return_value = TrOCRResult(
        text="500mg dally", confidence=0.65, per_char_confidences=[],
    )

    mock_codex = AsyncMock()
    mock_codex.send_vision_request.return_value = VLMResponse(
        content={"corrected_text": "500mg daily"},
        confidence=0.82,
        model="gpt-4o",
        usage_tokens=150,
        duration_ms=1000,
    )

    mock_claude = AsyncMock()
    mock_claude.send_vision_request.return_value = VLMResponse(
        content={"verified_text": "500mg dally", "hallucination_risk": "medium"},
        confidence=0.70,
        model="claude-opus-4-20250514",
        usage_tokens=120,
        duration_ms=900,
    )

    specialist = HandwritingSpecialist(
        trocr_tool=mock_trocr,
        docentr_tool=None,
        codex_client=mock_codex,
        claude_client=mock_claude,
    )
    region = await specialist.extract(
        image_path=img_path,
        region_id="h2",
        page=7,
        bbox=BoundingBox(x=0.0, y=0.0, w=1.0, h=1.0),
        needs_enhancement=False,
    )

    # Should use Codex result (higher confidence)
    assert region.content.text == "500mg daily"
    assert region.needs_review is True


@pytest.mark.asyncio
async def test_handwriting_specialist_codex_failure(tmp_path: pathlib.Path):
    """If Codex fails, Claude alone should still work."""
    img = Image.new("RGB", (600, 200), "white")
    img_path = tmp_path / "hw_codex_fail.png"
    img.save(img_path)

    mock_trocr = MagicMock()
    mock_trocr.extract.return_value = TrOCRResult(
        text="some handwritten text", confidence=0.70, per_char_confidences=[],
    )

    mock_codex = AsyncMock()
    mock_codex.send_vision_request.side_effect = RuntimeError("Codex API down")

    mock_claude = AsyncMock()
    mock_claude.send_vision_request.return_value = VLMResponse(
        content={"verified_text": "some handwritten text", "hallucination_risk": "low"},
        confidence=0.85,
        model="claude-opus-4-20250514",
        usage_tokens=100,
        duration_ms=800,
    )

    specialist = HandwritingSpecialist(
        trocr_tool=mock_trocr,
        docentr_tool=None,
        codex_client=mock_codex,
        claude_client=mock_claude,
    )
    region = await specialist.extract(
        image_path=img_path,
        region_id="h3",
        page=1,
        bbox=BoundingBox(x=0.0, y=0.0, w=1.0, h=1.0),
        needs_enhancement=False,
    )

    assert region.content.text == "some handwritten text"
    assert "claude" in region.extraction_method


@pytest.mark.asyncio
async def test_handwriting_specialist_enhances_degraded(tmp_path: pathlib.Path):
    """When needs_enhancement=True, DocEnTr should be called first."""
    img = Image.new("RGB", (600, 200), (80, 80, 80))
    img_path = tmp_path / "hw_degraded.png"
    img.save(img_path)

    enhanced = tmp_path / "enhanced.png"
    Image.new("RGB", (600, 200), "white").save(enhanced)

    mock_docentr = MagicMock()
    mock_docentr.enhance.return_value = enhanced

    mock_trocr = MagicMock()
    mock_trocr.extract.return_value = TrOCRResult(
        text="enhanced text", confidence=0.80, per_char_confidences=[],
    )

    specialist = HandwritingSpecialist(
        trocr_tool=mock_trocr,
        docentr_tool=mock_docentr,
        codex_client=None,
        claude_client=None,
    )
    region = await specialist.extract(
        image_path=img_path,
        region_id="h4",
        page=1,
        bbox=BoundingBox(x=0.0, y=0.0, w=1.0, h=1.0),
        needs_enhancement=True,
    )

    mock_docentr.enhance.assert_called_once()
    assert region.content.text == "enhanced text"
    assert "docentr" in region.extraction_method


@pytest.mark.asyncio
async def test_handwriting_specialist_all_vlm_fail_fallback(tmp_path: pathlib.Path):
    """If both VLMs fail, fall back to raw TrOCR output."""
    img = Image.new("RGB", (600, 200), "white")
    img_path = tmp_path / "hw_fallback.png"
    img.save(img_path)

    mock_trocr = MagicMock()
    mock_trocr.extract.return_value = TrOCRResult(
        text="raw ocr output", confidence=0.60, per_char_confidences=[],
    )

    mock_codex = AsyncMock()
    mock_codex.send_vision_request.side_effect = RuntimeError("Codex down")
    mock_claude = AsyncMock()
    mock_claude.send_vision_request.side_effect = RuntimeError("Claude down")

    specialist = HandwritingSpecialist(
        trocr_tool=mock_trocr,
        docentr_tool=None,
        codex_client=mock_codex,
        claude_client=mock_claude,
    )
    region = await specialist.extract(
        image_path=img_path,
        region_id="h5",
        page=1,
        bbox=BoundingBox(x=0.0, y=0.0, w=1.0, h=1.0),
        needs_enhancement=False,
    )

    assert region.content.text == "raw ocr output"
    assert region.confidence == 0.60
    assert region.extraction_method == "trocr"
