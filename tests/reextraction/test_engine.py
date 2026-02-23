"""Tests for the re-extraction engine with model switching."""
import pathlib
from unittest.mock import AsyncMock, MagicMock

import pytest

from agentic_extract.clients.vlm import VLMResponse
from agentic_extract.models import (
    BoundingBox,
    Region,
    RegionType,
    TextContent,
    TableContent,
    FigureContent,
    HandwritingContent,
)
from agentic_extract.reextraction.engine import (
    ReExtractionEngine,
    ReExtractionResult,
    ReExtractionCandidate,
    ModelSwitchStrategy,
)


def test_reextraction_result_model():
    candidate = ReExtractionCandidate(
        value="500mg",
        confidence=0.85,
        model="claude-opus-4-20250514",
        extraction_method="paddleocr_3.0 + claude-opus-4-20250514",
    )
    result = ReExtractionResult(
        region_id="r1",
        field_name="dosage",
        original_value="500mg",
        original_confidence=0.75,
        original_model="claude-opus-4-20250514",
        candidates=[candidate],
        final_value="500mg",
        final_confidence=0.85,
        retries_used=1,
        models_agreed=False,
        accepted=False,
        flagged=True,
    )
    assert result.region_id == "r1"
    assert result.retries_used == 1
    assert len(result.candidates) == 1
    assert result.flagged is True


def test_reextraction_candidate_model():
    c = ReExtractionCandidate(
        value="BRCA1",
        confidence=0.92,
        model="gpt-4o",
        extraction_method="paddleocr_3.0 + gpt-4o",
    )
    assert c.value == "BRCA1"
    assert c.confidence == 0.92


def test_model_switch_strategy_claude_to_codex():
    strategy = ModelSwitchStrategy()
    alt = strategy.get_alternate_model("claude-opus-4-20250514")
    assert alt == "gpt-4o"


def test_model_switch_strategy_codex_to_claude():
    strategy = ModelSwitchStrategy()
    alt = strategy.get_alternate_model("gpt-4o")
    assert alt == "claude-opus-4-20250514"


def test_model_switch_strategy_unknown_defaults_to_claude():
    strategy = ModelSwitchStrategy()
    alt = strategy.get_alternate_model("some-unknown-model")
    assert alt == "claude-opus-4-20250514"


@pytest.mark.asyncio
async def test_reextraction_engine_init():
    mock_claude = AsyncMock()
    mock_codex = AsyncMock()
    engine = ReExtractionEngine(
        claude_client=mock_claude,
        codex_client=mock_codex,
        max_retries=2,
    )
    assert engine.max_retries == 2


@pytest.mark.asyncio
async def test_reextraction_models_agree_boosts_confidence(tmp_path: pathlib.Path):
    """When both models agree, confidence should be boosted by +0.10."""
    from PIL import Image
    img = Image.new("RGB", (200, 50), "white")
    img_path = tmp_path / "region.png"
    img.save(img_path)

    mock_claude = AsyncMock()
    mock_codex = AsyncMock()

    # Original extraction was by Claude at 0.82
    # Codex re-extraction returns the same value at 0.84
    mock_codex.send_vision_request.return_value = VLMResponse(
        content={"corrected_text": "500mg"},
        confidence=0.84,
        model="gpt-4o",
        usage_tokens=100,
        duration_ms=800,
    )

    engine = ReExtractionEngine(
        claude_client=mock_claude,
        codex_client=mock_codex,
        max_retries=2,
    )

    original_region = Region(
        id="r1",
        type=RegionType.TEXT,
        subtype=None,
        page=1,
        bbox=BoundingBox(x=0.1, y=0.1, w=0.8, h=0.05),
        content=TextContent(text="500mg", markdown="500mg"),
        confidence=0.82,
        extraction_method="paddleocr_3.0 + claude-opus-4-20250514",
    )

    result = await engine.re_extract_field(
        region=original_region,
        field_name="dosage",
        image_path=img_path,
        original_model="claude-opus-4-20250514",
    )

    assert isinstance(result, ReExtractionResult)
    # Models agreed on "500mg", so confidence boosted by 0.10
    assert result.models_agreed is True
    assert result.final_confidence >= 0.92  # 0.82 + 0.10
    assert result.final_value == "500mg"
    assert result.accepted is True  # >= 0.90 threshold
    assert result.retries_used == 1


@pytest.mark.asyncio
async def test_reextraction_models_disagree_flags_field(tmp_path: pathlib.Path):
    """When models disagree after max retries, field should be flagged."""
    from PIL import Image
    img = Image.new("RGB", (200, 50), "white")
    img_path = tmp_path / "region.png"
    img.save(img_path)

    mock_claude = AsyncMock()
    mock_codex = AsyncMock()

    # Claude says "500mg", Codex says "800mg" (disagreement)
    mock_codex.send_vision_request.return_value = VLMResponse(
        content={"corrected_text": "800mg"},
        confidence=0.80,
        model="gpt-4o",
        usage_tokens=100,
        duration_ms=800,
    )
    # On second retry (back to Claude with different prompt), still says "500mg"
    mock_claude.send_vision_request.return_value = VLMResponse(
        content={"corrected_text": "500mg"},
        confidence=0.83,
        model="claude-opus-4-20250514",
        usage_tokens=120,
        duration_ms=900,
    )

    engine = ReExtractionEngine(
        claude_client=mock_claude,
        codex_client=mock_codex,
        max_retries=2,
    )

    original_region = Region(
        id="r2",
        type=RegionType.TEXT,
        subtype=None,
        page=1,
        bbox=BoundingBox(x=0.1, y=0.3, w=0.8, h=0.05),
        content=TextContent(text="500mg", markdown="500mg"),
        confidence=0.78,
        extraction_method="paddleocr_3.0 + claude-opus-4-20250514",
    )

    result = await engine.re_extract_field(
        region=original_region,
        field_name="dosage",
        image_path=img_path,
        original_model="claude-opus-4-20250514",
    )

    assert result.models_agreed is False
    assert result.flagged is True
    assert result.retries_used == 2
    assert len(result.candidates) == 2  # Both candidates preserved


@pytest.mark.asyncio
async def test_reextraction_respects_max_retries(tmp_path: pathlib.Path):
    """Engine must not exceed max_retries."""
    from PIL import Image
    img = Image.new("RGB", (200, 50), "white")
    img_path = tmp_path / "region.png"
    img.save(img_path)

    mock_claude = AsyncMock()
    mock_codex = AsyncMock()

    # Both always return low confidence, different values
    mock_codex.send_vision_request.return_value = VLMResponse(
        content={"corrected_text": "value_b"},
        confidence=0.72,
        model="gpt-4o",
        usage_tokens=50,
        duration_ms=500,
    )
    mock_claude.send_vision_request.return_value = VLMResponse(
        content={"corrected_text": "value_c"},
        confidence=0.71,
        model="claude-opus-4-20250514",
        usage_tokens=60,
        duration_ms=600,
    )

    engine = ReExtractionEngine(
        claude_client=mock_claude,
        codex_client=mock_codex,
        max_retries=2,
    )

    original_region = Region(
        id="r3",
        type=RegionType.TEXT,
        subtype=None,
        page=2,
        bbox=BoundingBox(x=0.0, y=0.0, w=1.0, h=1.0),
        content=TextContent(text="value_a", markdown="value_a"),
        confidence=0.75,
        extraction_method="paddleocr_3.0 + claude-opus-4-20250514",
    )

    result = await engine.re_extract_field(
        region=original_region,
        field_name="field_x",
        image_path=img_path,
        original_model="claude-opus-4-20250514",
    )

    assert result.retries_used <= 2
    assert result.flagged is True


@pytest.mark.asyncio
async def test_reextraction_vlm_failure_counts_as_retry(tmp_path: pathlib.Path):
    """If a VLM call fails during re-extraction, it counts as a used retry."""
    from PIL import Image
    img = Image.new("RGB", (200, 50), "white")
    img_path = tmp_path / "region.png"
    img.save(img_path)

    mock_claude = AsyncMock()
    mock_codex = AsyncMock()

    # Codex fails on first retry
    mock_codex.send_vision_request.side_effect = RuntimeError("API timeout")
    # Claude succeeds on second retry but disagrees
    mock_claude.send_vision_request.return_value = VLMResponse(
        content={"corrected_text": "fallback_value"},
        confidence=0.76,
        model="claude-opus-4-20250514",
        usage_tokens=80,
        duration_ms=700,
    )

    engine = ReExtractionEngine(
        claude_client=mock_claude,
        codex_client=mock_codex,
        max_retries=2,
    )

    original_region = Region(
        id="r4",
        type=RegionType.TEXT,
        subtype=None,
        page=1,
        bbox=BoundingBox(x=0.2, y=0.2, w=0.6, h=0.1),
        content=TextContent(text="original_value", markdown="original_value"),
        confidence=0.80,
        extraction_method="paddleocr_3.0 + claude-opus-4-20250514",
    )

    result = await engine.re_extract_field(
        region=original_region,
        field_name="field_y",
        image_path=img_path,
        original_model="claude-opus-4-20250514",
    )

    assert result.retries_used == 2
    # Should still produce a result (not crash)
    assert result.final_value is not None


@pytest.mark.asyncio
async def test_reextraction_table_region(tmp_path: pathlib.Path):
    """Re-extraction should work for table regions too."""
    from PIL import Image
    img = Image.new("RGB", (400, 200), "white")
    img_path = tmp_path / "table_region.png"
    img.save(img_path)

    mock_claude = AsyncMock()
    mock_codex = AsyncMock()

    mock_codex.send_vision_request.return_value = VLMResponse(
        content={"headers": ["Gene", "Value"], "rows": [{"Gene": "BRCA1", "Value": "3.2"}]},
        confidence=0.88,
        model="gpt-4o",
        usage_tokens=150,
        duration_ms=1200,
    )

    engine = ReExtractionEngine(
        claude_client=mock_claude,
        codex_client=mock_codex,
        max_retries=2,
    )

    original_region = Region(
        id="t1",
        type=RegionType.TABLE,
        subtype=None,
        page=1,
        bbox=BoundingBox(x=0.05, y=0.2, w=0.9, h=0.4),
        content=TableContent(
            html="<table><tr><td>BRCA1</td><td>3.2</td></tr></table>",
            json_data={"headers": ["Gene", "Value"], "rows": [{"Gene": "BRCA1", "Value": "3.2"}]},
        ),
        confidence=0.82,
        extraction_method="docling + claude-opus-4-20250514",
    )

    result = await engine.re_extract_field(
        region=original_region,
        field_name="table_data",
        image_path=img_path,
        original_model="claude-opus-4-20250514",
    )

    assert isinstance(result, ReExtractionResult)
    assert result.retries_used >= 1
