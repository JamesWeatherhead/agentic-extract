# tests/validators/test_semantic_validator.py
"""Tests for Validator Layer 3: LLM-assisted semantic validation."""
import pathlib
from unittest.mock import AsyncMock, MagicMock

import pytest

from agentic_extract.clients.vlm import VLMResponse
from agentic_extract.models import (
    BoundingBox,
    Region,
    RegionType,
    TableContent,
    TextContent,
)
from agentic_extract.validators.semantic_validator import (
    SemanticIssue,
    validate_semantics,
)


def _make_text_region(rid: str, text: str) -> Region:
    return Region(
        id=rid,
        type=RegionType.TEXT,
        subtype=None,
        page=1,
        bbox=BoundingBox(x=0.0, y=0.0, w=1.0, h=0.5),
        content=TextContent(text=text, markdown=text),
        confidence=0.95,
        extraction_method="paddleocr_3.0",
    )


def _make_table_region(rid: str) -> Region:
    return Region(
        id=rid,
        type=RegionType.TABLE,
        subtype=None,
        page=1,
        bbox=BoundingBox(x=0.0, y=0.5, w=1.0, h=0.5),
        content=TableContent(
            html="<table><tr><th>Gene</th><th>Expression</th></tr></table>",
            json_data={"headers": ["Gene", "Expression"], "rows": [{"Gene": "BRCA1", "Expression": "3.2"}]},
        ),
        confidence=0.92,
        extraction_method="docling",
    )


def test_semantic_issue_model():
    si = SemanticIssue(
        region_id="r1",
        description="Table shows mortality increasing but text says it decreased",
        confidence_penalty=0.15,
    )
    assert si.region_id == "r1"
    assert si.confidence_penalty == 0.15


@pytest.mark.asyncio
async def test_semantic_validation_no_issues():
    """When Claude finds no issues, return empty list."""
    mock_claude = AsyncMock()
    mock_claude.send_vision_request.return_value = VLMResponse(
        content={"issues": []},
        confidence=0.95,
        model="claude-opus-4-20250514",
        usage_tokens=300,
        duration_ms=2000,
    )

    regions = [
        _make_text_region("r1", "Gene expression of BRCA1 was elevated."),
        _make_table_region("t1"),
    ]

    issues = await validate_semantics(regions, claude_client=mock_claude)
    assert issues == []
    mock_claude.send_vision_request.assert_called_once()


@pytest.mark.asyncio
async def test_semantic_validation_finds_issues():
    """When Claude flags inconsistencies, return SemanticIssue objects."""
    mock_claude = AsyncMock()
    mock_claude.send_vision_request.return_value = VLMResponse(
        content={
            "issues": [
                {
                    "region_id": "r1",
                    "description": "Text says mortality decreased by 15% but Table 1 shows it increased",
                    "confidence_penalty": 0.20,
                },
                {
                    "region_id": "t1",
                    "description": "P-value column contains non-numeric value 'N/A' without explanation",
                    "confidence_penalty": 0.05,
                },
            ],
        },
        confidence=0.88,
        model="claude-opus-4-20250514",
        usage_tokens=500,
        duration_ms=3000,
    )

    regions = [
        _make_text_region("r1", "Mortality decreased by 15%."),
        _make_table_region("t1"),
    ]

    issues = await validate_semantics(regions, claude_client=mock_claude)
    assert len(issues) == 2
    assert issues[0].region_id == "r1"
    assert issues[0].confidence_penalty == 0.20
    assert "mortality" in issues[0].description.lower()


@pytest.mark.asyncio
async def test_semantic_validation_claude_failure():
    """If Claude fails, return empty list (graceful degradation)."""
    mock_claude = AsyncMock()
    mock_claude.send_vision_request.side_effect = RuntimeError("API error")

    regions = [_make_text_region("r1", "Some text")]

    issues = await validate_semantics(regions, claude_client=mock_claude)
    assert issues == []


@pytest.mark.asyncio
async def test_semantic_validation_without_claude():
    """Without a Claude client, skip semantic validation entirely."""
    regions = [_make_text_region("r1", "Some text")]
    issues = await validate_semantics(regions, claude_client=None)
    assert issues == []


@pytest.mark.asyncio
async def test_semantic_validation_single_call_for_all_regions():
    """Semantic validation uses a SINGLE Claude call per document, not per field."""
    mock_claude = AsyncMock()
    mock_claude.send_vision_request.return_value = VLMResponse(
        content={"issues": []},
        confidence=0.95,
        model="claude-opus-4-20250514",
        usage_tokens=400,
        duration_ms=2500,
    )

    regions = [
        _make_text_region("r1", "First paragraph."),
        _make_text_region("r2", "Second paragraph."),
        _make_text_region("r3", "Third paragraph."),
        _make_table_region("t1"),
    ]

    await validate_semantics(regions, claude_client=mock_claude)
    # Must be exactly ONE call, not one per region
    assert mock_claude.send_vision_request.call_count == 1
