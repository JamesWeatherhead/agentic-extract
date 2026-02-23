# tests/specialists/test_table.py
"""Tests for the Table Specialist (Docling + Claude + Codex)."""
import json
import pathlib
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from PIL import Image

from agentic_extract.clients.vlm import VLMResponse
from agentic_extract.models import BoundingBox, Region, RegionType, TableContent
from agentic_extract.specialists.table import (
    DoclingTool,
    DoclingResult,
    TableSpecialist,
)
from agentic_extract.tools.docker_runner import ToolOutput


def test_docling_result_dataclass():
    result = DoclingResult(
        html="<table><tr><td>A</td></tr></table>",
        json_data={"headers": ["Col"], "rows": [{"Col": "A"}]},
        confidence=0.95,
    )
    assert result.html.startswith("<table>")


@patch.object(DoclingTool, "_docker_tool")
def test_docling_extracts_table(mock_tool: MagicMock, tmp_path: pathlib.Path):
    img = Image.new("RGB", (800, 400), "white")
    img_path = tmp_path / "table.png"
    img.save(img_path)

    docling_output = json.dumps({
        "html": "<table><tr><th>Gene</th><th>Value</th></tr><tr><td>BRCA1</td><td>3.2</td></tr></table>",
        "json": {"headers": ["Gene", "Value"], "rows": [{"Gene": "BRCA1", "Value": "3.2"}]},
        "confidence": 0.94,
    })
    mock_tool.return_value.run.return_value = ToolOutput(
        stdout=docling_output, stderr="", exit_code=0, duration_ms=1200,
    )

    tool = DoclingTool()
    result = tool.extract(img_path)
    assert "BRCA1" in result.html
    assert result.confidence == 0.94


@patch.object(DoclingTool, "_docker_tool")
def test_docling_handles_failure(mock_tool: MagicMock, tmp_path: pathlib.Path):
    img = Image.new("RGB", (100, 100), "white")
    img_path = tmp_path / "bad_table.png"
    img.save(img_path)

    mock_tool.return_value.run.return_value = ToolOutput(
        stdout="", stderr="Docling error", exit_code=1, duration_ms=50,
    )

    tool = DoclingTool()
    with pytest.raises(RuntimeError, match="Docling failed"):
        tool.extract(img_path)


@pytest.mark.asyncio
async def test_table_specialist_full_pipeline(tmp_path: pathlib.Path):
    """Test the full OCR-then-LLM pipeline: Docling -> Claude -> Codex."""
    img = Image.new("RGB", (800, 400), "white")
    img_path = tmp_path / "table_full.png"
    img.save(img_path)

    mock_docling = MagicMock()
    mock_docling.extract.return_value = DoclingResult(
        html="<table><tr><td>A</td><td>B</td></tr></table>",
        json_data={"headers": ["A", "B"], "rows": []},
        confidence=0.91,
    )

    mock_claude = AsyncMock()
    mock_claude.send_vision_request.return_value = VLMResponse(
        content={"corrections": [], "verified": True},
        confidence=0.93,
        model="claude-opus-4-20250514",
        usage_tokens=300,
        duration_ms=2000,
    )

    mock_codex = AsyncMock()
    mock_codex.send_vision_request.return_value = VLMResponse(
        content={"headers": ["A", "B"], "rows": [{"A": "1", "B": "2"}]},
        confidence=0.95,
        model="gpt-4o",
        usage_tokens=250,
        duration_ms=1500,
    )

    specialist = TableSpecialist(
        docling_tool=mock_docling,
        claude_client=mock_claude,
        codex_client=mock_codex,
    )

    region = await specialist.extract(
        image_path=img_path,
        region_id="t1",
        page=2,
        bbox=BoundingBox(x=0.05, y=0.2, w=0.9, h=0.4),
    )

    assert isinstance(region, Region)
    assert region.type == RegionType.TABLE
    assert isinstance(region.content, TableContent)
    assert region.content.html is not None
    assert "docling" in region.extraction_method
    mock_claude.send_vision_request.assert_called_once()
    mock_codex.send_vision_request.assert_called_once()


@pytest.mark.asyncio
async def test_table_specialist_without_codex(tmp_path: pathlib.Path):
    """Table specialist should work even without Codex (just Docling + Claude)."""
    img = Image.new("RGB", (800, 400), "white")
    img_path = tmp_path / "table_no_codex.png"
    img.save(img_path)

    mock_docling = MagicMock()
    mock_docling.extract.return_value = DoclingResult(
        html="<table><tr><td>X</td></tr></table>",
        json_data={"headers": ["X"], "rows": [{"X": "1"}]},
        confidence=0.88,
    )

    mock_claude = AsyncMock()
    mock_claude.send_vision_request.return_value = VLMResponse(
        content={"corrections": [], "verified": True},
        confidence=0.90,
        model="claude-opus-4-20250514",
        usage_tokens=200,
        duration_ms=1000,
    )

    specialist = TableSpecialist(
        docling_tool=mock_docling,
        claude_client=mock_claude,
        codex_client=None,
    )

    region = await specialist.extract(
        image_path=img_path,
        region_id="t2",
        page=1,
        bbox=BoundingBox(x=0.0, y=0.0, w=1.0, h=1.0),
    )

    assert region.content.json_data["headers"] == ["X"]
    assert "codex" not in region.extraction_method.lower()


@pytest.mark.asyncio
async def test_table_specialist_vlm_failures_degrade_gracefully(tmp_path: pathlib.Path):
    """If both Claude and Codex fail, fall back to raw Docling output."""
    img = Image.new("RGB", (800, 400), "white")
    img_path = tmp_path / "table_fallback.png"
    img.save(img_path)

    mock_docling = MagicMock()
    mock_docling.extract.return_value = DoclingResult(
        html="<table><tr><td>Z</td></tr></table>",
        json_data={"headers": ["Z"], "rows": [{"Z": "9"}]},
        confidence=0.85,
    )

    mock_claude = AsyncMock()
    mock_claude.send_vision_request.side_effect = RuntimeError("API error")

    mock_codex = AsyncMock()
    mock_codex.send_vision_request.side_effect = RuntimeError("API error")

    specialist = TableSpecialist(
        docling_tool=mock_docling,
        claude_client=mock_claude,
        codex_client=mock_codex,
    )

    region = await specialist.extract(
        image_path=img_path,
        region_id="t3",
        page=1,
        bbox=BoundingBox(x=0.0, y=0.0, w=1.0, h=1.0),
    )

    assert region.content.html == "<table><tr><td>Z</td></tr></table>"
    assert region.confidence == 0.85
    assert region.extraction_method == "docling"
