# tests/specialists/test_visual_chart.py
"""Tests for the Visual Specialist chart mode (DePlot + Claude reasoning)."""
import json
import pathlib
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from PIL import Image
from pydantic import ValidationError

from agentic_extract.clients.vlm import VLMResponse
from agentic_extract.models import (
    BoundingBox,
    ChartAxis,
    ChartContent,
    DataSeries,
    Region,
    RegionType,
)
from agentic_extract.specialists.visual_chart import (
    ChartSpecialist,
    DeplotResult,
    DeplotTool,
)
from agentic_extract.tools.docker_runner import ToolOutput


def test_chart_axis_model():
    axis = ChartAxis(label="Year", type="temporal")
    assert axis.label == "Year"
    assert axis.type == "temporal"


def test_data_series_model():
    ds = DataSeries(name="Revenue", values=[100.0, 150.0, 200.0])
    assert ds.name == "Revenue"
    assert len(ds.values) == 3


def test_data_series_allows_mixed_values():
    ds = DataSeries(name="Mixed", values=[1.0, None, "N/A"])
    assert ds.values[1] is None
    assert ds.values[2] == "N/A"


def test_chart_content_model():
    cc = ChartContent(
        figure_type="bar_chart",
        title="Gene Expression",
        x_axis=ChartAxis(label="Condition", type="categorical"),
        y_axis=ChartAxis(label="Fold Change", type="numerical"),
        data_series=[DataSeries(name="BRCA1", values=[1.0, 3.2, 4.1])],
        description="Bar chart showing gene expression levels.",
    )
    assert cc.figure_type == "bar_chart"
    assert cc.title == "Gene Expression"
    assert len(cc.data_series) == 1
    assert cc.x_axis.label == "Condition"


def test_chart_content_minimal():
    cc = ChartContent(
        figure_type="unknown",
        description="A chart.",
    )
    assert cc.title is None
    assert cc.x_axis is None
    assert cc.data_series == []


def test_deplot_result_dataclass():
    result = DeplotResult(
        raw_table="Year | Revenue\n2020 | 100\n2021 | 150",
        confidence=0.88,
    )
    assert "Revenue" in result.raw_table
    assert result.confidence == 0.88


@patch.object(DeplotTool, "_docker_tool")
def test_deplot_extracts_chart_data(mock_tool: MagicMock, tmp_path: pathlib.Path):
    img = Image.new("RGB", (600, 400), "white")
    img_path = tmp_path / "chart.png"
    img.save(img_path)

    deplot_output = json.dumps({
        "table": "Year | Revenue\n2020 | 100\n2021 | 150\n2022 | 200",
        "confidence": 0.87,
    })
    mock_tool.return_value.run.return_value = ToolOutput(
        stdout=deplot_output, stderr="", exit_code=0, duration_ms=900,
    )

    tool = DeplotTool()
    result = tool.extract(img_path)

    assert "Revenue" in result.raw_table
    assert result.confidence == 0.87


@patch.object(DeplotTool, "_docker_tool")
def test_deplot_handles_failure(mock_tool: MagicMock, tmp_path: pathlib.Path):
    img = Image.new("RGB", (100, 100), "white")
    img_path = tmp_path / "bad_chart.png"
    img.save(img_path)

    mock_tool.return_value.run.return_value = ToolOutput(
        stdout="", stderr="DePlot segfault", exit_code=139, duration_ms=50,
    )

    tool = DeplotTool()
    with pytest.raises(RuntimeError, match="DePlot failed"):
        tool.extract(img_path)


@pytest.mark.asyncio
async def test_chart_specialist_full_pipeline(tmp_path: pathlib.Path):
    """DePlot extracts table, Claude interprets chart structure."""
    img = Image.new("RGB", (600, 400), "white")
    img_path = tmp_path / "chart_full.png"
    img.save(img_path)

    mock_deplot = MagicMock()
    mock_deplot.extract.return_value = DeplotResult(
        raw_table="Condition | BRCA1 | TP53\nControl | 1.0 | 1.0\nTreatment | 3.2 | 1.8",
        confidence=0.85,
    )

    mock_claude = AsyncMock()
    mock_claude.send_vision_request.return_value = VLMResponse(
        content={
            "figure_type": "bar_chart",
            "title": "Gene Expression Across Conditions",
            "x_axis": {"label": "Condition", "type": "categorical"},
            "y_axis": {"label": "Fold Change", "type": "numerical"},
            "data_series": [
                {"name": "BRCA1", "values": [1.0, 3.2]},
                {"name": "TP53", "values": [1.0, 1.8]},
            ],
            "description": "Bar chart comparing gene expression.",
        },
        confidence=0.91,
        model="claude-opus-4-20250514",
        usage_tokens=400,
        duration_ms=2500,
    )

    specialist = ChartSpecialist(
        deplot_tool=mock_deplot, claude_client=mock_claude,
    )
    region = await specialist.extract(
        image_path=img_path,
        region_id="c1",
        page=3,
        bbox=BoundingBox(x=0.1, y=0.05, w=0.8, h=0.45),
    )

    assert isinstance(region, Region)
    assert region.type == RegionType.FIGURE
    assert region.subtype == "bar_chart"
    assert isinstance(region.content, ChartContent)
    assert region.content.figure_type == "bar_chart"
    assert len(region.content.data_series) == 2
    assert "deplot" in region.extraction_method
    assert "claude" in region.extraction_method
    mock_claude.send_vision_request.assert_called_once()


@pytest.mark.asyncio
async def test_chart_specialist_claude_failure_fallback(tmp_path: pathlib.Path):
    """If Claude fails, fall back to raw DePlot output."""
    img = Image.new("RGB", (600, 400), "white")
    img_path = tmp_path / "chart_fallback.png"
    img.save(img_path)

    mock_deplot = MagicMock()
    mock_deplot.extract.return_value = DeplotResult(
        raw_table="X | Y\n1 | 10\n2 | 20",
        confidence=0.80,
    )

    mock_claude = AsyncMock()
    mock_claude.send_vision_request.side_effect = RuntimeError("API timeout")

    specialist = ChartSpecialist(
        deplot_tool=mock_deplot, claude_client=mock_claude,
    )
    region = await specialist.extract(
        image_path=img_path,
        region_id="c2",
        page=1,
        bbox=BoundingBox(x=0.0, y=0.0, w=1.0, h=1.0),
    )

    assert region.content.figure_type == "unknown"
    assert "X | Y" in region.content.description
    assert region.confidence == 0.80
    assert region.extraction_method == "deplot"


@pytest.mark.asyncio
async def test_chart_specialist_without_claude(tmp_path: pathlib.Path):
    """Chart specialist works with DePlot only when no VLM client provided."""
    img = Image.new("RGB", (600, 400), "white")
    img_path = tmp_path / "chart_no_vlm.png"
    img.save(img_path)

    mock_deplot = MagicMock()
    mock_deplot.extract.return_value = DeplotResult(
        raw_table="A | B\n1 | 2",
        confidence=0.75,
    )

    specialist = ChartSpecialist(deplot_tool=mock_deplot, claude_client=None)
    region = await specialist.extract(
        image_path=img_path,
        region_id="c3",
        page=1,
        bbox=BoundingBox(x=0.0, y=0.0, w=1.0, h=1.0),
    )

    assert region.confidence == 0.75
    assert region.extraction_method == "deplot"
