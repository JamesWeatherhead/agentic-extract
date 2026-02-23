# src/agentic_extract/specialists/visual_chart.py
"""Visual Specialist - Chart Mode: DePlot + Claude chart reasoning.

Follows the OCR-then-LLM pattern:
1. DePlot converts chart images to linearized table data
2. Claude interprets chart structure (type, axes, data series)
3. Returns structured ChartContent with full chart metadata
"""
from __future__ import annotations

import json
import logging
import pathlib
from dataclasses import dataclass

from agentic_extract.clients.vlm import VLMClient, VLMResponse
from agentic_extract.models import (
    BoundingBox,
    ChartAxis,
    ChartContent,
    DataSeries,
    Region,
    RegionType,
)
from agentic_extract.tools.docker_runner import DockerTool

logger = logging.getLogger(__name__)

CLAUDE_CHART_PROMPT = """You are a scientific chart interpretation expert. You are given:
1. A linearized table extracted from a chart by DePlot
2. The original chart image

Your task: Interpret the chart and produce structured metadata.

DePlot table output:
{deplot_table}

Return JSON:
{{
  "figure_type": "bar_chart|line_chart|scatter_plot|pie_chart|histogram|box_plot|heatmap|other",
  "title": "chart title or null if not visible",
  "x_axis": {{"label": "axis label", "type": "categorical|numerical|temporal"}},
  "y_axis": {{"label": "axis label", "type": "categorical|numerical|temporal"}},
  "data_series": [
    {{"name": "series name", "values": [1.0, 2.0, 3.0]}}
  ],
  "description": "One-sentence description of what the chart shows"
}}

Rules:
- Use null for any field you cannot determine from the image
- Use "unknown" for figure_type if unclear
- Preserve numerical precision from the DePlot table
- List ALL data series visible in the chart
"""


@dataclass
class DeplotResult:
    """Raw DePlot chart-to-table extraction result."""

    raw_table: str
    confidence: float


class DeplotTool:
    """DePlot (Google, 282M) Docker wrapper for chart-to-table extraction.

    DePlot converts chart images into linearized table representations,
    achieving 29.4% improvement over prior SOTA on ChartQA.
    """

    IMAGE_NAME = "deplot:latest"

    def __init__(
        self,
        image_name: str | None = None,
        volumes: dict[str, str] | None = None,
    ) -> None:
        self._image_name = image_name or self.IMAGE_NAME
        self._volumes = volumes or {}

    @staticmethod
    def _docker_tool(
        image_name: str, volumes: dict[str, str],
    ) -> DockerTool:
        return DockerTool(
            image_name=image_name,
            default_timeout=120,
            volumes=volumes,
        )

    def extract(self, image_path: pathlib.Path) -> DeplotResult:
        """Run DePlot on a chart image.

        Args:
            image_path: Path to the cropped chart region image.

        Returns:
            DeplotResult with linearized table and confidence.

        Raises:
            RuntimeError: If DePlot container fails.
        """
        tool = self._docker_tool(self._image_name, self._volumes)
        result = tool.run(["--input", str(image_path), "--format", "json"])

        if result.exit_code != 0:
            raise RuntimeError(
                f"DePlot failed (exit {result.exit_code}): {result.stderr}"
            )

        data = json.loads(result.stdout)
        return DeplotResult(
            raw_table=data.get("table", ""),
            confidence=data.get("confidence", 0.0),
        )


class ChartSpecialist:
    """Chart extraction specialist: DePlot + Claude reasoning.

    Pipeline:
    1. DePlot extracts linearized table from chart image
    2. Claude interprets chart structure and data series
    3. Falls back to raw DePlot table if Claude fails
    """

    def __init__(
        self,
        deplot_tool: DeplotTool | None = None,
        claude_client: VLMClient | None = None,
    ) -> None:
        self.deplot_tool = deplot_tool or DeplotTool()
        self.claude_client = claude_client

    async def extract(
        self,
        image_path: pathlib.Path,
        region_id: str,
        page: int,
        bbox: BoundingBox,
    ) -> Region:
        """Extract chart data from a region image.

        Args:
            image_path: Path to the cropped chart image.
            region_id: Unique identifier for this region.
            page: Page number (1-based).
            bbox: Normalized bounding box of this region.

        Returns:
            Region with ChartContent populated.
        """
        # Stage 1: DePlot
        deplot_result = self.deplot_tool.extract(image_path)
        extraction_method = "deplot"
        final_confidence = deplot_result.confidence

        # Default chart content (from raw DePlot only)
        chart_content = ChartContent(
            figure_type="unknown",
            description=deplot_result.raw_table,
        )

        # Stage 2: Claude chart reasoning
        if self.claude_client is not None:
            try:
                prompt = CLAUDE_CHART_PROMPT.format(
                    deplot_table=deplot_result.raw_table,
                )
                vlm_response = await self.claude_client.send_vision_request(
                    image_path=image_path,
                    prompt=prompt,
                )
                content = vlm_response.content
                if isinstance(content, dict):
                    x_axis = None
                    y_axis = None
                    if content.get("x_axis"):
                        x_axis = ChartAxis(**content["x_axis"])
                    if content.get("y_axis"):
                        y_axis = ChartAxis(**content["y_axis"])

                    series = []
                    for s in content.get("data_series", []):
                        series.append(DataSeries(
                            name=s.get("name", ""),
                            values=s.get("values", []),
                        ))

                    chart_content = ChartContent(
                        figure_type=content.get("figure_type", "unknown"),
                        title=content.get("title"),
                        x_axis=x_axis,
                        y_axis=y_axis,
                        data_series=series,
                        description=content.get("description", ""),
                    )
                    final_confidence = max(
                        final_confidence, vlm_response.confidence,
                    )
                    extraction_method += f" + {vlm_response.model}"
            except Exception as exc:
                logger.warning(
                    "Claude chart reasoning failed for %s, using DePlot fallback: %s",
                    region_id, exc,
                )

        return Region(
            id=region_id,
            type=RegionType.FIGURE,
            subtype=chart_content.figure_type,
            page=page,
            bbox=bbox,
            content=chart_content,
            confidence=final_confidence,
            extraction_method=extraction_method,
            needs_review=final_confidence < 0.90,
            review_reason=(
                f"Chart confidence {final_confidence:.2f} below 0.90 threshold"
                if final_confidence < 0.90
                else None
            ),
        )
