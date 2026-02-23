# src/agentic_extract/specialists/table.py
"""Table Specialist: Docling + Claude reasoning + Codex schema enforcement.

Follows the OCR-then-LLM pattern:
1. Docling extracts HTML table structure (97.9% accuracy on complex tables)
2. Claude reasons about merged cells, ambiguous headers, multi-page tables
3. Codex enforces the JSON output schema via Structured Outputs API
"""
from __future__ import annotations

import json
import logging
import pathlib
from dataclasses import dataclass, field
from typing import Any

from agentic_extract.clients.vlm import VLMClient, VLMResponse
from agentic_extract.models import BoundingBox, Region, RegionType, TableContent
from agentic_extract.tools.docker_runner import DockerTool

logger = logging.getLogger(__name__)

CLAUDE_TABLE_PROMPT = """You are a document table extraction expert. You are given:
1. An HTML table extracted by Docling from a document
2. The original image of the table region

Your task: Verify the table extraction and identify any errors.
Check for: merged cells, ambiguous headers, missing data, misaligned columns.

HTML table from Docling:
{html_table}

Return JSON:
{{
  "corrections": [list of correction descriptions],
  "verified": true/false (true if table looks correct)
}}
"""

CODEX_TABLE_SCHEMA = {
    "type": "object",
    "properties": {
        "headers": {
            "type": "array",
            "items": {"type": "string"},
        },
        "rows": {
            "type": "array",
            "items": {"type": "object"},
        },
    },
    "required": ["headers", "rows"],
}

CODEX_TABLE_PROMPT = """Extract the table data from this image into structured JSON.
Use the exact column headers visible in the table.
Each row should be an object mapping header names to cell values.
Return null for any cell that is empty or unreadable.
"""


@dataclass
class DoclingResult:
    """Raw Docling table extraction result."""

    html: str
    json_data: dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0


class DoclingTool:
    """Docling Docker wrapper for table structure extraction."""

    IMAGE_NAME = "docling:latest"

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
            default_timeout=180,
            volumes=volumes,
        )

    def extract(self, image_path: pathlib.Path) -> DoclingResult:
        """Run Docling on a table image.

        Args:
            image_path: Path to the cropped table region image.

        Returns:
            DoclingResult with HTML table and JSON data.

        Raises:
            RuntimeError: If Docling container fails.
        """
        tool = self._docker_tool(self._image_name, self._volumes)
        result = tool.run([
            "--input", str(image_path),
            "--format", "json",
        ])

        if result.exit_code != 0:
            raise RuntimeError(
                f"Docling failed (exit {result.exit_code}): {result.stderr}"
            )

        data = json.loads(result.stdout)
        return DoclingResult(
            html=data.get("html", ""),
            json_data=data.get("json", {}),
            confidence=data.get("confidence", 0.0),
        )


class TableSpecialist:
    """Table extraction specialist: Docling + Claude + Codex.

    Pipeline:
    1. Docling extracts HTML table structure
    2. Claude verifies and reasons about ambiguous cells
    3. Codex enforces JSON schema via Structured Outputs
    """

    def __init__(
        self,
        docling_tool: DoclingTool | None = None,
        claude_client: VLMClient | None = None,
        codex_client: VLMClient | None = None,
    ) -> None:
        self.docling_tool = docling_tool or DoclingTool()
        self.claude_client = claude_client
        self.codex_client = codex_client

    async def extract(
        self,
        image_path: pathlib.Path,
        region_id: str,
        page: int,
        bbox: BoundingBox,
    ) -> Region:
        """Extract a table from a region image.

        Args:
            image_path: Path to the cropped table image.
            region_id: Unique identifier for this region.
            page: Page number (1-based).
            bbox: Normalized bounding box of this region.

        Returns:
            Region with TableContent populated.
        """
        # Stage 1: Docling
        docling_result = self.docling_tool.extract(image_path)
        extraction_method = "docling"
        final_html = docling_result.html
        final_json = docling_result.json_data
        final_confidence = docling_result.confidence

        # Stage 2: Claude reasoning about ambiguous cells
        if self.claude_client is not None:
            try:
                prompt = CLAUDE_TABLE_PROMPT.format(html_table=docling_result.html)
                claude_resp = await self.claude_client.send_vision_request(
                    image_path=image_path,
                    prompt=prompt,
                )
                extraction_method += f" + {claude_resp.model}"
                # If Claude verified the table, boost confidence slightly
                if isinstance(claude_resp.content, dict):
                    if claude_resp.content.get("verified", False):
                        final_confidence = max(
                            final_confidence, claude_resp.confidence,
                        )
            except Exception as exc:
                logger.warning(
                    "Claude table verification failed for %s: %s",
                    region_id, exc,
                )

        # Stage 3: Codex schema enforcement
        if self.codex_client is not None:
            try:
                codex_resp = await self.codex_client.send_vision_request(
                    image_path=image_path,
                    prompt=CODEX_TABLE_PROMPT,
                    schema=CODEX_TABLE_SCHEMA,
                )
                if isinstance(codex_resp.content, dict):
                    if "headers" in codex_resp.content:
                        final_json = codex_resp.content
                        final_confidence = max(
                            final_confidence, codex_resp.confidence,
                        )
                extraction_method += f" + {codex_resp.model}"
            except Exception as exc:
                logger.warning(
                    "Codex schema enforcement failed for %s: %s",
                    region_id, exc,
                )

        return Region(
            id=region_id,
            type=RegionType.TABLE,
            subtype=None,
            page=page,
            bbox=bbox,
            content=TableContent(
                html=final_html,
                json_data=final_json,
                cell_bboxes=[],
            ),
            confidence=final_confidence,
            extraction_method=extraction_method,
            needs_review=final_confidence < 0.90,
            review_reason=(
                f"Table confidence {final_confidence:.2f} below 0.90 threshold"
                if final_confidence < 0.90
                else None
            ),
        )
