# src/agentic_extract/specialists/visual_figure.py
"""Visual Specialist - Figure Mode: FigEx2 + DECIMER + GelGenie + classifiers.

Handles multi-panel splitting, domain-specific figure extraction, and
general figure interpretation via dual-model (Codex for figure matching,
Claude for reasoning).

Figure type classification:
1. Deterministic: keyword matching on caption text
2. Claude fallback: VLM-based classification for ambiguous cases
"""
from __future__ import annotations

import json
import logging
import pathlib
from dataclasses import dataclass, field
from typing import Any

from agentic_extract.clients.vlm import VLMClient, VLMResponse
from agentic_extract.models import (
    BoundingBox,
    FigureContent,
    Region,
    RegionType,
)
from agentic_extract.tools.docker_runner import DockerTool

logger = logging.getLogger(__name__)

# Keywords for deterministic figure type classification
MOLECULAR_KEYWORDS = {
    "molecular", "molecule", "chemical", "structure", "compound",
    "smiles", "inchi", "formula", "reagent", "synthesis",
}
GEL_KEYWORDS = {
    "gel", "electrophoresis", "western blot", "blot", "sds-page",
    "agarose", "band", "ladder", "marker", "kda",
}

CODEX_FIGURE_PROMPT = """Identify the elements in this scientific figure.
Return JSON:
{{
  "figure_type": "type of figure (molecular_structure, gel, pathway, diagram, micrograph, photo, other)",
  "elements": ["list of key visual elements identified"]
}}
"""

CLAUDE_FIGURE_PROMPT = """Describe this scientific figure concisely.
Return JSON:
{{
  "description": "One-paragraph description of the figure content and significance"
}}
"""

CLAUDE_CLASSIFY_PROMPT = """Classify this scientific figure into one of these types:
- molecular: chemical/molecular structures
- gel: gel electrophoresis, western blots, band patterns
- general: diagrams, photographs, micrographs, pathways, other

Caption: {caption}

Return JSON: {{"figure_type": "molecular|gel|general"}}
"""


@dataclass
class FigEx2Result:
    """Result from FigEx2 multi-panel figure splitting."""

    panel_paths: list[pathlib.Path]
    panel_labels: list[str]
    confidence: float


@dataclass
class DecimerResult:
    """Result from DECIMER molecular structure recognition."""

    smiles: str
    inchi: str
    confidence: float


@dataclass
class GelGenieResult:
    """Result from GelGenie gel electrophoresis analysis."""

    bands: list[dict[str, Any]]
    lane_count: int
    confidence: float


class FigEx2Tool:
    """FigEx2 Docker wrapper for multi-panel figure splitting.

    Cross-domain few-shot with reward-augmented training (Jan 2026).
    """

    IMAGE_NAME = "figex2:latest"

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

    def split(
        self,
        image_path: pathlib.Path,
        output_dir: pathlib.Path,
    ) -> FigEx2Result:
        """Split a multi-panel figure into individual panels.

        Args:
            image_path: Path to the figure image.
            output_dir: Directory to write panel images.

        Returns:
            FigEx2Result with paths to panel images.

        Raises:
            RuntimeError: If FigEx2 container fails.
        """
        tool = self._docker_tool(self._image_name, self._volumes)
        result = tool.run([
            "--input", str(image_path),
            "--output_dir", str(output_dir),
            "--format", "json",
        ])

        if result.exit_code != 0:
            raise RuntimeError(
                f"FigEx2 failed (exit {result.exit_code}): {result.stderr}"
            )

        data = json.loads(result.stdout)
        panels = data.get("panels", [])
        return FigEx2Result(
            panel_paths=[pathlib.Path(p["path"]) for p in panels],
            panel_labels=[p.get("label", "") for p in panels],
            confidence=data.get("confidence", 0.0),
        )


class DecimerTool:
    """DECIMER.ai Docker wrapper for chemical structure recognition.

    Converts molecular structure images to SMILES/InChI notation.
    Nature Communications 2023; production-proven.
    """

    IMAGE_NAME = "decimer:latest"

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

    def extract(self, image_path: pathlib.Path) -> DecimerResult:
        """Extract molecular structure as SMILES/InChI.

        Args:
            image_path: Path to the molecular structure image.

        Returns:
            DecimerResult with SMILES and InChI strings.

        Raises:
            RuntimeError: If DECIMER container fails.
        """
        tool = self._docker_tool(self._image_name, self._volumes)
        result = tool.run(["--input", str(image_path), "--format", "json"])

        if result.exit_code != 0:
            raise RuntimeError(
                f"DECIMER failed (exit {result.exit_code}): {result.stderr}"
            )

        data = json.loads(result.stdout)
        return DecimerResult(
            smiles=data.get("smiles", ""),
            inchi=data.get("inchi", ""),
            confidence=data.get("confidence", 0.0),
        )


class GelGenieTool:
    """GelGenie Docker wrapper for gel electrophoresis analysis.

    AI-powered band identification. Nature Communications 2025.
    """

    IMAGE_NAME = "gelgenie:latest"

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

    def extract(self, image_path: pathlib.Path) -> GelGenieResult:
        """Identify bands in a gel electrophoresis image.

        Args:
            image_path: Path to the gel image.

        Returns:
            GelGenieResult with band data and lane count.

        Raises:
            RuntimeError: If GelGenie container fails.
        """
        tool = self._docker_tool(self._image_name, self._volumes)
        result = tool.run(["--input", str(image_path), "--format", "json"])

        if result.exit_code != 0:
            raise RuntimeError(
                f"GelGenie failed (exit {result.exit_code}): {result.stderr}"
            )

        data = json.loads(result.stdout)
        return GelGenieResult(
            bands=data.get("bands", []),
            lane_count=data.get("lane_count", 0),
            confidence=data.get("confidence", 0.0),
        )


class FigureTypeClassifier:
    """Deterministic figure type classifier with Claude fallback.

    First tries keyword matching on caption text. If that yields
    'general' (no match), optionally calls Claude for VLM-based
    classification.
    """

    def classify_deterministic(self, caption: str) -> str:
        """Classify figure type from caption text using keyword matching.

        Returns: 'molecular', 'gel', or 'general'.
        """
        caption_lower = caption.lower()
        for kw in MOLECULAR_KEYWORDS:
            if kw in caption_lower:
                return "molecular"
        for kw in GEL_KEYWORDS:
            if kw in caption_lower:
                return "gel"
        return "general"

    async def classify(
        self,
        caption: str,
        image_path: pathlib.Path,
        claude_client: VLMClient | None = None,
    ) -> str:
        """Classify figure type with optional Claude fallback.

        Args:
            caption: Figure caption text.
            image_path: Path to the figure image.
            claude_client: Optional Claude client for ambiguous cases.

        Returns:
            Figure type string: 'molecular', 'gel', or 'general'.
        """
        det_result = self.classify_deterministic(caption)
        if det_result != "general" or claude_client is None:
            return det_result

        # Ambiguous: use Claude
        try:
            prompt = CLAUDE_CLASSIFY_PROMPT.format(caption=caption)
            resp = await claude_client.send_vision_request(
                image_path=image_path,
                prompt=prompt,
            )
            if isinstance(resp.content, dict):
                fig_type = resp.content.get("figure_type", "general")
                if fig_type in ("molecular", "gel", "general"):
                    return fig_type
        except Exception as exc:
            logger.warning("Claude figure classification failed: %s", exc)

        return det_result


class FigureSpecialist:
    """Figure extraction specialist: FigEx2 + domain tools + dual-model.

    Pipeline:
    1. FigEx2 splits multi-panel figures into individual panels
    2. Classifier determines figure type per panel
    3. Domain tool extracts structured data:
       - Molecular: DECIMER for SMILES/InChI
       - Gel: GelGenie for band identification
       - General: VLM description
    4. Codex for figure matching (SciFIBench advantage)
    5. Claude for scientific reasoning and interpretation
    """

    def __init__(
        self,
        figex2_tool: FigEx2Tool | None = None,
        decimer_tool: DecimerTool | None = None,
        gelgenie_tool: GelGenieTool | None = None,
        classifier: FigureTypeClassifier | None = None,
        codex_client: VLMClient | None = None,
        claude_client: VLMClient | None = None,
    ) -> None:
        self.figex2_tool = figex2_tool or FigEx2Tool()
        self.decimer_tool = decimer_tool or DecimerTool()
        self.gelgenie_tool = gelgenie_tool or GelGenieTool()
        self.classifier = classifier or FigureTypeClassifier()
        self.codex_client = codex_client
        self.claude_client = claude_client

    async def _extract_molecular(
        self, image_path: pathlib.Path,
    ) -> tuple[dict[str, Any], float, str]:
        """Extract molecular structure data."""
        result = self.decimer_tool.extract(image_path)
        figure_json: dict[str, Any] = {
            "smiles": result.smiles,
            "inchi": result.inchi,
        }
        return figure_json, result.confidence, "decimer"

    async def _extract_gel(
        self, image_path: pathlib.Path,
    ) -> tuple[dict[str, Any], float, str]:
        """Extract gel electrophoresis data."""
        result = self.gelgenie_tool.extract(image_path)
        figure_json: dict[str, Any] = {
            "bands": result.bands,
            "lane_count": result.lane_count,
        }
        return figure_json, result.confidence, "gelgenie"

    async def _extract_general(
        self, image_path: pathlib.Path,
    ) -> tuple[dict[str, Any], float, str]:
        """Extract general figure data via VLM."""
        figure_json: dict[str, Any] = {}
        confidence = 0.5
        method = "visual"

        # Codex for figure matching (SciFIBench: 75.4%)
        if self.codex_client is not None:
            try:
                codex_resp = await self.codex_client.send_vision_request(
                    image_path=image_path,
                    prompt=CODEX_FIGURE_PROMPT,
                )
                if isinstance(codex_resp.content, dict):
                    figure_json.update(codex_resp.content)
                    confidence = max(confidence, codex_resp.confidence)
                    method += f" + {codex_resp.model}"
            except Exception as exc:
                logger.warning("Codex figure matching failed: %s", exc)

        # Claude for reasoning (CharXiv: ~60%)
        if self.claude_client is not None:
            try:
                claude_resp = await self.claude_client.send_vision_request(
                    image_path=image_path,
                    prompt=CLAUDE_FIGURE_PROMPT,
                )
                if isinstance(claude_resp.content, dict):
                    figure_json["description"] = claude_resp.content.get(
                        "description", "",
                    )
                    confidence = max(confidence, claude_resp.confidence)
                    method += f" + {claude_resp.model}"
            except Exception as exc:
                logger.warning("Claude figure reasoning failed: %s", exc)

        return figure_json, confidence, method

    async def extract(
        self,
        image_path: pathlib.Path,
        region_id: str,
        page: int,
        bbox: BoundingBox,
        caption: str = "",
    ) -> Region:
        """Extract figure data from a region image.

        Args:
            image_path: Path to the cropped figure image.
            region_id: Unique identifier for this region.
            page: Page number (1-based).
            bbox: Normalized bounding box of this region.
            caption: Figure caption text (used for classification).

        Returns:
            Region with FigureContent populated.
        """
        extraction_method = ""
        figure_json: dict[str, Any] = {}
        description = caption
        figure_type = "general"
        final_confidence = 0.5

        # Stage 1: FigEx2 panel splitting
        try:
            figex_result = self.figex2_tool.split(
                image_path, output_dir=image_path.parent,
            )
            panels = figex_result.panel_paths
            extraction_method = "figex2"
        except Exception as exc:
            logger.warning("FigEx2 failed, treating as single panel: %s", exc)
            panels = [image_path]

        # Process first panel (primary; multi-panel recursive processing
        # would iterate all panels in production)
        panel_path = panels[0]

        # Stage 2: Classify figure type
        figure_type = await self.classifier.classify(
            caption=caption,
            image_path=panel_path,
            claude_client=self.claude_client,
        )

        # Stage 3: Route to domain-specific tool
        if figure_type == "molecular":
            try:
                figure_json, final_confidence, method = (
                    await self._extract_molecular(panel_path)
                )
                extraction_method += f" + {method}" if extraction_method else method
            except Exception as exc:
                logger.warning("Molecular extraction failed: %s", exc)
                figure_type = "general"

        if figure_type == "gel":
            try:
                figure_json, final_confidence, method = (
                    await self._extract_gel(panel_path)
                )
                extraction_method += f" + {method}" if extraction_method else method
            except Exception as exc:
                logger.warning("Gel extraction failed: %s", exc)
                figure_type = "general"

        if figure_type == "general":
            figure_json, final_confidence, method = (
                await self._extract_general(panel_path)
            )
            extraction_method += f" + {method}" if extraction_method else method

        # Extract description if available
        if "description" in figure_json:
            description = figure_json["description"]

        return Region(
            id=region_id,
            type=RegionType.FIGURE,
            subtype=figure_type,
            page=page,
            bbox=bbox,
            content=FigureContent(
                description=description,
                figure_type=figure_type,
                figure_json=figure_json,
            ),
            confidence=final_confidence,
            extraction_method=extraction_method or "visual",
            needs_review=final_confidence < 0.90,
            review_reason=(
                f"Figure confidence {final_confidence:.2f} below 0.90 threshold"
                if final_confidence < 0.90
                else None
            ),
        )
