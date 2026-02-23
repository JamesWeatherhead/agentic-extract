# src/agentic_extract/specialists/visual_handwriting.py
"""Visual Specialist - Handwriting Mode: TrOCR + DocEnTr + dual-model verification.

Follows the OCR-then-LLM pattern with dual-model cross-validation:
1. DocEnTr enhances degraded images (when flagged by quality assessment)
2. TrOCR extracts handwritten text with per-character confidence
3. Codex verifies OCR output (primary: better raw accuracy, edit distance 0.02)
4. Claude checks for hallucination (secondary: hallucination rate 0.09%)
5. Agreement between models boosts confidence; disagreement flags for review
"""
from __future__ import annotations

import json
import logging
import pathlib
from dataclasses import dataclass, field

from agentic_extract.clients.vlm import VLMClient, VLMResponse
from agentic_extract.models import (
    BoundingBox,
    HandwritingContent,
    Region,
    RegionType,
)
from agentic_extract.tools.docker_runner import DockerTool

logger = logging.getLogger(__name__)

CODEX_HANDWRITING_PROMPT = """You are an expert at reading handwritten text. You are given:
1. Raw OCR output from TrOCR (may contain errors)
2. The original handwriting image

Your task: Correct any OCR errors and produce accurate text.
Return JSON: {{"corrected_text": "the corrected text"}}

OCR output: {ocr_text}
OCR confidence: {ocr_confidence}

Rules:
- Fix character confusion (l/1, O/0, rn/m, etc.)
- Preserve original words and meaning
- Return null for corrected_text ONLY if completely unreadable
"""

CLAUDE_HANDWRITING_PROMPT = """You are a handwriting verification expert. You are given:
1. A candidate text transcription
2. The original handwriting image

Your task: Verify the transcription and check for hallucination.
Return JSON:
{{
  "verified_text": "the verified text (or corrected if errors found)",
  "hallucination_risk": "low|medium|high"
}}

Candidate text: {candidate_text}

Rules:
- If the candidate text matches the image, return it as-is
- If you see errors, correct them
- Flag hallucination_risk as "high" if the text appears fabricated
- Return null for verified_text ONLY if completely unreadable
"""


@dataclass
class TrOCRResult:
    """Raw TrOCR handwritten text recognition result."""

    text: str
    confidence: float
    per_char_confidences: list[float] = field(default_factory=list)


class TrOCRTool:
    """TrOCR (Microsoft) Docker wrapper for handwritten text recognition.

    State-of-the-art transformer HTR from the HuggingFace ecosystem.
    """

    IMAGE_NAME = "trocr:latest"

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

    def extract(self, image_path: pathlib.Path) -> TrOCRResult:
        """Run TrOCR on a handwriting image.

        Args:
            image_path: Path to the handwriting region image.

        Returns:
            TrOCRResult with text and confidence scores.

        Raises:
            RuntimeError: If TrOCR container fails.
        """
        tool = self._docker_tool(self._image_name, self._volumes)
        result = tool.run(["--input", str(image_path), "--format", "json"])

        if result.exit_code != 0:
            raise RuntimeError(
                f"TrOCR failed (exit {result.exit_code}): {result.stderr}"
            )

        data = json.loads(result.stdout)
        return TrOCRResult(
            text=data.get("text", ""),
            confidence=data.get("confidence", 0.0),
            per_char_confidences=data.get("per_char_confidences", []),
        )


class DocEnTrTool:
    """DocEnTr Docker wrapper for document image enhancement.

    Transformer-based cleaning, binarization, deblurring for degraded scans.
    """

    IMAGE_NAME = "docentr:latest"

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

    def enhance(
        self,
        image_path: pathlib.Path,
        output_dir: pathlib.Path,
    ) -> pathlib.Path:
        """Enhance a degraded document image.

        Args:
            image_path: Path to the degraded image.
            output_dir: Directory to write enhanced image.

        Returns:
            Path to the enhanced image.

        Raises:
            RuntimeError: If DocEnTr container fails.
        """
        tool = self._docker_tool(self._image_name, self._volumes)
        result = tool.run([
            "--input", str(image_path),
            "--output_dir", str(output_dir),
            "--format", "json",
        ])

        if result.exit_code != 0:
            raise RuntimeError(
                f"DocEnTr failed (exit {result.exit_code}): {result.stderr}"
            )

        data = json.loads(result.stdout)
        return pathlib.Path(data["enhanced_path"])


class HandwritingSpecialist:
    """Handwriting extraction specialist: TrOCR + DocEnTr + dual-model.

    Pipeline:
    1. DocEnTr enhances image (if flagged as degraded)
    2. TrOCR extracts text with per-character confidence
    3. Codex corrects OCR errors (primary: better raw accuracy)
    4. Claude verifies and checks for hallucination (secondary)
    5. Dual-model agreement boosts confidence; disagreement flags review
    """

    def __init__(
        self,
        trocr_tool: TrOCRTool | None = None,
        docentr_tool: DocEnTrTool | None = None,
        codex_client: VLMClient | None = None,
        claude_client: VLMClient | None = None,
    ) -> None:
        self.trocr_tool = trocr_tool or TrOCRTool()
        self.docentr_tool = docentr_tool
        self.codex_client = codex_client
        self.claude_client = claude_client

    async def extract(
        self,
        image_path: pathlib.Path,
        region_id: str,
        page: int,
        bbox: BoundingBox,
        needs_enhancement: bool = False,
    ) -> Region:
        """Extract handwritten text from a region image.

        Args:
            image_path: Path to the cropped handwriting image.
            region_id: Unique identifier for this region.
            page: Page number (1-based).
            bbox: Normalized bounding box of this region.
            needs_enhancement: Whether DocEnTr enhancement is needed.

        Returns:
            Region with HandwritingContent populated.
        """
        extraction_method = ""
        working_image = image_path

        # Stage 0: Enhancement for degraded images
        if needs_enhancement and self.docentr_tool is not None:
            try:
                working_image = self.docentr_tool.enhance(
                    image_path, output_dir=image_path.parent,
                )
                extraction_method = "docentr + "
            except Exception as exc:
                logger.warning(
                    "DocEnTr enhancement failed for %s, using original: %s",
                    region_id, exc,
                )

        # Stage 1: TrOCR
        trocr_result = self.trocr_tool.extract(working_image)
        extraction_method += "trocr"
        final_text = trocr_result.text
        final_confidence = trocr_result.confidence

        # Stage 2: Codex correction (primary, better raw accuracy)
        codex_text: str | None = None
        codex_confidence = 0.0
        if self.codex_client is not None:
            try:
                prompt = CODEX_HANDWRITING_PROMPT.format(
                    ocr_text=trocr_result.text,
                    ocr_confidence=trocr_result.confidence,
                )
                codex_resp = await self.codex_client.send_vision_request(
                    image_path=working_image,
                    prompt=prompt,
                )
                if isinstance(codex_resp.content, dict):
                    codex_text = codex_resp.content.get("corrected_text")
                    codex_confidence = codex_resp.confidence
                    if codex_text:
                        final_text = codex_text
                        final_confidence = codex_confidence
                        extraction_method += f" + {codex_resp.model}"
            except Exception as exc:
                logger.warning(
                    "Codex handwriting correction failed for %s: %s",
                    region_id, exc,
                )

        # Stage 3: Claude hallucination check (secondary)
        claude_text: str | None = None
        claude_confidence = 0.0
        if self.claude_client is not None:
            try:
                prompt = CLAUDE_HANDWRITING_PROMPT.format(
                    candidate_text=final_text,
                )
                claude_resp = await self.claude_client.send_vision_request(
                    image_path=working_image,
                    prompt=prompt,
                )
                if isinstance(claude_resp.content, dict):
                    claude_text = claude_resp.content.get("verified_text")
                    claude_confidence = claude_resp.confidence
                    hallucination_risk = claude_resp.content.get(
                        "hallucination_risk", "unknown",
                    )
                    extraction_method += f" + {claude_resp.model}"

                    # If Claude found a different text with higher confidence, use it
                    if (
                        claude_text
                        and claude_text != final_text
                        and claude_confidence > final_confidence
                    ):
                        final_text = claude_text
                        final_confidence = claude_confidence

                    # If models agree, boost confidence
                    if (
                        codex_text
                        and claude_text
                        and codex_text == claude_text
                    ):
                        final_confidence = min(1.0, final_confidence + 0.05)

                    # High hallucination risk reduces confidence
                    if hallucination_risk == "high":
                        final_confidence = max(0.0, final_confidence - 0.15)
            except Exception as exc:
                logger.warning(
                    "Claude hallucination check failed for %s: %s",
                    region_id, exc,
                )

        # Determine review status
        models_disagree = (
            codex_text is not None
            and claude_text is not None
            and codex_text != claude_text
        )

        return Region(
            id=region_id,
            type=RegionType.HANDWRITING,
            subtype=None,
            page=page,
            bbox=bbox,
            content=HandwritingContent(text=final_text, latex=None),
            confidence=final_confidence,
            extraction_method=extraction_method,
            needs_review=final_confidence < 0.90 or models_disagree,
            review_reason=(
                "Dual-model disagreement on handwriting transcription"
                if models_disagree
                else (
                    f"Handwriting confidence {final_confidence:.2f} below 0.90 threshold"
                    if final_confidence < 0.90
                    else None
                )
            ),
        )
