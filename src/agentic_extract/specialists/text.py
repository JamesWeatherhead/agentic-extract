"""Text Specialist: PaddleOCR + Claude enhancement.

Follows the OCR-then-LLM pattern. PaddleOCR extracts raw text with
per-character confidence. If confidence is below threshold, Claude
corrects errors using both the OCR output and the original image.
"""
from __future__ import annotations

import json
import logging
import pathlib
from dataclasses import dataclass, field

from agentic_extract.clients.vlm import VLMClient, VLMResponse
from agentic_extract.models import BoundingBox, Region, RegionType, TextContent
from agentic_extract.tools.docker_runner import DockerTool

logger = logging.getLogger(__name__)

# If PaddleOCR confidence is at or above this, skip VLM enhancement
VLM_SKIP_THRESHOLD = 0.95

CLAUDE_TEXT_PROMPT = """You are a document text extraction expert. You are given:
1. Raw OCR output from PaddleOCR (may contain errors)
2. The original image of the text region

Your task: Correct any OCR errors and produce clean, accurate text.
Return JSON: {{"corrected_text": "the corrected text here"}}

OCR output: {ocr_text}
OCR confidence: {ocr_confidence}

Rules:
- Fix obvious OCR errors (0/O confusion, 1/l confusion, etc.)
- Preserve original formatting and line breaks
- If uncertain about a character, keep the OCR version
- Return null for corrected_text ONLY if the image is unreadable
"""


@dataclass
class OCRResult:
    """Raw OCR extraction result."""

    text: str
    confidence: float
    per_char_confidences: list[float] = field(default_factory=list)


class PaddleOCRTool:
    """PaddleOCR 3.0 Docker wrapper for text extraction."""

    IMAGE_NAME = "paddlepaddle/paddleocr:latest"

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

    def extract(self, image_path: pathlib.Path) -> OCRResult:
        """Run PaddleOCR on an image and return text with confidence.

        Args:
            image_path: Path to the cropped region image.

        Returns:
            OCRResult with extracted text and confidence scores.

        Raises:
            RuntimeError: If PaddleOCR container fails.
        """
        tool = self._docker_tool(self._image_name, self._volumes)
        result = tool.run([
            "--image_dir", str(image_path),
            "--type", "ocr",
            "--output_format", "json",
        ])

        if result.exit_code != 0:
            raise RuntimeError(
                f"PaddleOCR failed (exit {result.exit_code}): {result.stderr}"
            )

        data = json.loads(result.stdout)
        return OCRResult(
            text=data.get("text", ""),
            confidence=data.get("confidence", 0.0),
            per_char_confidences=data.get("per_char_confidences", []),
        )


class TextSpecialist:
    """Text extraction specialist using OCR-then-LLM pattern.

    1. PaddleOCR extracts raw text with per-character confidence
    2. If confidence < 0.95, Claude enhances the text using the
       OCR output and original image
    3. If Claude fails, falls back to raw OCR output
    """

    def __init__(
        self,
        ocr_tool: PaddleOCRTool | None = None,
        vlm_client: VLMClient | None = None,
    ) -> None:
        self.ocr_tool = ocr_tool or PaddleOCRTool()
        self.vlm_client = vlm_client

    async def extract(
        self,
        image_path: pathlib.Path,
        region_id: str,
        page: int,
        bbox: BoundingBox,
    ) -> Region:
        """Extract text from a region image.

        Args:
            image_path: Path to the cropped region image.
            region_id: Unique identifier for this region.
            page: Page number (1-based).
            bbox: Normalized bounding box of this region.

        Returns:
            Region with TextContent populated.
        """
        # Stage 1: PaddleOCR
        ocr_result = self.ocr_tool.extract(image_path)
        extraction_method = "paddleocr_3.0"
        final_text = ocr_result.text
        final_confidence = ocr_result.confidence

        # Stage 2: Claude enhancement (only if needed)
        if (
            ocr_result.confidence < VLM_SKIP_THRESHOLD
            and self.vlm_client is not None
        ):
            try:
                prompt = CLAUDE_TEXT_PROMPT.format(
                    ocr_text=ocr_result.text,
                    ocr_confidence=ocr_result.confidence,
                )
                vlm_response = await self.vlm_client.send_vision_request(
                    image_path=image_path,
                    prompt=prompt,
                )
                corrected = vlm_response.content
                if isinstance(corrected, dict) and corrected.get("corrected_text"):
                    final_text = corrected["corrected_text"]
                    final_confidence = vlm_response.confidence
                    extraction_method = f"paddleocr_3.0 + {vlm_response.model}"
            except Exception as exc:
                logger.warning(
                    "VLM enhancement failed for region %s, using OCR fallback: %s",
                    region_id, exc,
                )

        return Region(
            id=region_id,
            type=RegionType.TEXT,
            subtype=None,
            page=page,
            bbox=bbox,
            content=TextContent(text=final_text, markdown=final_text),
            confidence=final_confidence,
            extraction_method=extraction_method,
            needs_review=final_confidence < 0.90,
            review_reason=(
                f"Text confidence {final_confidence:.2f} below 0.90 threshold"
                if final_confidence < 0.90
                else None
            ),
        )
