"""Re-extraction engine: model-switch retry loop for low-confidence fields.

When the Validator flags a field with confidence in [0.70, 0.90), this
engine re-extracts using the alternate model. If Claude produced the
original, Codex retries (and vice versa). Max 2 retries per field.
Agreement between models boosts confidence by +0.10.
"""
from __future__ import annotations

import logging
import pathlib
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field

from agentic_extract.clients.vlm import VLMClient, VLMResponse
from agentic_extract.models import (
    BoundingBox,
    Region,
    RegionType,
    TextContent,
    TableContent,
    FigureContent,
    HandwritingContent,
    FormulaContent,
)

logger = logging.getLogger(__name__)

ACCEPT_THRESHOLD = 0.90
REEXTRACT_LOW = 0.70
CONFIDENCE_BOOST = 0.10

REEXTRACTION_PROMPT = """You are a document extraction expert performing a re-extraction.
Previous extraction attempt produced a value with low confidence.

Region type: {region_type}
Field name: {field_name}
Previous value: {previous_value}
Previous confidence: {previous_confidence}

Your task: Look at this document region image and extract the value for
the field "{field_name}". Return JSON with key "corrected_text" for text
fields, or the appropriate structured data for tables/figures.

Rules:
- Extract directly from the image; do not simply copy the previous value
- If the value is genuinely unreadable, return null for corrected_text
- Be precise: character-level accuracy matters
"""


class ReExtractionCandidate(BaseModel):
    """A single re-extraction attempt result."""

    value: Any
    confidence: float = Field(..., ge=0.0, le=1.0)
    model: str
    extraction_method: str


class ReExtractionResult(BaseModel):
    """Result of the re-extraction loop for a single field."""

    region_id: str
    field_name: str
    original_value: Any
    original_confidence: float = Field(..., ge=0.0, le=1.0)
    original_model: str
    candidates: list[ReExtractionCandidate]
    final_value: Any
    final_confidence: float = Field(..., ge=0.0, le=1.0)
    retries_used: int = Field(..., ge=0)
    models_agreed: bool
    accepted: bool
    flagged: bool


class ModelSwitchStrategy:
    """Determines the alternate model for re-extraction.

    If Claude produced the failing extraction, switch to Codex.
    If Codex produced the failing extraction, switch to Claude.
    Unknown models default to Claude.
    """

    CLAUDE_MODELS = {"claude-opus-4-20250514", "claude-sonnet-4-20250514"}
    CODEX_MODELS = {"gpt-4o", "gpt-4o-mini"}

    DEFAULT_CLAUDE = "claude-opus-4-20250514"
    DEFAULT_CODEX = "gpt-4o"

    def get_alternate_model(self, original_model: str) -> str:
        """Return the alternate model name for re-extraction."""
        if original_model in self.CLAUDE_MODELS or original_model.startswith("claude"):
            return self.DEFAULT_CODEX
        if original_model in self.CODEX_MODELS or original_model.startswith("gpt"):
            return self.DEFAULT_CLAUDE
        # Unknown model: default to Claude
        return self.DEFAULT_CLAUDE

    def is_claude_model(self, model: str) -> bool:
        return model in self.CLAUDE_MODELS or model.startswith("claude")

    def is_codex_model(self, model: str) -> bool:
        return model in self.CODEX_MODELS or model.startswith("gpt")


def _extract_value_from_response(response: VLMResponse, region_type: RegionType) -> Any:
    """Pull the extracted value from a VLM response based on region type."""
    content = response.content
    if isinstance(content, dict):
        # Text-like regions
        if region_type in (RegionType.TEXT, RegionType.HANDWRITING):
            return content.get("corrected_text", content.get("text", str(content)))
        # Table regions
        if region_type == RegionType.TABLE:
            return content
        # Figure regions
        if region_type == RegionType.FIGURE:
            return content
        # Formula regions
        if region_type == RegionType.FORMULA:
            return content.get("latex", str(content))
    return str(content) if content is not None else None


def _get_region_value(region: Region) -> Any:
    """Extract the primary value from a region's content."""
    content = region.content
    if isinstance(content, TextContent):
        return content.text
    if isinstance(content, TableContent):
        return content.json_data
    if isinstance(content, FigureContent):
        return content.figure_json
    if isinstance(content, HandwritingContent):
        return content.text
    if isinstance(content, FormulaContent):
        return content.latex
    return None


def _values_agree(val_a: Any, val_b: Any) -> bool:
    """Check if two extracted values agree (exact match for strings, deep equal for dicts)."""
    if isinstance(val_a, str) and isinstance(val_b, str):
        return val_a.strip().lower() == val_b.strip().lower()
    if isinstance(val_a, dict) and isinstance(val_b, dict):
        return val_a == val_b
    return str(val_a) == str(val_b)


class ReExtractionEngine:
    """Re-extraction engine with model switching.

    Workflow:
    1. Receive a region that failed validation (confidence in [0.70, 0.90))
    2. Identify the original model used for extraction
    3. Re-extract using the alternate model (Claude -> Codex or Codex -> Claude)
    4. If models agree: boost confidence by +0.10
    5. If models disagree and retries remain: try again with original model + different prompt
    6. If retries exhausted and still disagreeing: flag with both candidates

    Args:
        claude_client: VLM client for Claude API calls.
        codex_client: VLM client for Codex/GPT-4o API calls.
        max_retries: Maximum re-extraction attempts per field (default 2).
    """

    def __init__(
        self,
        claude_client: VLMClient,
        codex_client: VLMClient,
        max_retries: int = 2,
    ) -> None:
        self.claude_client = claude_client
        self.codex_client = codex_client
        self.max_retries = max_retries
        self.strategy = ModelSwitchStrategy()

    def _get_client_for_model(self, model: str) -> VLMClient:
        """Return the appropriate VLM client for a given model name."""
        if self.strategy.is_codex_model(model):
            return self.codex_client
        return self.claude_client

    async def re_extract_field(
        self,
        region: Region,
        field_name: str,
        image_path: pathlib.Path,
        original_model: str,
    ) -> ReExtractionResult:
        """Run the re-extraction loop for a single field.

        Args:
            region: The region containing the low-confidence field.
            field_name: Name of the field being re-extracted.
            image_path: Path to the cropped region image.
            original_model: The model that produced the original extraction.

        Returns:
            ReExtractionResult with candidates, final value, and flags.
        """
        original_value = _get_region_value(region)
        original_confidence = region.confidence
        candidates: list[ReExtractionCandidate] = []
        retries_used = 0
        best_value = original_value
        best_confidence = original_confidence
        models_agreed = False

        # Alternate between models: first retry uses alternate, second uses original
        model_sequence = [
            self.strategy.get_alternate_model(original_model),
            original_model,
        ]

        for retry_idx in range(self.max_retries):
            retries_used += 1
            current_model = model_sequence[retry_idx % len(model_sequence)]
            client = self._get_client_for_model(current_model)

            prompt = REEXTRACTION_PROMPT.format(
                region_type=region.type.value,
                field_name=field_name,
                previous_value=str(best_value),
                previous_confidence=best_confidence,
            )

            try:
                response = await client.send_vision_request(
                    image_path=image_path,
                    prompt=prompt,
                )
                new_value = _extract_value_from_response(response, region.type)
                new_confidence = response.confidence

                candidate = ReExtractionCandidate(
                    value=new_value,
                    confidence=new_confidence,
                    model=current_model,
                    extraction_method=f"re_extraction_{retry_idx + 1} + {current_model}",
                )
                candidates.append(candidate)

                # Check cross-model agreement with original value
                # Only counts as "models agreed" when a different model
                # family confirms the same value (not same-model consistency)
                is_cross_model = current_model != original_model
                if _values_agree(original_value, new_value) and is_cross_model:
                    models_agreed = True
                    boosted = min(1.0, max(original_confidence, new_confidence) + CONFIDENCE_BOOST)
                    best_value = new_value
                    best_confidence = boosted
                    break
                else:
                    # Take the higher-confidence value as best so far
                    if new_confidence > best_confidence:
                        best_value = new_value
                        best_confidence = new_confidence

            except Exception as exc:
                logger.warning(
                    "Re-extraction attempt %d failed for region %s field %s: %s",
                    retry_idx + 1, region.id, field_name, exc,
                )
                candidates.append(ReExtractionCandidate(
                    value=None,
                    confidence=0.0,
                    model=current_model,
                    extraction_method=f"re_extraction_{retry_idx + 1}_failed",
                ))

        accepted = best_confidence >= ACCEPT_THRESHOLD
        flagged = not accepted

        return ReExtractionResult(
            region_id=region.id,
            field_name=field_name,
            original_value=original_value,
            original_confidence=original_confidence,
            original_model=original_model,
            candidates=candidates,
            final_value=best_value,
            final_confidence=best_confidence,
            retries_used=retries_used,
            models_agreed=models_agreed,
            accepted=accepted,
            flagged=flagged,
        )
