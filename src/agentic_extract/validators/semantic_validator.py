# src/agentic_extract/validators/semantic_validator.py
"""Validator Layer 3: LLM-assisted semantic validation.

Single Claude call per document (not per field). Uses Claude's document
understanding (DocVQA 95.2%) to check whether extracted fields make
sense together.

This catches errors no field-level validation can find, such as:
- Text says "mortality decreased" but table shows it increasing
- Extracted entity names that don't appear anywhere in the document
- Numerical values that are inconsistent across regions
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

from agentic_extract.clients.vlm import VLMClient
from agentic_extract.models import (
    Region,
    RegionType,
    TableContent,
    TextContent,
)

logger = logging.getLogger(__name__)

CLAUDE_SEMANTIC_PROMPT = """You are a document validation expert. You are given extracted regions from a document.
Your task: Check whether the extracted fields make sense together.

Look for:
1. Contradictions between text and table data
2. Numerical values that are inconsistent across regions
3. Entity names mentioned in text that do not appear in tables (or vice versa)
4. Implausible values that may indicate extraction hallucination
5. Missing context that makes extracted data ambiguous

Extracted regions:
{regions_summary}

Return JSON:
{{
  "issues": [
    {{
      "region_id": "the region with the issue",
      "description": "clear description of the semantic problem",
      "confidence_penalty": 0.05 to 0.30 (how much to reduce confidence)
    }}
  ]
}}

Return {{"issues": []}} if all fields are semantically consistent.
"""


@dataclass
class SemanticIssue:
    """A semantic inconsistency found by LLM validation."""

    region_id: str
    description: str
    confidence_penalty: float


def _build_regions_summary(regions: list[Region]) -> str:
    """Build a text summary of all regions for the Claude prompt."""
    lines: list[str] = []
    for region in regions:
        line = f"[{region.id}] Type={region.type.value}, Page={region.page}"
        if isinstance(region.content, TextContent):
            # Truncate long text
            text = region.content.text[:500]
            line += f", Text: {text}"
        elif isinstance(region.content, TableContent):
            headers = region.content.json_data.get("headers", [])
            rows = region.content.json_data.get("rows", [])
            line += f", Table: headers={headers}, rows={rows[:5]}"
        else:
            line += f", Content type: {type(region.content).__name__}"
        lines.append(line)
    return "\n".join(lines)


async def validate_semantics(
    regions: list[Region],
    claude_client: VLMClient | None = None,
) -> list[SemanticIssue]:
    """Run Layer 3 semantic validation using a single Claude call.

    This is the only LLM call in the validation pipeline. It sends
    a summary of ALL regions to Claude and asks whether the extracted
    fields are semantically consistent.

    Args:
        regions: List of extracted regions to validate.
        claude_client: Claude VLM client. If None, skips validation.

    Returns:
        List of SemanticIssue objects (empty if all consistent or no client).
    """
    if claude_client is None or not regions:
        return []

    regions_summary = _build_regions_summary(regions)
    prompt = CLAUDE_SEMANTIC_PROMPT.format(regions_summary=regions_summary)

    try:
        # Use a dummy image path; semantic validation works on text summaries.
        # In production, this would use the first page image for grounding.
        # For now, we pass the prompt as a text-only request by using
        # any available region's page image, or a placeholder.
        import pathlib
        import tempfile
        from PIL import Image

        # Create a minimal placeholder image for the API call
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            placeholder = pathlib.Path(f.name)
            Image.new("RGB", (100, 100), "white").save(placeholder)

        try:
            response = await claude_client.send_vision_request(
                image_path=placeholder,
                prompt=prompt,
            )
        finally:
            placeholder.unlink(missing_ok=True)

        if not isinstance(response.content, dict):
            return []

        raw_issues = response.content.get("issues", [])
        issues: list[SemanticIssue] = []
        for item in raw_issues:
            if isinstance(item, dict):
                issues.append(SemanticIssue(
                    region_id=item.get("region_id", "unknown"),
                    description=item.get("description", ""),
                    confidence_penalty=float(item.get("confidence_penalty", 0.10)),
                ))
        return issues

    except Exception as exc:
        logger.warning("Semantic validation failed (Claude error): %s", exc)
        return []
