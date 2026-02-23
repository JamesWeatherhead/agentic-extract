# src/agentic_extract/coordinator/assembly.py
"""Result assembly: merge specialist outputs into final Markdown + JSON.

Takes extracted regions (ordered by reading order), document metadata,
and produces the complete ExtractionResult with both Markdown and
JSON representations.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

from agentic_extract.models import (
    AuditTrail,
    DocumentMetadata,
    ExtractionResult,
    ProcessingStage,
    Region,
    RegionType,
    TableContent,
    TextContent,
)


def _region_to_markdown(region: Region) -> str:
    """Convert a single region to its Markdown representation."""
    lines: list[str] = []
    review_tag = " [NEEDS REVIEW]" if region.needs_review else ""

    if region.type == RegionType.TEXT:
        content = region.content
        if isinstance(content, TextContent):
            lines.append(f"{content.markdown}{review_tag}")
            lines.append("")
            lines.append(
                f"*Confidence: {region.confidence:.2f} | "
                f"Method: {region.extraction_method}*"
            )

    elif region.type == RegionType.TABLE:
        content = region.content
        if isinstance(content, TableContent) and content.json_data:
            headers = content.json_data.get("headers", [])
            rows = content.json_data.get("rows", [])

            lines.append(f"**Table (Page {region.page})**{review_tag}")
            lines.append("")
            if headers:
                lines.append("| " + " | ".join(str(h) for h in headers) + " |")
                lines.append("| " + " | ".join("---" for _ in headers) + " |")
                for row in rows:
                    vals = [str(row.get(h, "")) for h in headers]
                    lines.append("| " + " | ".join(vals) + " |")
            lines.append("")
            lines.append(
                f"*Confidence: {region.confidence:.2f} | "
                f"Method: {region.extraction_method}*"
            )

    elif region.type == RegionType.FIGURE:
        lines.append(f"**Figure (Page {region.page})**{review_tag}")
        lines.append("")
        if hasattr(region.content, "description"):
            lines.append(region.content.description)
        lines.append("")
        lines.append(
            f"*Confidence: {region.confidence:.2f} | "
            f"Method: {region.extraction_method}*"
        )

    elif region.type in (RegionType.HANDWRITING, RegionType.FORMULA):
        label = region.type.value.capitalize()
        lines.append(f"**{label} (Page {region.page})**{review_tag}")
        lines.append("")
        if hasattr(region.content, "text"):
            lines.append(region.content.text)
        elif hasattr(region.content, "latex"):
            lines.append(f"$${region.content.latex}$$")
        lines.append("")
        lines.append(
            f"*Confidence: {region.confidence:.2f} | "
            f"Method: {region.extraction_method}*"
        )

    else:
        lines.append(f"**{region.type.value} (Page {region.page})**{review_tag}")

    return "\n".join(lines)


def generate_markdown_output(
    regions: list[Region],
    metadata: DocumentMetadata,
) -> str:
    """Generate the full Markdown output document.

    Args:
        regions: Regions in reading order.
        metadata: Document-level metadata.

    Returns:
        Complete Markdown string.
    """
    lines: list[str] = []
    lines.append(f"# Document: {metadata.source}")
    lines.append("")
    lines.append(
        f"**Source:** {metadata.source} | "
        f"**Pages:** {metadata.page_count} | "
        f"**Processed:** {metadata.processing_timestamp.strftime('%Y-%m-%d')} | "
        f"**Confidence:** {metadata.total_confidence:.2f}"
    )
    lines.append("")
    lines.append("---")
    lines.append("")

    for region in regions:
        lines.append(_region_to_markdown(region))
        lines.append("")
        lines.append("---")
        lines.append("")

    return "\n".join(lines)


def generate_json_output(
    regions: list[Region],
    metadata: DocumentMetadata,
    audit_trail: AuditTrail,
    extracted_entities: dict | None = None,
) -> str:
    """Generate the full JSON output string.

    Args:
        regions: Regions in reading order.
        metadata: Document-level metadata.
        audit_trail: Processing audit trail.
        extracted_entities: Optional entity extractions.

    Returns:
        JSON string conforming to the agentic-extract/v1 schema.
    """
    result = ExtractionResult(
        document=metadata,
        markdown=generate_markdown_output(regions, metadata),
        regions=regions,
        extracted_entities=extracted_entities or {},
        audit_trail=audit_trail,
    )
    return result.model_dump_json(indent=2)


def assemble_result(
    regions: list[Region],
    metadata: DocumentMetadata,
    audit_trail: AuditTrail,
    extracted_entities: dict | None = None,
) -> ExtractionResult:
    """Assemble regions into a final ExtractionResult with a pre-built audit trail.

    This is the pipeline-facing assembly function. Unlike ``assemble``, it
    accepts a fully constructed AuditTrail (with per-stage timing from the
    pipeline orchestrator) instead of building one from scratch.

    Args:
        regions: Extracted regions in reading order.
        metadata: Document metadata.
        audit_trail: Pre-built audit trail from the pipeline.
        extracted_entities: Optional entity extractions.

    Returns:
        Complete ExtractionResult.
    """
    markdown = generate_markdown_output(regions, metadata)
    return ExtractionResult(
        document=metadata,
        markdown=markdown,
        regions=regions,
        extracted_entities=extracted_entities or {},
        audit_trail=audit_trail,
    )


def assemble(
    regions: list[Region],
    reading_order: list[str],
    metadata: DocumentMetadata,
) -> ExtractionResult:
    """Assemble specialist outputs into the final ExtractionResult.

    Args:
        regions: All extracted regions (unordered).
        reading_order: Ordered list of region IDs.
        metadata: Document metadata.

    Returns:
        Complete ExtractionResult with Markdown, JSON, and audit trail.
    """
    # Build region lookup and order by reading order
    region_map = {r.id: r for r in regions}
    ordered_regions: list[Region] = []
    for rid in reading_order:
        if rid in region_map:
            ordered_regions.append(region_map[rid])
    # Append any regions not in reading_order (safety net)
    seen = set(reading_order)
    for r in regions:
        if r.id not in seen:
            ordered_regions.append(r)

    # Collect extraction methods used
    models_used = sorted(set(
        method.strip()
        for r in ordered_regions
        for method in r.extraction_method.split("+")
    ))

    # Count LLM calls (heuristic: count model names that are VLMs)
    vlm_indicators = {"claude", "codex", "gpt", "opus", "sonnet"}
    llm_calls = sum(
        1 for r in ordered_regions
        for part in r.extraction_method.lower().split("+")
        if any(v in part for v in vlm_indicators)
    )

    flagged = sum(1 for r in ordered_regions if r.needs_review)

    audit_trail = AuditTrail(
        models_used=models_used,
        total_llm_calls=llm_calls,
        re_extractions=0,
        fields_flagged=flagged,
        processing_stages=[
            ProcessingStage(stage="assembly", duration_ms=0),
        ],
    )

    markdown = generate_markdown_output(ordered_regions, metadata)

    return ExtractionResult(
        document=metadata,
        markdown=markdown,
        regions=ordered_regions,
        extracted_entities={},
        audit_trail=audit_trail,
    )
