# src/agentic_extract/skill.py
"""Claude Code skill entry point for /extract command.

This module provides the interface between the Claude Code skill
definition and the extraction pipeline. It handles argument parsing,
pipeline invocation, output file writing, and summary reporting.

Usage (from Claude Code):
    /extract /path/to/document.pdf --output /path/to/output/
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import pathlib
from typing import Any

from agentic_extract.clients.vlm import ClaudeClient, CodexClient
from agentic_extract.models import ExtractionResult
from agentic_extract.pipeline import Pipeline, PipelineConfig

logger = logging.getLogger(__name__)


def format_summary(result: ExtractionResult) -> str:
    """Format a human-readable summary of the extraction result.

    Args:
        result: The completed extraction result.

    Returns:
        A multi-line summary string suitable for display.
    """
    doc = result.document
    audit = result.audit_trail
    regions_by_type: dict[str, int] = {}
    for r in result.regions:
        rtype = r.type.value
        regions_by_type[rtype] = regions_by_type.get(rtype, 0) + 1

    flagged_regions = [r for r in result.regions if r.needs_review]

    lines = [
        f"Extraction Complete: {doc.source}",
        f"Pages: {doc.page_count} | Regions: {len(result.regions)} | "
        f"Confidence: {doc.total_confidence:.2f}",
        f"Time: {doc.processing_time_ms}ms",
        "",
        "Regions by type:",
    ]
    for rtype, count in sorted(regions_by_type.items()):
        lines.append(f"  {rtype}: {count}")

    lines.append("")
    lines.append(
        f"Models used: {', '.join(audit.models_used)}"
    )
    lines.append(f"LLM calls: {audit.total_llm_calls}")
    lines.append(f"Re-extractions: {audit.re_extractions}")
    lines.append(f"Fields flagged: {audit.fields_flagged}")

    if flagged_regions:
        lines.append("")
        lines.append("Flagged regions requiring review:")
        for r in flagged_regions:
            lines.append(
                f"  {r.id} (page {r.page}, {r.type.value}): "
                f"confidence {r.confidence:.2f} - {r.review_reason or 'no reason'}"
            )

    lines.append("")
    lines.append("Stage timing:")
    for stage in audit.processing_stages:
        lines.append(f"  {stage.stage}: {stage.duration_ms}ms")

    return "\n".join(lines)


def write_outputs(
    result: ExtractionResult,
    output_dir: pathlib.Path,
) -> tuple[pathlib.Path, pathlib.Path]:
    """Write extraction results to markdown and JSON files.

    Args:
        result: The completed extraction result.
        output_dir: Directory to write output files.

    Returns:
        Tuple of (markdown_path, json_path).
    """
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Derive output filename from source
    source_name = pathlib.Path(result.document.source).stem
    md_path = output_dir / f"{source_name}.extracted.md"
    json_path = output_dir / f"{source_name}.extracted.json"

    # Write markdown
    md_path.write_text(result.markdown, encoding="utf-8")

    # Write JSON
    json_str = result.model_dump_json(indent=2)
    json_path.write_text(json_str, encoding="utf-8")

    return md_path, json_path


async def run_extraction(
    file_path: pathlib.Path,
    output_dir: pathlib.Path | None = None,
    schema: dict[str, Any] | None = None,
) -> ExtractionResult:
    """Run the full extraction pipeline and write output files.

    This is the main entry point called by the Claude Code skill.

    Args:
        file_path: Path to the document to extract.
        output_dir: Directory for output files. Defaults to file_path's directory.
        schema: Optional JSON schema for structured entity extraction.

    Returns:
        ExtractionResult with all extracted data.
    """
    file_path = pathlib.Path(file_path)

    if output_dir is None:
        output_dir = file_path.parent

    # Initialize VLM clients from environment
    claude_api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    codex_api_key = os.environ.get("OPENAI_API_KEY", "")

    claude_client = ClaudeClient(api_key=claude_api_key)
    codex_client = CodexClient(api_key=codex_api_key)

    pipeline = Pipeline(
        claude_client=claude_client,
        codex_client=codex_client,
    )

    result = await pipeline.extract(file_path, schema=schema)

    # Write output files
    md_path, json_path = write_outputs(result, output_dir)
    logger.info("Markdown output: %s", md_path)
    logger.info("JSON output: %s", json_path)

    return result
