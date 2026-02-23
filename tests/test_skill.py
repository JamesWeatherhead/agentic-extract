# tests/test_skill.py
"""Tests for the Claude Code skill entry point."""
import asyncio
import json
import pathlib
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from PIL import Image

from agentic_extract.models import (
    AuditTrail,
    BoundingBox,
    DocumentMetadata,
    ExtractionResult,
    ProcessingStage,
    Region,
    RegionType,
    TextContent,
)
from agentic_extract.skill import (
    run_extraction,
    format_summary,
    write_outputs,
)


def _make_sample_result(source: str = "test.pdf") -> ExtractionResult:
    """Helper to build a sample ExtractionResult for testing."""
    return ExtractionResult(
        document=DocumentMetadata(
            id="doc-test",
            source=source,
            page_count=2,
            processing_timestamp=datetime.now(timezone.utc),
            approach="B",
            total_confidence=0.93,
            processing_time_ms=5000,
        ),
        markdown="# Test Document\n\nSample extracted text.",
        regions=[
            Region(
                id="r1",
                type=RegionType.TEXT,
                subtype=None,
                page=1,
                bbox=BoundingBox(x=0.05, y=0.1, w=0.9, h=0.15),
                content=TextContent(text="Sample extracted text.", markdown="Sample extracted text."),
                confidence=0.95,
                extraction_method="paddleocr_3.0",
            ),
            Region(
                id="r2",
                type=RegionType.TEXT,
                subtype=None,
                page=2,
                bbox=BoundingBox(x=0.05, y=0.1, w=0.9, h=0.10),
                content=TextContent(text="Page two content.", markdown="Page two content."),
                confidence=0.91,
                extraction_method="paddleocr_3.0 + claude-opus-4-20250514",
            ),
        ],
        extracted_entities={},
        audit_trail=AuditTrail(
            models_used=["claude-opus-4-20250514", "paddleocr_3.0"],
            total_llm_calls=1,
            re_extractions=0,
            fields_flagged=0,
            processing_stages=[
                ProcessingStage(stage="ingestion", duration_ms=500),
                ProcessingStage(stage="extraction", duration_ms=3000),
                ProcessingStage(stage="validation", duration_ms=1000),
                ProcessingStage(stage="assembly", duration_ms=500),
            ],
        ),
    )


def test_format_summary_basic():
    result = _make_sample_result()
    summary = format_summary(result)

    assert "test.pdf" in summary
    assert "2" in summary  # page count
    assert "0.93" in summary or "93" in summary  # confidence
    assert "paddleocr" in summary.lower() or "models" in summary.lower()
    assert isinstance(summary, str)
    assert len(summary) > 50


def test_format_summary_with_flags():
    result = _make_sample_result()
    result.audit_trail.fields_flagged = 3
    result.audit_trail.re_extractions = 2
    summary = format_summary(result)

    assert "3" in summary  # flagged fields
    assert "2" in summary  # re-extractions


def test_write_outputs_creates_files(tmp_path: pathlib.Path):
    result = _make_sample_result()
    output_dir = tmp_path / "output"

    write_outputs(result, output_dir)

    md_file = output_dir / "test.extracted.md"
    json_file = output_dir / "test.extracted.json"

    assert md_file.exists()
    assert json_file.exists()

    # Verify markdown content
    md_content = md_file.read_text()
    assert "Test Document" in md_content or "Sample extracted text" in md_content

    # Verify JSON is valid and round-trips
    json_content = json.loads(json_file.read_text())
    assert json_content["document"]["source"] == "test.pdf"
    assert len(json_content["regions"]) == 2


def test_write_outputs_custom_dir(tmp_path: pathlib.Path):
    result = _make_sample_result(source="report.png")
    output_dir = tmp_path / "custom" / "nested"

    write_outputs(result, output_dir)

    assert (output_dir / "report.extracted.md").exists()
    assert (output_dir / "report.extracted.json").exists()


@pytest.mark.asyncio
async def test_run_extraction_calls_pipeline(tmp_path: pathlib.Path):
    """run_extraction should create a Pipeline and call extract."""
    img = Image.new("RGB", (100, 100), "white")
    img_path = tmp_path / "doc.png"
    img.save(img_path)

    sample_result = _make_sample_result(source=str(img_path))

    with patch("agentic_extract.skill.Pipeline") as MockPipeline:
        mock_instance = AsyncMock()
        mock_instance.extract.return_value = sample_result
        MockPipeline.return_value = mock_instance

        result = await run_extraction(
            file_path=img_path,
            output_dir=tmp_path / "out",
        )

        assert isinstance(result, ExtractionResult)
        MockPipeline.assert_called_once()
        mock_instance.extract.assert_called_once_with(img_path, schema=None)


@pytest.mark.asyncio
async def test_run_extraction_with_schema(tmp_path: pathlib.Path):
    """run_extraction should pass schema to Pipeline.extract."""
    img = Image.new("RGB", (100, 100), "white")
    img_path = tmp_path / "doc.png"
    img.save(img_path)

    schema = {"type": "object", "properties": {"name": {"type": "string"}}}
    sample_result = _make_sample_result(source=str(img_path))

    with patch("agentic_extract.skill.Pipeline") as MockPipeline:
        mock_instance = AsyncMock()
        mock_instance.extract.return_value = sample_result
        MockPipeline.return_value = mock_instance

        result = await run_extraction(
            file_path=img_path,
            output_dir=tmp_path / "out",
            schema=schema,
        )

        mock_instance.extract.assert_called_once_with(img_path, schema=schema)


@pytest.mark.asyncio
async def test_run_extraction_writes_output_files(tmp_path: pathlib.Path):
    """run_extraction should write both markdown and JSON output files."""
    img = Image.new("RGB", (100, 100), "white")
    img_path = tmp_path / "invoice.png"
    img.save(img_path)

    sample_result = _make_sample_result(source=str(img_path))
    output_dir = tmp_path / "results"

    with patch("agentic_extract.skill.Pipeline") as MockPipeline:
        mock_instance = AsyncMock()
        mock_instance.extract.return_value = sample_result
        MockPipeline.return_value = mock_instance

        await run_extraction(
            file_path=img_path,
            output_dir=output_dir,
        )

        # Output files should exist
        assert (output_dir / "invoice.extracted.md").exists()
        assert (output_dir / "invoice.extracted.json").exists()
