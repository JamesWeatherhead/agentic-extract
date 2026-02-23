# tests/coordinator/test_assembly.py
"""Tests for result assembly and output generation."""
import json
from datetime import datetime, timezone

import pytest

from agentic_extract.coordinator.assembly import (
    assemble,
    generate_json_output,
    generate_markdown_output,
)
from agentic_extract.models import (
    AuditTrail,
    BoundingBox,
    DocumentMetadata,
    ExtractionResult,
    ProcessingStage,
    Region,
    RegionType,
    TableContent,
    TextContent,
)


def _make_text_region(rid: str, page: int, text: str, conf: float) -> Region:
    return Region(
        id=rid,
        type=RegionType.TEXT,
        subtype=None,
        page=page,
        bbox=BoundingBox(x=0.05, y=0.10, w=0.90, h=0.10),
        content=TextContent(text=text, markdown=text),
        confidence=conf,
        extraction_method="paddleocr_3.0",
    )


def _make_table_region(rid: str, page: int, conf: float) -> Region:
    return Region(
        id=rid,
        type=RegionType.TABLE,
        subtype=None,
        page=page,
        bbox=BoundingBox(x=0.05, y=0.30, w=0.90, h=0.30),
        content=TableContent(
            html="<table><tr><th>Gene</th><th>Value</th></tr><tr><td>BRCA1</td><td>3.2</td></tr></table>",
            json_data={"headers": ["Gene", "Value"], "rows": [{"Gene": "BRCA1", "Value": "3.2"}]},
            cell_bboxes=[],
        ),
        confidence=conf,
        extraction_method="docling + claude-opus-4-20250514",
    )


def _make_metadata() -> DocumentMetadata:
    return DocumentMetadata(
        id="doc-test-001",
        source="test_paper.pdf",
        page_count=2,
        processing_timestamp=datetime(2026, 2, 23, 12, 0, 0, tzinfo=timezone.utc),
        approach="B",
        total_confidence=0.93,
        processing_time_ms=5000,
    )


def test_assemble_produces_extraction_result():
    regions = [
        _make_text_region("r1", page=1, text="Introduction paragraph.", conf=0.97),
        _make_table_region("r2", page=1, conf=0.94),
    ]
    reading_order = ["r1", "r2"]
    metadata = _make_metadata()

    result = assemble(regions, reading_order, metadata)

    assert isinstance(result, ExtractionResult)
    assert result.document.source == "test_paper.pdf"
    assert len(result.regions) == 2
    assert result.regions[0].id == "r1"  # reading order preserved
    assert result.regions[1].id == "r2"
    assert result.audit_trail is not None


def test_assemble_orders_by_reading_order():
    """Regions should appear in reading_order sequence, not insertion order."""
    regions = [
        _make_text_region("r2", page=1, text="Second", conf=0.95),
        _make_text_region("r1", page=1, text="First", conf=0.97),
    ]
    reading_order = ["r1", "r2"]

    result = assemble(regions, reading_order, _make_metadata())
    assert result.regions[0].id == "r1"
    assert result.regions[1].id == "r2"


def test_generate_markdown_output():
    regions = [
        _make_text_region("r1", page=1, text="This is the introduction.", conf=0.97),
        _make_table_region("r2", page=1, conf=0.94),
        _make_text_region("r3", page=2, text="Conclusion paragraph.", conf=0.91),
    ]
    metadata = _make_metadata()

    md = generate_markdown_output(regions, metadata)

    assert "test_paper.pdf" in md
    assert "This is the introduction." in md
    assert "Gene" in md  # table header
    assert "BRCA1" in md  # table data
    assert "Conclusion paragraph." in md
    assert "0.94" in md  # table confidence annotation


def test_generate_markdown_flags_low_confidence():
    regions = [
        _make_text_region("r1", page=1, text="Unclear text", conf=0.72),
    ]
    regions[0] = Region(
        **{**regions[0].model_dump(), "needs_review": True, "review_reason": "Low confidence"},
    )
    metadata = _make_metadata()

    md = generate_markdown_output(regions, metadata)
    assert "NEEDS REVIEW" in md


def test_generate_json_output():
    regions = [
        _make_text_region("r1", page=1, text="Hello", conf=0.97),
    ]
    metadata = _make_metadata()
    audit = AuditTrail(
        models_used=["paddleocr_3.0"],
        total_llm_calls=0,
        re_extractions=0,
        fields_flagged=0,
        processing_stages=[ProcessingStage(stage="ingestion", duration_ms=100)],
    )

    json_str = generate_json_output(regions, metadata, audit)
    parsed = json.loads(json_str)

    assert parsed["document"]["source"] == "test_paper.pdf"
    assert len(parsed["regions"]) == 1
    assert parsed["regions"][0]["id"] == "r1"
    assert parsed["audit_trail"]["total_llm_calls"] == 0


def test_assemble_empty_document():
    """Empty document with no regions should still produce valid output."""
    result = assemble([], [], _make_metadata())
    assert result.regions == []
    assert result.markdown is not None
    assert len(result.markdown) > 0  # at least the header


def test_assemble_json_roundtrip():
    """The assembled result must serialize and deserialize cleanly."""
    regions = [
        _make_text_region("r1", page=1, text="Test", conf=0.95),
        _make_table_region("r2", page=2, conf=0.92),
    ]
    result = assemble(regions, ["r1", "r2"], _make_metadata())

    json_str = result.model_dump_json()
    restored = ExtractionResult.model_validate_json(json_str)
    assert restored.document.id == "doc-test-001"
    assert len(restored.regions) == 2
