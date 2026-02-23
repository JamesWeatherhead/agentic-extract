# tests/test_models.py
"""Tests for core Pydantic data models."""
import json
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError


def test_bounding_box_valid():
    from agentic_extract.models import BoundingBox
    bb = BoundingBox(x=0.1, y=0.2, w=0.5, h=0.3)
    assert bb.x == 0.1
    assert bb.y == 0.2
    assert bb.w == 0.5
    assert bb.h == 0.3


def test_bounding_box_rejects_out_of_range():
    from agentic_extract.models import BoundingBox
    with pytest.raises(ValidationError):
        BoundingBox(x=-0.1, y=0.0, w=0.5, h=0.5)
    with pytest.raises(ValidationError):
        BoundingBox(x=0.0, y=0.0, w=1.5, h=0.5)


def test_region_type_enum_values():
    from agentic_extract.models import RegionType
    assert RegionType.TEXT == "text"
    assert RegionType.TABLE == "table"
    assert RegionType.FIGURE == "figure"
    assert RegionType.HANDWRITING == "handwriting"
    assert RegionType.FORMULA == "formula"
    assert RegionType.FORM_FIELD == "form_field"


def test_text_content():
    from agentic_extract.models import TextContent
    tc = TextContent(text="Hello world", markdown="**Hello** world")
    assert tc.text == "Hello world"
    assert tc.markdown == "**Hello** world"


def test_table_content():
    from agentic_extract.models import TableContent, BoundingBox
    cell_bbox = {"row": 0, "col": 0, "bbox": BoundingBox(x=0.1, y=0.2, w=0.3, h=0.05)}
    tc = TableContent(
        html="<table><tr><td>A</td></tr></table>",
        json_data={"headers": ["Col1"], "rows": [{"Col1": "A"}]},
        cell_bboxes=[cell_bbox],
    )
    assert tc.html.startswith("<table>")
    assert tc.json_data["headers"] == ["Col1"]


def test_figure_content():
    from agentic_extract.models import FigureContent
    fc = FigureContent(
        description="A bar chart",
        figure_type="bar_chart",
        figure_json={"title": "Test Chart"},
    )
    assert fc.figure_type == "bar_chart"


def test_handwriting_content():
    from agentic_extract.models import HandwritingContent
    hc = HandwritingContent(text="Patient notes", latex=None)
    assert hc.text == "Patient notes"
    assert hc.latex is None


def test_formula_content():
    from agentic_extract.models import FormulaContent
    fc = FormulaContent(latex=r"\frac{a}{b}", mathml=None)
    assert fc.latex == r"\frac{a}{b}"


def test_region_with_text_content():
    from agentic_extract.models import (
        Region, RegionType, BoundingBox, TextContent,
    )
    region = Region(
        id="region_001",
        type=RegionType.TEXT,
        subtype=None,
        page=1,
        bbox=BoundingBox(x=0.05, y=0.10, w=0.90, h=0.15),
        content=TextContent(text="Hello", markdown="Hello"),
        confidence=0.97,
        extraction_method="paddleocr_3.0",
        model_agreement=None,
        needs_review=False,
        review_reason=None,
    )
    assert region.id == "region_001"
    assert region.type == RegionType.TEXT
    assert region.confidence == 0.97
    assert region.needs_review is False


def test_region_rejects_confidence_out_of_range():
    from agentic_extract.models import (
        Region, RegionType, BoundingBox, TextContent,
    )
    with pytest.raises(ValidationError):
        Region(
            id="r1", type=RegionType.TEXT, subtype=None, page=1,
            bbox=BoundingBox(x=0, y=0, w=1, h=1),
            content=TextContent(text="x", markdown="x"),
            confidence=1.5,
            extraction_method="test",
        )


def test_document_metadata():
    from agentic_extract.models import DocumentMetadata
    dm = DocumentMetadata(
        id="doc-123",
        source="test.pdf",
        page_count=10,
        processing_timestamp=datetime.now(timezone.utc),
        approach="B",
        total_confidence=0.92,
        processing_time_ms=18500,
    )
    assert dm.page_count == 10
    assert dm.approach == "B"


def test_processing_stage():
    from agentic_extract.models import ProcessingStage
    ps = ProcessingStage(stage="ingestion", duration_ms=1200)
    assert ps.stage == "ingestion"


def test_audit_trail():
    from agentic_extract.models import AuditTrail, ProcessingStage
    at = AuditTrail(
        models_used=["claude_opus_4.6", "paddleocr_3.0"],
        total_llm_calls=5,
        re_extractions=1,
        fields_flagged=0,
        processing_stages=[
            ProcessingStage(stage="ingestion", duration_ms=1200),
            ProcessingStage(stage="ocr", duration_ms=3400),
        ],
    )
    assert len(at.models_used) == 2
    assert at.total_llm_calls == 5


def test_extraction_result_full():
    from agentic_extract.models import (
        ExtractionResult, DocumentMetadata, Region, RegionType,
        BoundingBox, TextContent, AuditTrail, ProcessingStage,
    )
    result = ExtractionResult(
        document=DocumentMetadata(
            id="doc-1", source="test.pdf", page_count=1,
            processing_timestamp=datetime.now(timezone.utc),
            approach="B", total_confidence=0.95, processing_time_ms=5000,
        ),
        markdown="# Test\n\nHello world",
        regions=[
            Region(
                id="r1", type=RegionType.TEXT, subtype=None, page=1,
                bbox=BoundingBox(x=0, y=0, w=1, h=0.5),
                content=TextContent(text="Hello world", markdown="Hello world"),
                confidence=0.95, extraction_method="paddleocr_3.0",
            ),
        ],
        extracted_entities={"fields": {}},
        audit_trail=AuditTrail(
            models_used=["paddleocr_3.0"], total_llm_calls=0,
            re_extractions=0, fields_flagged=0,
            processing_stages=[ProcessingStage(stage="ingestion", duration_ms=100)],
        ),
    )
    assert len(result.regions) == 1
    # Must serialize to JSON without error
    json_str = result.model_dump_json()
    parsed = json.loads(json_str)
    assert parsed["document"]["source"] == "test.pdf"


def test_extraction_result_json_roundtrip():
    from agentic_extract.models import (
        ExtractionResult, DocumentMetadata, AuditTrail, ProcessingStage,
    )
    result = ExtractionResult(
        document=DocumentMetadata(
            id="doc-rt", source="roundtrip.pdf", page_count=0,
            processing_timestamp=datetime.now(timezone.utc),
            approach="B", total_confidence=1.0, processing_time_ms=0,
        ),
        markdown="",
        regions=[],
        extracted_entities={},
        audit_trail=AuditTrail(
            models_used=[], total_llm_calls=0, re_extractions=0,
            fields_flagged=0, processing_stages=[],
        ),
    )
    json_str = result.model_dump_json()
    restored = ExtractionResult.model_validate_json(json_str)
    assert restored.document.id == "doc-rt"
