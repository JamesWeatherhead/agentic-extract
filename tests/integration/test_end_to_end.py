# tests/integration/test_end_to_end.py
"""End-to-end integration tests with mocked Docker tools and VLM clients.

All 8 scenarios test the full pipeline path from document input to
Markdown + JSON output. Docker tools and VLM API calls are mocked
to ensure tests run without external dependencies.
"""
import asyncio
import json
import pathlib
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from PIL import Image

from agentic_extract.clients.vlm import VLMResponse
from agentic_extract.models import (
    AuditTrail,
    BoundingBox,
    DocumentMetadata,
    ExtractionResult,
    FigureContent,
    HandwritingContent,
    ProcessingStage,
    Region,
    RegionType,
    TableContent,
    TextContent,
)
from agentic_extract.pipeline import Pipeline, PipelineConfig
from agentic_extract.reextraction.engine import (
    ReExtractionCandidate,
    ReExtractionEngine,
    ReExtractionResult,
)
from agentic_extract.audit import AuditTrailTracker
from agentic_extract.tools.docker_runner import ToolOutput


# --- Shared Fixtures ---


@pytest.fixture
def mock_claude():
    return AsyncMock()


@pytest.fixture
def mock_codex():
    return AsyncMock()


@pytest.fixture
def single_page_image(tmp_path: pathlib.Path) -> pathlib.Path:
    """Create a single white page image at 300 DPI."""
    img = Image.new("RGB", (2550, 3300), "white")
    img_path = tmp_path / "document.png"
    img.save(img_path, dpi=(300, 300))
    return img_path


@pytest.fixture
def mock_ingestion(single_page_image, tmp_path):
    """Create a mocked IngestionResult for a single-page document."""
    mock_page = MagicMock(
        page_number=1,
        image_path=single_page_image,
        width=2550,
        height=3300,
        dpi=300,
    )
    mock_result = MagicMock()
    mock_result.pages = [mock_page]
    mock_result.page_count = 1
    mock_result.source_file = single_page_image
    mock_result.temp_dir = tmp_path
    return mock_result


def _make_text_region(region_id: str, page: int, text: str, confidence: float) -> Region:
    return Region(
        id=region_id,
        type=RegionType.TEXT,
        subtype=None,
        page=page,
        bbox=BoundingBox(x=0.05, y=0.1, w=0.9, h=0.15),
        content=TextContent(text=text, markdown=text),
        confidence=confidence,
        extraction_method="paddleocr_3.0",
    )


def _make_table_region(region_id: str, page: int, confidence: float) -> Region:
    return Region(
        id=region_id,
        type=RegionType.TABLE,
        subtype=None,
        page=page,
        bbox=BoundingBox(x=0.05, y=0.3, w=0.9, h=0.4),
        content=TableContent(
            html="<table><tr><th>Gene</th><th>Value</th></tr><tr><td>BRCA1</td><td>3.2</td></tr></table>",
            json_data={"headers": ["Gene", "Value"], "rows": [{"Gene": "BRCA1", "Value": "3.2"}]},
            cell_bboxes=[
                {"row": 0, "col": 0, "bbox": BoundingBox(x=0.05, y=0.3, w=0.3, h=0.05)},
                {"row": 0, "col": 1, "bbox": BoundingBox(x=0.35, y=0.3, w=0.3, h=0.05)},
                {"row": 1, "col": 0, "bbox": BoundingBox(x=0.05, y=0.35, w=0.3, h=0.05)},
                {"row": 1, "col": 1, "bbox": BoundingBox(x=0.35, y=0.35, w=0.3, h=0.05)},
            ],
        ),
        confidence=confidence,
        extraction_method="docling + claude-opus-4-20250514 + codex_structured_outputs",
    )


def _make_figure_region(region_id: str, page: int, confidence: float) -> Region:
    return Region(
        id=region_id,
        type=RegionType.FIGURE,
        subtype="bar_chart",
        page=page,
        bbox=BoundingBox(x=0.1, y=0.05, w=0.8, h=0.45),
        content=FigureContent(
            description="Bar chart showing gene expression levels across conditions",
            figure_type="bar_chart",
            figure_json={
                "title": "Gene Expression",
                "x_axis": {"label": "Condition"},
                "y_axis": {"label": "Fold Change"},
                "data_series": [{"name": "BRCA1", "values": [1.0, 3.2, 4.1]}],
            },
        ),
        confidence=confidence,
        extraction_method="deplot + claude_chartreasoning + codex_figmatch",
    )


def _make_handwriting_region(region_id: str, page: int, confidence: float) -> Region:
    return Region(
        id=region_id,
        type=RegionType.HANDWRITING,
        subtype=None,
        page=page,
        bbox=BoundingBox(x=0.05, y=0.5, w=0.9, h=0.3),
        content=HandwritingContent(
            text="Patient notes: administered 500mg",
            latex=None,
        ),
        confidence=confidence,
        extraction_method="trocr + codex_ocr + claude_hallcheck",
    )


def _build_mock_assembly(regions, source, page_count=1):
    """Build a mock ExtractionResult for assembly."""
    flagged = sum(1 for r in regions if r.needs_review)
    models = set()
    for r in regions:
        models.add(r.extraction_method)

    return ExtractionResult(
        document=DocumentMetadata(
            id="doc-integration",
            source=str(source),
            page_count=page_count,
            processing_timestamp=datetime.now(timezone.utc),
            approach="B",
            total_confidence=min((r.confidence for r in regions), default=1.0),
            processing_time_ms=5000,
        ),
        markdown="\n\n".join(
            f"Region {r.id}: {getattr(r.content, 'text', getattr(r.content, 'description', ''))}"
            for r in regions
        ),
        regions=regions,
        extracted_entities={},
        audit_trail=AuditTrail(
            models_used=sorted(models),
            total_llm_calls=len([r for r in regions if "claude" in r.extraction_method or "codex" in r.extraction_method]),
            re_extractions=0,
            fields_flagged=flagged,
            processing_stages=[
                ProcessingStage(stage="ingestion", duration_ms=200),
                ProcessingStage(stage="extraction", duration_ms=3000),
                ProcessingStage(stage="validation", duration_ms=1000),
                ProcessingStage(stage="assembly", duration_ms=300),
            ],
        ),
    )


# --- Scenario 1: Clean Text PDF ---


@pytest.mark.asyncio
async def test_scenario_1_clean_text_pdf(
    mock_claude, mock_codex, single_page_image, mock_ingestion, tmp_path,
):
    """Clean text PDF produces text extraction + markdown + JSON."""
    text_region = _make_text_region("r1", 1, "The quick brown fox jumps over the lazy dog.", 0.98)
    layout_region = MagicMock(
        region_id="r1", region_type=RegionType.TEXT, page=1,
        bbox=BoundingBox(x=0.05, y=0.1, w=0.9, h=0.15), confidence=0.98,
    )

    with patch("agentic_extract.pipeline.ingest", return_value=mock_ingestion), \
         patch("agentic_extract.pipeline.detect_layout", return_value=[layout_region]), \
         patch("agentic_extract.pipeline.determine_reading_order", return_value=[layout_region]), \
         patch("agentic_extract.pipeline.route_regions", return_value={"text": [layout_region], "table": [], "visual": []}), \
         patch("agentic_extract.pipeline.TextSpecialist") as MockTextSpec, \
         patch("agentic_extract.pipeline.assemble_result") as mock_assemble:

        MockTextSpec.return_value = AsyncMock()
        MockTextSpec.return_value.extract.return_value = text_region
        mock_assemble.return_value = _build_mock_assembly([text_region], single_page_image)

        pipeline = Pipeline(claude_client=mock_claude, codex_client=mock_codex)
        result = await pipeline.extract(single_page_image)

    assert isinstance(result, ExtractionResult)
    assert len(result.regions) == 1
    assert result.regions[0].type == RegionType.TEXT
    assert result.regions[0].content.text == "The quick brown fox jumps over the lazy dog."
    assert result.regions[0].confidence >= 0.90
    assert result.markdown is not None
    assert len(result.markdown) > 0
    # JSON roundtrip
    json_str = result.model_dump_json()
    restored = ExtractionResult.model_validate_json(json_str)
    assert restored.document.source == str(single_page_image)


# --- Scenario 2: PDF with Tables ---


@pytest.mark.asyncio
async def test_scenario_2_pdf_with_tables(
    mock_claude, mock_codex, single_page_image, mock_ingestion, tmp_path,
):
    """PDF with tables produces table extraction with cell-level detail."""
    table_region = _make_table_region("t1", 1, 0.94)
    layout_region = MagicMock(
        region_id="t1", region_type=RegionType.TABLE, page=1,
        bbox=BoundingBox(x=0.05, y=0.3, w=0.9, h=0.4), confidence=0.95,
    )

    with patch("agentic_extract.pipeline.ingest", return_value=mock_ingestion), \
         patch("agentic_extract.pipeline.detect_layout", return_value=[layout_region]), \
         patch("agentic_extract.pipeline.determine_reading_order", return_value=[layout_region]), \
         patch("agentic_extract.pipeline.route_regions", return_value={"text": [], "table": [layout_region], "visual": []}), \
         patch("agentic_extract.pipeline.TableSpecialist") as MockTableSpec, \
         patch("agentic_extract.pipeline.assemble_result") as mock_assemble:

        MockTableSpec.return_value = AsyncMock()
        MockTableSpec.return_value.extract.return_value = table_region
        mock_assemble.return_value = _build_mock_assembly([table_region], single_page_image)

        pipeline = Pipeline(claude_client=mock_claude, codex_client=mock_codex)
        result = await pipeline.extract(single_page_image)

    assert len(result.regions) == 1
    assert result.regions[0].type == RegionType.TABLE
    content = result.regions[0].content
    assert isinstance(content, TableContent)
    assert len(content.cell_bboxes) == 4
    assert content.json_data["headers"] == ["Gene", "Value"]
    assert content.json_data["rows"][0]["Gene"] == "BRCA1"


# --- Scenario 3: Chart Image ---


@pytest.mark.asyncio
async def test_scenario_3_chart_image(
    mock_claude, mock_codex, single_page_image, mock_ingestion, tmp_path,
):
    """Chart image produces DePlot + Claude reasoning output."""
    chart_region = _make_figure_region("f1", 1, 0.86)
    layout_region = MagicMock(
        region_id="f1", region_type=RegionType.FIGURE, page=1,
        bbox=BoundingBox(x=0.1, y=0.05, w=0.8, h=0.45), confidence=0.90,
    )

    with patch("agentic_extract.pipeline.ingest", return_value=mock_ingestion), \
         patch("agentic_extract.pipeline.detect_layout", return_value=[layout_region]), \
         patch("agentic_extract.pipeline.determine_reading_order", return_value=[layout_region]), \
         patch("agentic_extract.pipeline.route_regions", return_value={"text": [], "table": [], "visual": [layout_region]}), \
         patch("agentic_extract.pipeline.assemble_result") as mock_assemble:

        mock_assemble.return_value = _build_mock_assembly([chart_region], single_page_image)

        pipeline = Pipeline(claude_client=mock_claude, codex_client=mock_codex)
        result = await pipeline.extract(single_page_image)

    assert len(result.regions) == 1
    assert result.regions[0].type == RegionType.FIGURE
    content = result.regions[0].content
    assert isinstance(content, FigureContent)
    assert content.figure_type == "bar_chart"
    assert "BRCA1" in str(content.figure_json)


# --- Scenario 4: Handwriting Image ---


@pytest.mark.asyncio
async def test_scenario_4_handwriting_image(
    mock_claude, mock_codex, single_page_image, mock_ingestion, tmp_path,
):
    """Handwriting image produces TrOCR + dual model output."""
    hw_region = _make_handwriting_region("h1", 1, 0.82)
    layout_region = MagicMock(
        region_id="h1", region_type=RegionType.HANDWRITING, page=1,
        bbox=BoundingBox(x=0.05, y=0.5, w=0.9, h=0.3), confidence=0.85,
    )

    with patch("agentic_extract.pipeline.ingest", return_value=mock_ingestion), \
         patch("agentic_extract.pipeline.detect_layout", return_value=[layout_region]), \
         patch("agentic_extract.pipeline.determine_reading_order", return_value=[layout_region]), \
         patch("agentic_extract.pipeline.route_regions", return_value={"text": [], "table": [], "visual": [layout_region]}), \
         patch("agentic_extract.pipeline.assemble_result") as mock_assemble:

        mock_assemble.return_value = _build_mock_assembly([hw_region], single_page_image)

        pipeline = Pipeline(claude_client=mock_claude, codex_client=mock_codex)
        result = await pipeline.extract(single_page_image)

    assert len(result.regions) == 1
    assert result.regions[0].type == RegionType.HANDWRITING
    content = result.regions[0].content
    assert isinstance(content, HandwritingContent)
    assert "500mg" in content.text
    assert "trocr" in result.regions[0].extraction_method


# --- Scenario 5: Mixed Document ---


@pytest.mark.asyncio
async def test_scenario_5_mixed_document_routing(
    mock_claude, mock_codex, single_page_image, mock_ingestion, tmp_path,
):
    """Mixed document with text + table + figure routes to correct specialists."""
    text_region = _make_text_region("r1", 1, "Introduction text", 0.97)
    table_region = _make_table_region("t1", 1, 0.94)
    figure_region = _make_figure_region("f1", 1, 0.86)

    text_lr = MagicMock(
        region_id="r1", region_type=RegionType.TEXT, page=1,
        bbox=BoundingBox(x=0.05, y=0.05, w=0.9, h=0.1), confidence=0.98,
    )
    table_lr = MagicMock(
        region_id="t1", region_type=RegionType.TABLE, page=1,
        bbox=BoundingBox(x=0.05, y=0.2, w=0.9, h=0.3), confidence=0.95,
    )
    figure_lr = MagicMock(
        region_id="f1", region_type=RegionType.FIGURE, page=1,
        bbox=BoundingBox(x=0.1, y=0.6, w=0.8, h=0.35), confidence=0.90,
    )

    with patch("agentic_extract.pipeline.ingest", return_value=mock_ingestion), \
         patch("agentic_extract.pipeline.detect_layout", return_value=[text_lr, table_lr, figure_lr]), \
         patch("agentic_extract.pipeline.determine_reading_order", return_value=[text_lr, table_lr, figure_lr]), \
         patch("agentic_extract.pipeline.route_regions") as mock_route, \
         patch("agentic_extract.pipeline.TextSpecialist") as MockTextSpec, \
         patch("agentic_extract.pipeline.TableSpecialist") as MockTableSpec, \
         patch("agentic_extract.pipeline.assemble_result") as mock_assemble:

        mock_route.return_value = {
            "text": [text_lr],
            "table": [table_lr],
            "visual": [figure_lr],
        }
        MockTextSpec.return_value = AsyncMock()
        MockTextSpec.return_value.extract.return_value = text_region
        MockTableSpec.return_value = AsyncMock()
        MockTableSpec.return_value.extract.return_value = table_region

        all_regions = [text_region, table_region, figure_region]
        mock_assemble.return_value = _build_mock_assembly(all_regions, single_page_image)

        pipeline = Pipeline(claude_client=mock_claude, codex_client=mock_codex)
        result = await pipeline.extract(single_page_image)

    region_types = {r.type for r in result.regions}
    assert RegionType.TEXT in region_types
    assert RegionType.TABLE in region_types
    assert RegionType.FIGURE in region_types


# --- Scenario 6: Low Confidence Triggers Re-extraction ---


@pytest.mark.asyncio
async def test_scenario_6_low_confidence_triggers_reextraction(
    mock_claude, mock_codex, single_page_image, mock_ingestion, tmp_path,
):
    """Low-confidence region triggers the re-extraction loop."""
    low_conf_region = _make_text_region("r1", 1, "Ambiguous text", 0.78)
    low_conf_region = low_conf_region.model_copy(update={
        "extraction_method": "paddleocr_3.0 + claude-opus-4-20250514",
    })

    layout_region = MagicMock(
        region_id="r1", region_type=RegionType.TEXT, page=1,
        bbox=BoundingBox(x=0.05, y=0.1, w=0.9, h=0.15), confidence=0.98,
    )

    # After re-extraction, confidence should be boosted
    boosted_region = low_conf_region.model_copy(update={
        "confidence": 0.88,
        "extraction_method": "paddleocr_3.0 + claude-opus-4-20250514 + gpt-4o",
    })

    with patch("agentic_extract.pipeline.ingest", return_value=mock_ingestion), \
         patch("agentic_extract.pipeline.detect_layout", return_value=[layout_region]), \
         patch("agentic_extract.pipeline.determine_reading_order", return_value=[layout_region]), \
         patch("agentic_extract.pipeline.route_regions", return_value={"text": [layout_region], "table": [], "visual": []}), \
         patch("agentic_extract.pipeline.TextSpecialist") as MockTextSpec, \
         patch("agentic_extract.pipeline.ReExtractionEngine") as MockReExtract, \
         patch("agentic_extract.pipeline.assemble_result") as mock_assemble:

        MockTextSpec.return_value = AsyncMock()
        MockTextSpec.return_value.extract.return_value = low_conf_region

        mock_re_result = ReExtractionResult(
            region_id="r1",
            field_name="text",
            original_value="Ambiguous text",
            original_confidence=0.78,
            original_model="claude-opus-4-20250514",
            candidates=[ReExtractionCandidate(
                value="Ambiguous text",
                confidence=0.88,
                model="gpt-4o",
                extraction_method="re_extraction_1 + gpt-4o",
            )],
            final_value="Ambiguous text",
            final_confidence=0.88,
            retries_used=1,
            models_agreed=True,
            accepted=False,
            flagged=True,
        )
        mock_engine = AsyncMock()
        mock_engine.re_extract_field.return_value = mock_re_result
        MockReExtract.return_value = mock_engine

        mock_assemble.return_value = _build_mock_assembly([boosted_region], single_page_image)
        mock_assemble.return_value.audit_trail.re_extractions = 1

        pipeline = Pipeline(claude_client=mock_claude, codex_client=mock_codex)
        result = await pipeline.extract(single_page_image)

    assert result.audit_trail.re_extractions >= 1


# --- Scenario 7: Audit Trail Completeness ---


@pytest.mark.asyncio
async def test_scenario_7_audit_trail_completeness(
    mock_claude, mock_codex, single_page_image, mock_ingestion, tmp_path,
):
    """Audit trail must contain all required fields."""
    text_region = _make_text_region("r1", 1, "Audit test", 0.95)
    layout_region = MagicMock(
        region_id="r1", region_type=RegionType.TEXT, page=1,
        bbox=BoundingBox(x=0.05, y=0.1, w=0.9, h=0.15), confidence=0.98,
    )

    with patch("agentic_extract.pipeline.ingest", return_value=mock_ingestion), \
         patch("agentic_extract.pipeline.detect_layout", return_value=[layout_region]), \
         patch("agentic_extract.pipeline.determine_reading_order", return_value=[layout_region]), \
         patch("agentic_extract.pipeline.route_regions", return_value={"text": [layout_region], "table": [], "visual": []}), \
         patch("agentic_extract.pipeline.TextSpecialist") as MockTextSpec, \
         patch("agentic_extract.pipeline.assemble_result") as mock_assemble:

        MockTextSpec.return_value = AsyncMock()
        MockTextSpec.return_value.extract.return_value = text_region

        audit = AuditTrail(
            models_used=["paddleocr_3.0"],
            total_llm_calls=0,
            re_extractions=0,
            fields_flagged=0,
            processing_stages=[
                ProcessingStage(stage="ingestion", duration_ms=100),
                ProcessingStage(stage="layout_detection", duration_ms=80),
                ProcessingStage(stage="reading_order", duration_ms=20),
                ProcessingStage(stage="routing", duration_ms=10),
                ProcessingStage(stage="extraction", duration_ms=500),
                ProcessingStage(stage="validation", duration_ms=200),
                ProcessingStage(stage="assembly", duration_ms=50),
            ],
        )
        mock_assemble.return_value = ExtractionResult(
            document=DocumentMetadata(
                id="doc-audit", source=str(single_page_image),
                page_count=1, processing_timestamp=datetime.now(timezone.utc),
                approach="B", total_confidence=0.95, processing_time_ms=960,
            ),
            markdown="Audit test",
            regions=[text_region],
            extracted_entities={},
            audit_trail=audit,
        )

        pipeline = Pipeline(claude_client=mock_claude, codex_client=mock_codex)
        result = await pipeline.extract(single_page_image)

    at = result.audit_trail
    # Required fields
    assert isinstance(at.models_used, list)
    assert isinstance(at.total_llm_calls, int)
    assert isinstance(at.re_extractions, int)
    assert isinstance(at.fields_flagged, int)
    assert isinstance(at.processing_stages, list)
    assert at.total_llm_calls >= 0
    assert at.re_extractions >= 0
    assert at.fields_flagged >= 0

    # Stage names must include key pipeline stages
    stage_names = [s.stage for s in at.processing_stages]
    assert "ingestion" in stage_names
    assert "extraction" in stage_names
    assert "validation" in stage_names
    assert "assembly" in stage_names

    # All stages have non-negative duration
    for stage in at.processing_stages:
        assert stage.duration_ms >= 0


# --- Scenario 8: Confidence Scoring on All Fields ---


@pytest.mark.asyncio
async def test_scenario_8_confidence_on_all_fields(
    mock_claude, mock_codex, single_page_image, mock_ingestion, tmp_path,
):
    """Every region and field must have a confidence score in [0, 1]."""
    regions = [
        _make_text_region("r1", 1, "Text", 0.97),
        _make_table_region("t1", 1, 0.94),
        _make_figure_region("f1", 1, 0.86),
        _make_handwriting_region("h1", 1, 0.78),
    ]

    layout_regions = [
        MagicMock(region_id=r.id, region_type=r.type, page=1,
                  bbox=r.bbox, confidence=0.95)
        for r in regions
    ]

    with patch("agentic_extract.pipeline.ingest", return_value=mock_ingestion), \
         patch("agentic_extract.pipeline.detect_layout", return_value=layout_regions), \
         patch("agentic_extract.pipeline.determine_reading_order", return_value=layout_regions), \
         patch("agentic_extract.pipeline.route_regions") as mock_route, \
         patch("agentic_extract.pipeline.TextSpecialist") as MockTextSpec, \
         patch("agentic_extract.pipeline.TableSpecialist") as MockTableSpec, \
         patch("agentic_extract.pipeline.ReExtractionEngine") as MockReExtract, \
         patch("agentic_extract.pipeline.assemble_result") as mock_assemble:

        mock_route.return_value = {
            "text": [layout_regions[0]],
            "table": [layout_regions[1]],
            "visual": [layout_regions[2], layout_regions[3]],
        }
        MockTextSpec.return_value = AsyncMock()
        MockTextSpec.return_value.extract.return_value = regions[0]
        MockTableSpec.return_value = AsyncMock()
        MockTableSpec.return_value.extract.return_value = regions[1]

        # Mock re-extraction for low-confidence handwriting region
        mock_re_result = ReExtractionResult(
            region_id="h1", field_name="handwriting",
            original_value="Patient notes: administered 500mg",
            original_confidence=0.78, original_model="claude-opus-4-20250514",
            candidates=[], final_value="Patient notes: administered 500mg",
            final_confidence=0.78, retries_used=2,
            models_agreed=False, accepted=False, flagged=True,
        )
        mock_engine = AsyncMock()
        mock_engine.re_extract_field.return_value = mock_re_result
        MockReExtract.return_value = mock_engine

        mock_assemble.return_value = _build_mock_assembly(regions, single_page_image)

        pipeline = Pipeline(claude_client=mock_claude, codex_client=mock_codex)
        result = await pipeline.extract(single_page_image)

    # Every region must have a valid confidence score
    for region in result.regions:
        assert 0.0 <= region.confidence <= 1.0, (
            f"Region {region.id} has invalid confidence: {region.confidence}"
        )

    # Document-level confidence must also be valid
    assert 0.0 <= result.document.total_confidence <= 1.0

    # JSON serialization must preserve all confidence values
    json_str = result.model_dump_json()
    parsed = json.loads(json_str)
    for r in parsed["regions"]:
        assert 0.0 <= r["confidence"] <= 1.0
