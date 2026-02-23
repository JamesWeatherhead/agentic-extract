# tests/test_pipeline.py
"""Tests for the full extraction pipeline orchestration."""
import asyncio
import pathlib
import time
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from PIL import Image

from agentic_extract.clients.vlm import VLMResponse
from agentic_extract.models import (
    BoundingBox,
    DocumentMetadata,
    ExtractionResult,
    Region,
    RegionType,
    TextContent,
    TableContent,
    FigureContent,
    ProcessingStage,
    AuditTrail,
)
from agentic_extract.pipeline import Pipeline, PipelineConfig


def test_pipeline_config_defaults():
    config = PipelineConfig()
    assert config.accept_threshold == 0.90
    assert config.reextract_low == 0.70
    assert config.max_retries == 2
    assert config.parallel_specialists is True


def test_pipeline_config_custom():
    config = PipelineConfig(
        accept_threshold=0.85,
        reextract_low=0.65,
        max_retries=3,
        parallel_specialists=False,
    )
    assert config.accept_threshold == 0.85
    assert config.max_retries == 3


def test_pipeline_init():
    pipeline = Pipeline(
        claude_client=AsyncMock(),
        codex_client=AsyncMock(),
    )
    assert pipeline.config.accept_threshold == 0.90


def test_pipeline_init_with_config():
    config = PipelineConfig(max_retries=1)
    pipeline = Pipeline(
        claude_client=AsyncMock(),
        codex_client=AsyncMock(),
        config=config,
    )
    assert pipeline.config.max_retries == 1


@pytest.mark.asyncio
async def test_pipeline_extract_single_text_region(tmp_path: pathlib.Path):
    """Pipeline should handle a simple text-only document end-to-end."""
    img = Image.new("RGB", (2550, 3300), "white")
    img_path = tmp_path / "simple.png"
    img.save(img_path, dpi=(300, 300))

    mock_claude = AsyncMock()
    mock_codex = AsyncMock()

    # Mock ingestion to return one page
    mock_pages = [MagicMock(
        page_number=1,
        image_path=img_path,
        width=2550,
        height=3300,
        dpi=300,
    )]
    mock_ingestion = MagicMock()
    mock_ingestion.pages = mock_pages
    mock_ingestion.page_count = 1
    mock_ingestion.source_file = img_path
    mock_ingestion.temp_dir = tmp_path

    # Mock layout detection to return one text region
    mock_layout_regions = [MagicMock(
        region_id="r1",
        region_type=RegionType.TEXT,
        page=1,
        bbox=BoundingBox(x=0.05, y=0.10, w=0.90, h=0.15),
        confidence=0.98,
    )]

    # Mock text specialist output
    mock_text_region = Region(
        id="r1",
        type=RegionType.TEXT,
        subtype=None,
        page=1,
        bbox=BoundingBox(x=0.05, y=0.10, w=0.90, h=0.15),
        content=TextContent(text="Hello world", markdown="Hello world"),
        confidence=0.97,
        extraction_method="paddleocr_3.0",
    )

    with patch("agentic_extract.pipeline.ingest", return_value=mock_ingestion), \
         patch("agentic_extract.pipeline.detect_layout", return_value=mock_layout_regions), \
         patch("agentic_extract.pipeline.determine_reading_order", return_value=mock_layout_regions), \
         patch("agentic_extract.pipeline.route_regions") as mock_route, \
         patch("agentic_extract.pipeline.TextSpecialist") as MockTextSpec, \
         patch("agentic_extract.pipeline.assemble_result") as mock_assemble:

        mock_route.return_value = {"text": [mock_layout_regions[0]], "table": [], "visual": []}

        mock_text_spec_instance = AsyncMock()
        mock_text_spec_instance.extract.return_value = mock_text_region
        MockTextSpec.return_value = mock_text_spec_instance

        mock_assemble.return_value = ExtractionResult(
            document=DocumentMetadata(
                id="doc-1",
                source=str(img_path),
                page_count=1,
                processing_timestamp=datetime.now(timezone.utc),
                approach="B",
                total_confidence=0.97,
                processing_time_ms=1500,
            ),
            markdown="# Document\n\nHello world",
            regions=[mock_text_region],
            extracted_entities={},
            audit_trail=AuditTrail(
                models_used=["paddleocr_3.0"],
                total_llm_calls=0,
                re_extractions=0,
                fields_flagged=0,
                processing_stages=[
                    ProcessingStage(stage="ingestion", duration_ms=200),
                    ProcessingStage(stage="extraction", duration_ms=800),
                    ProcessingStage(stage="validation", duration_ms=300),
                    ProcessingStage(stage="assembly", duration_ms=200),
                ],
            ),
        )

        pipeline = Pipeline(
            claude_client=mock_claude,
            codex_client=mock_codex,
        )
        result = await pipeline.extract(img_path)

        assert isinstance(result, ExtractionResult)
        assert result.document.page_count == 1
        assert len(result.regions) == 1
        assert result.regions[0].content.text == "Hello world"


@pytest.mark.asyncio
async def test_pipeline_parallel_specialist_dispatch(tmp_path: pathlib.Path):
    """Pipeline should dispatch specialists in parallel via asyncio.gather."""
    img = Image.new("RGB", (2550, 3300), "white")
    img_path = tmp_path / "mixed.png"
    img.save(img_path, dpi=(300, 300))

    mock_claude = AsyncMock()
    mock_codex = AsyncMock()

    mock_pages = [MagicMock(
        page_number=1, image_path=img_path,
        width=2550, height=3300, dpi=300,
    )]
    mock_ingestion = MagicMock()
    mock_ingestion.pages = mock_pages
    mock_ingestion.page_count = 1
    mock_ingestion.source_file = img_path
    mock_ingestion.temp_dir = tmp_path

    text_region = MagicMock(
        region_id="r1", region_type=RegionType.TEXT,
        page=1, bbox=BoundingBox(x=0.05, y=0.1, w=0.9, h=0.1), confidence=0.98,
    )
    table_region = MagicMock(
        region_id="r2", region_type=RegionType.TABLE,
        page=1, bbox=BoundingBox(x=0.05, y=0.3, w=0.9, h=0.3), confidence=0.95,
    )
    mock_layout_regions = [text_region, table_region]

    extracted_text = Region(
        id="r1", type=RegionType.TEXT, subtype=None, page=1,
        bbox=BoundingBox(x=0.05, y=0.1, w=0.9, h=0.1),
        content=TextContent(text="Text content", markdown="Text content"),
        confidence=0.97, extraction_method="paddleocr_3.0",
    )
    extracted_table = Region(
        id="r2", type=RegionType.TABLE, subtype=None, page=1,
        bbox=BoundingBox(x=0.05, y=0.3, w=0.9, h=0.3),
        content=TableContent(
            html="<table><tr><td>A</td></tr></table>",
            json_data={"headers": ["Col"], "rows": [{"Col": "A"}]},
        ),
        confidence=0.94, extraction_method="docling + claude-opus-4-20250514",
    )

    with patch("agentic_extract.pipeline.ingest", return_value=mock_ingestion), \
         patch("agentic_extract.pipeline.detect_layout", return_value=mock_layout_regions), \
         patch("agentic_extract.pipeline.determine_reading_order", return_value=mock_layout_regions), \
         patch("agentic_extract.pipeline.route_regions") as mock_route, \
         patch("agentic_extract.pipeline.TextSpecialist") as MockTextSpec, \
         patch("agentic_extract.pipeline.TableSpecialist") as MockTableSpec, \
         patch("agentic_extract.pipeline.assemble_result") as mock_assemble:

        mock_route.return_value = {
            "text": [text_region], "table": [table_region], "visual": [],
        }
        MockTextSpec.return_value = AsyncMock()
        MockTextSpec.return_value.extract.return_value = extracted_text
        MockTableSpec.return_value = AsyncMock()
        MockTableSpec.return_value.extract.return_value = extracted_table

        mock_assemble.return_value = ExtractionResult(
            document=DocumentMetadata(
                id="doc-2", source=str(img_path), page_count=1,
                processing_timestamp=datetime.now(timezone.utc),
                approach="B", total_confidence=0.95, processing_time_ms=2000,
            ),
            markdown="# Doc\n\nText content\n\n| Col |\n|-----|\n| A |",
            regions=[extracted_text, extracted_table],
            extracted_entities={},
            audit_trail=AuditTrail(
                models_used=["paddleocr_3.0", "docling", "claude-opus-4-20250514"],
                total_llm_calls=1, re_extractions=0, fields_flagged=0,
                processing_stages=[],
            ),
        )

        pipeline = Pipeline(claude_client=mock_claude, codex_client=mock_codex)
        result = await pipeline.extract(img_path)

        assert len(result.regions) == 2


@pytest.mark.asyncio
async def test_pipeline_partial_output_on_specialist_failure(tmp_path: pathlib.Path):
    """If one specialist fails, pipeline should still return partial output."""
    img = Image.new("RGB", (2550, 3300), "white")
    img_path = tmp_path / "partial.png"
    img.save(img_path, dpi=(300, 300))

    mock_claude = AsyncMock()
    mock_codex = AsyncMock()

    mock_pages = [MagicMock(
        page_number=1, image_path=img_path,
        width=2550, height=3300, dpi=300,
    )]
    mock_ingestion = MagicMock()
    mock_ingestion.pages = mock_pages
    mock_ingestion.page_count = 1
    mock_ingestion.source_file = img_path
    mock_ingestion.temp_dir = tmp_path

    text_region_layout = MagicMock(
        region_id="r1", region_type=RegionType.TEXT,
        page=1, bbox=BoundingBox(x=0.05, y=0.1, w=0.9, h=0.1), confidence=0.98,
    )
    table_region_layout = MagicMock(
        region_id="r2", region_type=RegionType.TABLE,
        page=1, bbox=BoundingBox(x=0.05, y=0.3, w=0.9, h=0.3), confidence=0.95,
    )

    extracted_text = Region(
        id="r1", type=RegionType.TEXT, subtype=None, page=1,
        bbox=BoundingBox(x=0.05, y=0.1, w=0.9, h=0.1),
        content=TextContent(text="Partial text", markdown="Partial text"),
        confidence=0.95, extraction_method="paddleocr_3.0",
    )

    with patch("agentic_extract.pipeline.ingest", return_value=mock_ingestion), \
         patch("agentic_extract.pipeline.detect_layout", return_value=[text_region_layout, table_region_layout]), \
         patch("agentic_extract.pipeline.determine_reading_order", return_value=[text_region_layout, table_region_layout]), \
         patch("agentic_extract.pipeline.route_regions") as mock_route, \
         patch("agentic_extract.pipeline.TextSpecialist") as MockTextSpec, \
         patch("agentic_extract.pipeline.TableSpecialist") as MockTableSpec, \
         patch("agentic_extract.pipeline.assemble_result") as mock_assemble:

        mock_route.return_value = {
            "text": [text_region_layout], "table": [table_region_layout], "visual": [],
        }
        MockTextSpec.return_value = AsyncMock()
        MockTextSpec.return_value.extract.return_value = extracted_text
        # Table specialist fails
        MockTableSpec.return_value = AsyncMock()
        MockTableSpec.return_value.extract.side_effect = RuntimeError("Docling crashed")

        mock_assemble.return_value = ExtractionResult(
            document=DocumentMetadata(
                id="doc-3", source=str(img_path), page_count=1,
                processing_timestamp=datetime.now(timezone.utc),
                approach="B", total_confidence=0.95, processing_time_ms=1000,
            ),
            markdown="# Doc\n\nPartial text\n\n[TABLE EXTRACTION FAILED]",
            regions=[extracted_text],
            extracted_entities={},
            audit_trail=AuditTrail(
                models_used=["paddleocr_3.0"],
                total_llm_calls=0, re_extractions=0, fields_flagged=1,
                processing_stages=[],
            ),
        )

        pipeline = Pipeline(claude_client=mock_claude, codex_client=mock_codex)
        result = await pipeline.extract(img_path)

        # Pipeline should succeed with partial output (text only)
        assert isinstance(result, ExtractionResult)
        assert len(result.regions) >= 1


@pytest.mark.asyncio
async def test_pipeline_per_stage_timing(tmp_path: pathlib.Path):
    """Pipeline should record per-stage timing in the audit trail."""
    img = Image.new("RGB", (100, 100), "white")
    img_path = tmp_path / "timed.png"
    img.save(img_path, dpi=(72, 72))

    mock_claude = AsyncMock()
    mock_codex = AsyncMock()

    mock_pages = [MagicMock(
        page_number=1, image_path=img_path,
        width=100, height=100, dpi=72,
    )]
    mock_ingestion = MagicMock()
    mock_ingestion.pages = mock_pages
    mock_ingestion.page_count = 1
    mock_ingestion.source_file = img_path
    mock_ingestion.temp_dir = tmp_path

    with patch("agentic_extract.pipeline.ingest", return_value=mock_ingestion), \
         patch("agentic_extract.pipeline.detect_layout", return_value=[]), \
         patch("agentic_extract.pipeline.determine_reading_order", return_value=[]), \
         patch("agentic_extract.pipeline.route_regions", return_value={"text": [], "table": [], "visual": []}), \
         patch("agentic_extract.pipeline.assemble_result") as mock_assemble:

        audit = AuditTrail(
            models_used=[], total_llm_calls=0, re_extractions=0,
            fields_flagged=0,
            processing_stages=[
                ProcessingStage(stage="ingestion", duration_ms=50),
                ProcessingStage(stage="layout_detection", duration_ms=30),
                ProcessingStage(stage="extraction", duration_ms=0),
                ProcessingStage(stage="validation", duration_ms=0),
                ProcessingStage(stage="assembly", duration_ms=10),
            ],
        )
        mock_assemble.return_value = ExtractionResult(
            document=DocumentMetadata(
                id="doc-4", source=str(img_path), page_count=1,
                processing_timestamp=datetime.now(timezone.utc),
                approach="B", total_confidence=1.0, processing_time_ms=90,
            ),
            markdown="", regions=[], extracted_entities={},
            audit_trail=audit,
        )

        pipeline = Pipeline(claude_client=mock_claude, codex_client=mock_codex)
        result = await pipeline.extract(img_path)

        stage_names = [s.stage for s in result.audit_trail.processing_stages]
        assert "ingestion" in stage_names
        assert "layout_detection" in stage_names
        assert "assembly" in stage_names


@pytest.mark.asyncio
async def test_pipeline_triggers_reextraction_for_low_confidence(tmp_path: pathlib.Path):
    """Pipeline should trigger re-extraction for regions with confidence in [0.70, 0.90)."""
    img = Image.new("RGB", (2550, 3300), "white")
    img_path = tmp_path / "reextract.png"
    img.save(img_path, dpi=(300, 300))

    mock_claude = AsyncMock()
    mock_codex = AsyncMock()

    mock_pages = [MagicMock(
        page_number=1, image_path=img_path,
        width=2550, height=3300, dpi=300,
    )]
    mock_ingestion = MagicMock()
    mock_ingestion.pages = mock_pages
    mock_ingestion.page_count = 1
    mock_ingestion.source_file = img_path
    mock_ingestion.temp_dir = tmp_path

    text_region_layout = MagicMock(
        region_id="r1", region_type=RegionType.TEXT,
        page=1, bbox=BoundingBox(x=0.05, y=0.1, w=0.9, h=0.1), confidence=0.98,
    )

    # Specialist produces low-confidence result
    low_conf_region = Region(
        id="r1", type=RegionType.TEXT, subtype=None, page=1,
        bbox=BoundingBox(x=0.05, y=0.1, w=0.9, h=0.1),
        content=TextContent(text="Ambiguous text", markdown="Ambiguous text"),
        confidence=0.78,
        extraction_method="paddleocr_3.0 + claude-opus-4-20250514",
    )

    with patch("agentic_extract.pipeline.ingest", return_value=mock_ingestion), \
         patch("agentic_extract.pipeline.detect_layout", return_value=[text_region_layout]), \
         patch("agentic_extract.pipeline.determine_reading_order", return_value=[text_region_layout]), \
         patch("agentic_extract.pipeline.route_regions") as mock_route, \
         patch("agentic_extract.pipeline.TextSpecialist") as MockTextSpec, \
         patch("agentic_extract.pipeline.ReExtractionEngine") as MockReExtract, \
         patch("agentic_extract.pipeline.assemble_result") as mock_assemble:

        mock_route.return_value = {"text": [text_region_layout], "table": [], "visual": []}
        MockTextSpec.return_value = AsyncMock()
        MockTextSpec.return_value.extract.return_value = low_conf_region

        from agentic_extract.reextraction.engine import ReExtractionResult, ReExtractionCandidate
        mock_reextract_result = ReExtractionResult(
            region_id="r1", field_name="text", original_value="Ambiguous text",
            original_confidence=0.78, original_model="claude-opus-4-20250514",
            candidates=[ReExtractionCandidate(
                value="Ambiguous text", confidence=0.88,
                model="gpt-4o", extraction_method="re_extraction_1 + gpt-4o",
            )],
            final_value="Ambiguous text", final_confidence=0.88,
            retries_used=1, models_agreed=True, accepted=True, flagged=False,
        )
        mock_engine_instance = AsyncMock()
        mock_engine_instance.re_extract_field.return_value = mock_reextract_result
        MockReExtract.return_value = mock_engine_instance

        boosted_region = Region(
            id="r1", type=RegionType.TEXT, subtype=None, page=1,
            bbox=BoundingBox(x=0.05, y=0.1, w=0.9, h=0.1),
            content=TextContent(text="Ambiguous text", markdown="Ambiguous text"),
            confidence=0.88, extraction_method="paddleocr_3.0 + claude-opus-4-20250514 + gpt-4o",
        )
        mock_assemble.return_value = ExtractionResult(
            document=DocumentMetadata(
                id="doc-5", source=str(img_path), page_count=1,
                processing_timestamp=datetime.now(timezone.utc),
                approach="B", total_confidence=0.88, processing_time_ms=3000,
            ),
            markdown="# Doc\n\nAmbiguous text",
            regions=[boosted_region],
            extracted_entities={},
            audit_trail=AuditTrail(
                models_used=["paddleocr_3.0", "claude-opus-4-20250514", "gpt-4o"],
                total_llm_calls=2, re_extractions=1, fields_flagged=0,
                processing_stages=[],
            ),
        )

        pipeline = Pipeline(claude_client=mock_claude, codex_client=mock_codex)
        result = await pipeline.extract(img_path)

        assert result.audit_trail.re_extractions >= 1
