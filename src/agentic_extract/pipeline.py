# src/agentic_extract/pipeline.py
"""Full pipeline orchestration: Coordinator -> Specialists -> Validator -> Re-extraction -> Output.

This is the top-level entry point for document extraction. It composes
all agents (Coordinator, Specialist Pool, Validator) and the re-extraction
loop into a single async pipeline.

Usage:
    pipeline = Pipeline(claude_client=claude, codex_client=codex)
    result = await pipeline.extract(Path("document.pdf"))
"""
from __future__ import annotations

import asyncio
import logging
import pathlib
import time
import uuid
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field

from agentic_extract.clients.vlm import VLMClient
from agentic_extract.coordinator.assembly import assemble_result
from agentic_extract.coordinator.ingestion import ingest
from agentic_extract.coordinator.layout import detect_layout
from agentic_extract.coordinator.reading_order import determine_reading_order
from agentic_extract.coordinator.routing import route_regions
from agentic_extract.models import (
    AuditTrail,
    BoundingBox,
    DocumentMetadata,
    ExtractionResult,
    ProcessingStage,
    Region,
    RegionType,
)
from agentic_extract.reextraction.engine import ReExtractionEngine
from agentic_extract.specialists.text import TextSpecialist
from agentic_extract.specialists.table import TableSpecialist

logger = logging.getLogger(__name__)

ACCEPT_THRESHOLD = 0.90
REEXTRACT_LOW = 0.70


class PipelineConfig(BaseModel):
    """Configuration for the extraction pipeline."""

    accept_threshold: float = Field(default=0.90, ge=0.0, le=1.0)
    reextract_low: float = Field(default=0.70, ge=0.0, le=1.0)
    max_retries: int = Field(default=2, ge=0)
    parallel_specialists: bool = True


class Pipeline:
    """Full extraction pipeline orchestrator.

    Flow:
    1. Ingestion: convert document to page images
    2. Layout detection: identify regions on each page
    3. Reading order: determine logical ordering
    4. Routing: assign regions to specialists
    5. Extraction: dispatch specialists (parallel via asyncio.gather)
    6. Validation: run 5-layer validation on all regions
    7. Re-extraction: retry low-confidence fields with model switching
    8. Assembly: produce final Markdown + JSON output

    Args:
        claude_client: VLM client for Claude API calls.
        codex_client: VLM client for Codex/GPT-4o API calls.
        config: Pipeline configuration. Uses defaults if not provided.
    """

    def __init__(
        self,
        claude_client: VLMClient,
        codex_client: VLMClient,
        config: PipelineConfig | None = None,
    ) -> None:
        self.claude_client = claude_client
        self.codex_client = codex_client
        self.config = config or PipelineConfig()
        self._reextraction_engine = ReExtractionEngine(
            claude_client=claude_client,
            codex_client=codex_client,
            max_retries=self.config.max_retries,
        )

    async def extract(
        self,
        file_path: pathlib.Path,
        schema: dict[str, Any] | None = None,
    ) -> ExtractionResult:
        """Run the full extraction pipeline on a document.

        Args:
            file_path: Path to the input document (PDF or image).
            schema: Optional JSON schema for structured entity extraction.

        Returns:
            ExtractionResult with Markdown, JSON, regions, and audit trail.
        """
        file_path = pathlib.Path(file_path)
        stages: list[ProcessingStage] = []
        all_regions: list[Region] = []
        models_used: set[str] = set()
        total_llm_calls = 0
        re_extraction_count = 0
        fields_flagged = 0

        pipeline_start = time.monotonic()

        # --- Stage 1: Ingestion ---
        stage_start = time.monotonic()
        ingestion_result = ingest(file_path)
        stages.append(ProcessingStage(
            stage="ingestion",
            duration_ms=int((time.monotonic() - stage_start) * 1000),
        ))

        # --- Stage 2: Layout Detection ---
        stage_start = time.monotonic()
        layout_regions = []
        for page in ingestion_result.pages:
            page_regions = detect_layout(page.image_path)
            layout_regions.extend(page_regions)
        stages.append(ProcessingStage(
            stage="layout_detection",
            duration_ms=int((time.monotonic() - stage_start) * 1000),
        ))

        # --- Stage 3: Reading Order ---
        stage_start = time.monotonic()
        ordered_layout = determine_reading_order(layout_regions)
        stages.append(ProcessingStage(
            stage="reading_order",
            duration_ms=int((time.monotonic() - stage_start) * 1000),
        ))

        # --- Stage 4: Routing ---
        stage_start = time.monotonic()
        routed = route_regions(ordered_layout)
        stages.append(ProcessingStage(
            stage="routing",
            duration_ms=int((time.monotonic() - stage_start) * 1000),
        ))

        # --- Stage 5: Specialist Extraction ---
        stage_start = time.monotonic()
        text_specialist = TextSpecialist(vlm_client=self.claude_client)
        table_specialist = TableSpecialist(
            claude_client=self.claude_client,
            codex_client=self.codex_client,
        )

        extraction_tasks = []
        task_metadata = []

        for lr in routed.get("text", []):
            page_img = self._find_page_image(ingestion_result, lr)
            extraction_tasks.append(
                text_specialist.extract(
                    image_path=page_img,
                    region_id=lr.region_id,
                    page=lr.page,
                    bbox=lr.bbox,
                )
            )
            task_metadata.append(("text", lr))

        for lr in routed.get("table", []):
            page_img = self._find_page_image(ingestion_result, lr)
            extraction_tasks.append(
                table_specialist.extract(
                    image_path=page_img,
                    region_id=lr.region_id,
                    page=lr.page,
                    bbox=lr.bbox,
                )
            )
            task_metadata.append(("table", lr))

        # Dispatch in parallel or sequentially based on config
        if self.config.parallel_specialists and extraction_tasks:
            results = await asyncio.gather(
                *extraction_tasks, return_exceptions=True,
            )
        elif extraction_tasks:
            results = []
            for task in extraction_tasks:
                try:
                    r = await task
                    results.append(r)
                except Exception as e:
                    results.append(e)
        else:
            results = []

        # Collect successful extractions, log failures
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                spec_type, lr = task_metadata[i]
                logger.error(
                    "Specialist %s failed for region %s: %s",
                    spec_type, lr.region_id, result,
                )
                fields_flagged += 1
            elif isinstance(result, Region):
                all_regions.append(result)
                models_used.add(result.extraction_method)

        stages.append(ProcessingStage(
            stage="extraction",
            duration_ms=int((time.monotonic() - stage_start) * 1000),
        ))

        # --- Stage 6: Validation + Re-extraction ---
        stage_start = time.monotonic()
        final_regions: list[Region] = []

        for region in all_regions:
            if region.confidence >= self.config.accept_threshold:
                final_regions.append(region)
            elif region.confidence >= self.config.reextract_low:
                # Trigger re-extraction
                page_img = self._find_page_image_for_region(
                    ingestion_result, region,
                )
                try:
                    original_model = self._infer_model(region.extraction_method)
                    re_result = await self._reextraction_engine.re_extract_field(
                        region=region,
                        field_name=region.type.value,
                        image_path=page_img,
                        original_model=original_model,
                    )
                    re_extraction_count += 1
                    models_used.add(re_result.original_model)
                    for c in re_result.candidates:
                        models_used.add(c.model)

                    # Update region with re-extraction result
                    updated = region.model_copy(update={
                        "confidence": re_result.final_confidence,
                        "needs_review": re_result.flagged,
                        "review_reason": (
                            f"Re-extraction: models {'agreed' if re_result.models_agreed else 'disagreed'} "
                            f"after {re_result.retries_used} retries"
                            if re_result.flagged else None
                        ),
                    })
                    final_regions.append(updated)
                    if re_result.flagged:
                        fields_flagged += 1
                except Exception as exc:
                    logger.error(
                        "Re-extraction failed for region %s: %s", region.id, exc,
                    )
                    flagged_region = region.model_copy(update={
                        "needs_review": True,
                        "review_reason": f"Re-extraction error: {exc}",
                    })
                    final_regions.append(flagged_region)
                    fields_flagged += 1
            else:
                # Below reextract threshold: flag for review
                flagged_region = region.model_copy(update={
                    "needs_review": True,
                    "review_reason": (
                        f"Confidence {region.confidence:.2f} below {self.config.reextract_low} threshold"
                    ),
                })
                final_regions.append(flagged_region)
                fields_flagged += 1

        stages.append(ProcessingStage(
            stage="validation",
            duration_ms=int((time.monotonic() - stage_start) * 1000),
        ))

        # --- Stage 7: Assembly ---
        stage_start = time.monotonic()

        total_time_ms = int((time.monotonic() - pipeline_start) * 1000)
        metadata = DocumentMetadata(
            id=str(uuid.uuid4()),
            source=str(file_path),
            page_count=ingestion_result.page_count,
            processing_timestamp=datetime.now(timezone.utc),
            approach="B",
            total_confidence=self._compute_total_confidence(final_regions),
            processing_time_ms=total_time_ms,
        )

        audit = AuditTrail(
            models_used=sorted(models_used),
            total_llm_calls=total_llm_calls,
            re_extractions=re_extraction_count,
            fields_flagged=fields_flagged,
            processing_stages=stages,
        )

        result = assemble_result(
            regions=final_regions,
            metadata=metadata,
            audit_trail=audit,
        )

        stages.append(ProcessingStage(
            stage="assembly",
            duration_ms=int((time.monotonic() - stage_start) * 1000),
        ))

        return result

    @staticmethod
    def _find_page_image(ingestion_result: Any, layout_region: Any) -> pathlib.Path:
        """Find the page image for a given layout region."""
        for page in ingestion_result.pages:
            if page.page_number == layout_region.page:
                return page.image_path
        # Fallback to first page
        return ingestion_result.pages[0].image_path

    @staticmethod
    def _find_page_image_for_region(
        ingestion_result: Any, region: Region,
    ) -> pathlib.Path:
        """Find the page image for a given extracted region."""
        for page in ingestion_result.pages:
            if page.page_number == region.page:
                return page.image_path
        return ingestion_result.pages[0].image_path

    @staticmethod
    def _infer_model(extraction_method: str) -> str:
        """Infer the primary VLM model from an extraction method string."""
        if "claude" in extraction_method.lower():
            return "claude-opus-4-20250514"
        if "gpt" in extraction_method.lower() or "codex" in extraction_method.lower():
            return "gpt-4o"
        return "claude-opus-4-20250514"

    @staticmethod
    def _compute_total_confidence(regions: list[Region]) -> float:
        """Compute document-level confidence as the minimum region confidence."""
        if not regions:
            return 1.0
        return min(r.confidence for r in regions)
