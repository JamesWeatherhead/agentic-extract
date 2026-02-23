# src/agentic_extract/audit.py
"""Audit trail tracker that accumulates metrics throughout the pipeline.

Integrates with the Pipeline class to record:
- models_used: all tools and VLMs invoked
- total_llm_calls: count of Claude/Codex API calls
- re_extractions: count of model-switch re-extraction attempts
- fields_flagged: count of fields marked needs_review
- per-stage timing: duration of each pipeline stage
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Generator

from agentic_extract.models import AuditTrail, ProcessingStage


class AuditTrailTracker:
    """Mutable tracker that accumulates audit data throughout pipeline execution.

    Usage:
        tracker = AuditTrailTracker()
        tracker.record_model("paddleocr_3.0")
        with tracker.stage("ingestion"):
            ... do ingestion ...
        tracker.record_llm_call()
        audit = tracker.build()  # Produces immutable AuditTrail
    """

    def __init__(self) -> None:
        self.models_used: set[str] = set()
        self.total_llm_calls: int = 0
        self.re_extractions: int = 0
        self.fields_flagged: int = 0
        self.stages: list[ProcessingStage] = []
        self._stage_starts: dict[str, float] = {}

    def record_model(self, model_name: str) -> None:
        """Record that a model or tool was used."""
        self.models_used.add(model_name)

    def record_llm_call(self) -> None:
        """Record a single LLM API call (Claude or Codex)."""
        self.total_llm_calls += 1

    def record_reextraction(self) -> None:
        """Record a re-extraction attempt."""
        self.re_extractions += 1

    def record_flagged_field(self) -> None:
        """Record a field that was flagged for review."""
        self.fields_flagged += 1

    def start_stage(self, name: str) -> None:
        """Mark the start of a pipeline stage for timing."""
        self._stage_starts[name] = time.monotonic()

    def stop_stage(self, name: str) -> None:
        """Mark the end of a pipeline stage and record its duration."""
        start = self._stage_starts.pop(name, None)
        if start is None:
            return
        duration_ms = int((time.monotonic() - start) * 1000)
        self.stages.append(ProcessingStage(stage=name, duration_ms=duration_ms))

    @contextmanager
    def stage(self, name: str) -> Generator[None, None, None]:
        """Context manager for timing a pipeline stage.

        The stage is recorded even if an exception occurs inside the block.
        """
        self.start_stage(name)
        try:
            yield
        finally:
            self.stop_stage(name)

    def merge(self, other: AuditTrailTracker) -> None:
        """Merge another tracker's data into this one.

        Useful when combining results from parallel specialist dispatch.
        """
        self.models_used |= other.models_used
        self.total_llm_calls += other.total_llm_calls
        self.re_extractions += other.re_extractions
        self.fields_flagged += other.fields_flagged
        self.stages.extend(other.stages)

    def reset(self) -> None:
        """Clear all accumulated data."""
        self.models_used.clear()
        self.total_llm_calls = 0
        self.re_extractions = 0
        self.fields_flagged = 0
        self.stages.clear()
        self._stage_starts.clear()

    def build(self) -> AuditTrail:
        """Build an immutable AuditTrail from the accumulated data.

        Returns:
            AuditTrail Pydantic model ready for inclusion in ExtractionResult.
        """
        return AuditTrail(
            models_used=sorted(self.models_used),
            total_llm_calls=self.total_llm_calls,
            re_extractions=self.re_extractions,
            fields_flagged=self.fields_flagged,
            processing_stages=list(self.stages),
        )
