# tests/test_audit.py
"""Tests for the audit trail tracker that accumulates throughout the pipeline."""
import time

import pytest

from agentic_extract.audit import AuditTrailTracker
from agentic_extract.models import AuditTrail, ProcessingStage


def test_tracker_init():
    tracker = AuditTrailTracker()
    assert tracker.total_llm_calls == 0
    assert tracker.re_extractions == 0
    assert tracker.fields_flagged == 0
    assert len(tracker.models_used) == 0
    assert len(tracker.stages) == 0


def test_tracker_record_model():
    tracker = AuditTrailTracker()
    tracker.record_model("paddleocr_3.0")
    tracker.record_model("claude-opus-4-20250514")
    tracker.record_model("paddleocr_3.0")  # duplicate
    assert len(tracker.models_used) == 2
    assert "paddleocr_3.0" in tracker.models_used
    assert "claude-opus-4-20250514" in tracker.models_used


def test_tracker_record_llm_call():
    tracker = AuditTrailTracker()
    tracker.record_llm_call()
    tracker.record_llm_call()
    tracker.record_llm_call()
    assert tracker.total_llm_calls == 3


def test_tracker_record_reextraction():
    tracker = AuditTrailTracker()
    tracker.record_reextraction()
    assert tracker.re_extractions == 1
    tracker.record_reextraction()
    assert tracker.re_extractions == 2


def test_tracker_record_flagged_field():
    tracker = AuditTrailTracker()
    tracker.record_flagged_field()
    tracker.record_flagged_field()
    assert tracker.fields_flagged == 2


def test_tracker_start_stop_stage():
    tracker = AuditTrailTracker()
    tracker.start_stage("ingestion")
    time.sleep(0.01)  # Ensure non-zero duration
    tracker.stop_stage("ingestion")
    assert len(tracker.stages) == 1
    assert tracker.stages[0].stage == "ingestion"
    assert tracker.stages[0].duration_ms >= 0


def test_tracker_multiple_stages():
    tracker = AuditTrailTracker()
    tracker.start_stage("ingestion")
    tracker.stop_stage("ingestion")
    tracker.start_stage("layout_detection")
    tracker.stop_stage("layout_detection")
    tracker.start_stage("extraction")
    tracker.stop_stage("extraction")
    assert len(tracker.stages) == 3
    stage_names = [s.stage for s in tracker.stages]
    assert stage_names == ["ingestion", "layout_detection", "extraction"]


def test_tracker_stop_unstarted_stage_is_safe():
    tracker = AuditTrailTracker()
    # Should not raise
    tracker.stop_stage("nonexistent_stage")
    assert len(tracker.stages) == 0


def test_tracker_context_manager():
    tracker = AuditTrailTracker()
    with tracker.stage("validation"):
        time.sleep(0.01)
    assert len(tracker.stages) == 1
    assert tracker.stages[0].stage == "validation"
    assert tracker.stages[0].duration_ms >= 0


def test_tracker_context_manager_on_exception():
    tracker = AuditTrailTracker()
    with pytest.raises(ValueError):
        with tracker.stage("failing_stage"):
            raise ValueError("test error")
    # Stage should still be recorded even on exception
    assert len(tracker.stages) == 1
    assert tracker.stages[0].stage == "failing_stage"


def test_tracker_build_audit_trail():
    tracker = AuditTrailTracker()
    tracker.record_model("paddleocr_3.0")
    tracker.record_model("claude-opus-4-20250514")
    tracker.record_llm_call()
    tracker.record_llm_call()
    tracker.record_reextraction()
    tracker.record_flagged_field()
    tracker.start_stage("ingestion")
    tracker.stop_stage("ingestion")
    tracker.start_stage("extraction")
    tracker.stop_stage("extraction")

    audit = tracker.build()

    assert isinstance(audit, AuditTrail)
    assert len(audit.models_used) == 2
    assert audit.total_llm_calls == 2
    assert audit.re_extractions == 1
    assert audit.fields_flagged == 1
    assert len(audit.processing_stages) == 2
    assert audit.processing_stages[0].stage == "ingestion"


def test_tracker_build_sorts_models():
    tracker = AuditTrailTracker()
    tracker.record_model("gpt-4o")
    tracker.record_model("claude-opus-4-20250514")
    tracker.record_model("docling")

    audit = tracker.build()
    assert audit.models_used == ["claude-opus-4-20250514", "docling", "gpt-4o"]


def test_tracker_merge():
    """Merging two trackers should combine all their data."""
    t1 = AuditTrailTracker()
    t1.record_model("paddleocr_3.0")
    t1.record_llm_call()
    t1.start_stage("ingestion")
    t1.stop_stage("ingestion")

    t2 = AuditTrailTracker()
    t2.record_model("claude-opus-4-20250514")
    t2.record_llm_call()
    t2.record_llm_call()
    t2.record_reextraction()
    t2.start_stage("extraction")
    t2.stop_stage("extraction")

    t1.merge(t2)

    assert len(t1.models_used) == 2
    assert t1.total_llm_calls == 3
    assert t1.re_extractions == 1
    assert len(t1.stages) == 2


def test_tracker_reset():
    tracker = AuditTrailTracker()
    tracker.record_model("test")
    tracker.record_llm_call()
    tracker.record_reextraction()
    tracker.record_flagged_field()
    tracker.start_stage("s1")
    tracker.stop_stage("s1")

    tracker.reset()

    assert len(tracker.models_used) == 0
    assert tracker.total_llm_calls == 0
    assert tracker.re_extractions == 0
    assert tracker.fields_flagged == 0
    assert len(tracker.stages) == 0
