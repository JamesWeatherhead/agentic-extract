# tests/coordinator/test_routing.py
"""Tests for deterministic region-to-specialist routing."""
import pathlib
from unittest.mock import AsyncMock, patch

import pytest

from agentic_extract.coordinator.layout import LayoutRegion
from agentic_extract.coordinator.quality import QualityAssessment
from agentic_extract.coordinator.routing import (
    RoutingEntry,
    RoutingPlan,
    SpecialistType,
    generate_routing_plan,
)
from agentic_extract.models import BoundingBox, RegionType


def _make_region(
    rid: str, rtype: RegionType, conf: float = 0.95, page: int = 1,
) -> LayoutRegion:
    return LayoutRegion(
        region_id=rid, region_type=rtype,
        bbox=BoundingBox(x=0.1, y=0.1, w=0.8, h=0.2),
        confidence=conf, page=page,
    )


def test_specialist_type_enum():
    assert SpecialistType.TEXT == "text_specialist"
    assert SpecialistType.TABLE == "table_specialist"
    assert SpecialistType.VISUAL == "visual_specialist"


def test_routing_entry_dataclass():
    entry = RoutingEntry(
        region_id="r1",
        specialist=SpecialistType.TEXT,
        model_assignment="claude",
        priority=1,
    )
    assert entry.specialist == SpecialistType.TEXT


def test_routing_plan_dataclass():
    plan = RoutingPlan(entries=[])
    assert plan.entries == []


def test_text_region_routes_to_text_specialist():
    regions = [_make_region("r1", RegionType.TEXT)]
    quality = QualityAssessment(dpi=300, skew_angle=0.0, degradation_score=0.1, needs_enhancement=False)
    plan = generate_routing_plan(regions, quality)
    assert len(plan.entries) == 1
    assert plan.entries[0].specialist == SpecialistType.TEXT
    assert plan.entries[0].region_id == "r1"


def test_table_region_routes_to_table_specialist():
    regions = [_make_region("r1", RegionType.TABLE)]
    quality = QualityAssessment(dpi=300, skew_angle=0.0, degradation_score=0.1, needs_enhancement=False)
    plan = generate_routing_plan(regions, quality)
    assert plan.entries[0].specialist == SpecialistType.TABLE


def test_figure_routes_to_visual():
    regions = [_make_region("r1", RegionType.FIGURE)]
    quality = QualityAssessment(dpi=300, skew_angle=0.0, degradation_score=0.1, needs_enhancement=False)
    plan = generate_routing_plan(regions, quality)
    assert plan.entries[0].specialist == SpecialistType.VISUAL


def test_handwriting_routes_to_visual():
    regions = [_make_region("r1", RegionType.HANDWRITING)]
    quality = QualityAssessment(dpi=300, skew_angle=0.0, degradation_score=0.1, needs_enhancement=False)
    plan = generate_routing_plan(regions, quality)
    assert plan.entries[0].specialist == SpecialistType.VISUAL


def test_formula_routes_to_visual():
    regions = [_make_region("r1", RegionType.FORMULA)]
    quality = QualityAssessment(dpi=300, skew_angle=0.0, degradation_score=0.1, needs_enhancement=False)
    plan = generate_routing_plan(regions, quality)
    assert plan.entries[0].specialist == SpecialistType.VISUAL


def test_form_field_routes_to_text():
    regions = [_make_region("r1", RegionType.FORM_FIELD)]
    quality = QualityAssessment(dpi=300, skew_angle=0.0, degradation_score=0.1, needs_enhancement=False)
    plan = generate_routing_plan(regions, quality)
    assert plan.entries[0].specialist == SpecialistType.TEXT


def test_mixed_regions_route_correctly():
    regions = [
        _make_region("r_text", RegionType.TEXT),
        _make_region("r_table", RegionType.TABLE),
        _make_region("r_fig", RegionType.FIGURE),
        _make_region("r_hw", RegionType.HANDWRITING),
    ]
    quality = QualityAssessment(dpi=300, skew_angle=0.0, degradation_score=0.1, needs_enhancement=False)
    plan = generate_routing_plan(regions, quality)
    assert len(plan.entries) == 4

    routing_map = {e.region_id: e.specialist for e in plan.entries}
    assert routing_map["r_text"] == SpecialistType.TEXT
    assert routing_map["r_table"] == SpecialistType.TABLE
    assert routing_map["r_fig"] == SpecialistType.VISUAL
    assert routing_map["r_hw"] == SpecialistType.VISUAL


def test_low_confidence_region_still_routes():
    """Low-confidence regions (< 0.5) should still get a routing entry.

    In the full system, these would trigger a Claude classification
    call, but routing should never drop a region.
    """
    regions = [_make_region("r_ambiguous", RegionType.TEXT, conf=0.3)]
    quality = QualityAssessment(dpi=300, skew_angle=0.0, degradation_score=0.1, needs_enhancement=False)
    plan = generate_routing_plan(regions, quality)
    assert len(plan.entries) == 1
    assert plan.entries[0].region_id == "r_ambiguous"


def test_degraded_quality_sets_higher_priority():
    """Degraded documents should get higher priority (lower number)."""
    regions = [_make_region("r1", RegionType.TEXT)]
    good_quality = QualityAssessment(dpi=300, skew_angle=0.0, degradation_score=0.1, needs_enhancement=False)
    bad_quality = QualityAssessment(dpi=72, skew_angle=5.0, degradation_score=0.7, needs_enhancement=True)

    plan_good = generate_routing_plan(regions, good_quality)
    plan_bad = generate_routing_plan(regions, bad_quality)

    # Higher priority (lower number) for degraded docs
    assert plan_bad.entries[0].priority <= plan_good.entries[0].priority
