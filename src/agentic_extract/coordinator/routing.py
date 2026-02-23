"""Deterministic routing of regions to extraction specialists.

Uses rule-based logic to assign each region to the appropriate
specialist (Text, Table, or Visual). For ambiguous regions with
confidence < 0.5, a Claude classification call would be made
in the full system; here we route based on the detected type.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from agentic_extract.coordinator.layout import LayoutRegion
from agentic_extract.coordinator.quality import QualityAssessment
from agentic_extract.models import RegionType


class SpecialistType(str, Enum):
    """Available extraction specialists."""

    TEXT = "text_specialist"
    TABLE = "table_specialist"
    VISUAL = "visual_specialist"


# Deterministic routing rules: RegionType -> SpecialistType
ROUTING_RULES: dict[RegionType, SpecialistType] = {
    RegionType.TEXT: SpecialistType.TEXT,
    RegionType.TABLE: SpecialistType.TABLE,
    RegionType.FIGURE: SpecialistType.VISUAL,
    RegionType.HANDWRITING: SpecialistType.VISUAL,
    RegionType.FORMULA: SpecialistType.VISUAL,
    RegionType.FORM_FIELD: SpecialistType.TEXT,
}

# Default model assignments per specialist
MODEL_ASSIGNMENTS: dict[SpecialistType, str] = {
    SpecialistType.TEXT: "claude",
    SpecialistType.TABLE: "claude+codex",
    SpecialistType.VISUAL: "codex+claude",
}

# Confidence threshold below which a region is considered ambiguous
AMBIGUITY_THRESHOLD = 0.5


@dataclass
class RoutingEntry:
    """Routing decision for a single region."""

    region_id: str
    specialist: SpecialistType
    model_assignment: str
    priority: int


@dataclass
class RoutingPlan:
    """Complete routing plan for all detected regions."""

    entries: list[RoutingEntry] = field(default_factory=list)


def generate_routing_plan(
    regions: list[LayoutRegion],
    quality: QualityAssessment,
) -> RoutingPlan:
    """Generate a deterministic routing plan for detected regions.

    Args:
        regions: Layout regions detected by DocLayout-YOLO.
        quality: Quality assessment for priority calculation.

    Returns:
        RoutingPlan with one entry per region.
    """
    # Base priority: lower = higher priority
    # Degraded documents get priority boost (lower number)
    base_priority = 1 if quality.needs_enhancement else 5

    entries: list[RoutingEntry] = []
    for region in regions:
        specialist = ROUTING_RULES.get(region.region_type, SpecialistType.TEXT)
        model = MODEL_ASSIGNMENTS.get(specialist, "claude")

        # Ambiguous regions get flagged (in full system: Claude classification call)
        priority = base_priority
        if region.confidence < AMBIGUITY_THRESHOLD:
            # Ambiguous regions get slightly higher priority
            priority = max(1, base_priority - 1)

        entries.append(
            RoutingEntry(
                region_id=region.region_id,
                specialist=specialist,
                model_assignment=model,
                priority=priority,
            )
        )

    return RoutingPlan(entries=entries)


def route_regions(
    regions: list[LayoutRegion],
) -> dict[str, list[LayoutRegion]]:
    """Group layout regions by specialist type for pipeline dispatch.

    Convenience function used by the Pipeline orchestrator. Buckets each
    region into "text", "table", or "visual" based on deterministic rules.

    Args:
        regions: Layout regions detected by DocLayout-YOLO.

    Returns:
        Dict with keys "text", "table", "visual", each mapping to a list
        of LayoutRegion objects assigned to that specialist.
    """
    buckets: dict[str, list[LayoutRegion]] = {
        "text": [],
        "table": [],
        "visual": [],
    }
    specialist_to_bucket = {
        SpecialistType.TEXT: "text",
        SpecialistType.TABLE: "table",
        SpecialistType.VISUAL: "visual",
    }
    for region in regions:
        specialist = ROUTING_RULES.get(region.region_type, SpecialistType.TEXT)
        bucket = specialist_to_bucket.get(specialist, "text")
        buckets[bucket].append(region)
    return buckets
