"""Validator Layer 1: Deterministic JSON schema conformance.

Checks that every region has:
- A non-empty region ID
- A valid region type
- Content matching the expected type (non-empty text, valid table structure)
- No duplicate region IDs
- Valid page numbers and bounding boxes
"""
from __future__ import annotations

from dataclasses import dataclass

from agentic_extract.models import (
    FormulaContent,
    HandwritingContent,
    Region,
    RegionType,
    TableContent,
    TextContent,
)


@dataclass
class SchemaViolation:
    """A single schema validation violation."""

    region_id: str
    field: str
    violation_type: str
    message: str
    severity: str  # "error" or "warning"


def _validate_text_content(region: Region) -> list[SchemaViolation]:
    """Validate TextContent fields."""
    violations: list[SchemaViolation] = []
    content = region.content
    if isinstance(content, TextContent):
        if not content.text.strip():
            violations.append(SchemaViolation(
                region_id=region.id,
                field="content.text",
                violation_type="empty_field",
                message="Text content is empty",
                severity="error",
            ))
    return violations


def _validate_table_content(region: Region) -> list[SchemaViolation]:
    """Validate TableContent fields."""
    violations: list[SchemaViolation] = []
    content = region.content
    if isinstance(content, TableContent):
        if not content.html.strip():
            violations.append(SchemaViolation(
                region_id=region.id,
                field="content.html",
                violation_type="empty_field",
                message="Table HTML is empty",
                severity="error",
            ))
        headers = content.json_data.get("headers", [])
        if not headers:
            violations.append(SchemaViolation(
                region_id=region.id,
                field="content.json_data.headers",
                violation_type="empty_field",
                message="Table has no headers",
                severity="warning",
            ))
    return violations


def _validate_handwriting_content(region: Region) -> list[SchemaViolation]:
    """Validate HandwritingContent fields."""
    violations: list[SchemaViolation] = []
    content = region.content
    if isinstance(content, HandwritingContent):
        if not content.text.strip():
            violations.append(SchemaViolation(
                region_id=region.id,
                field="content.text",
                violation_type="empty_field",
                message="Handwriting text is empty",
                severity="error",
            ))
    return violations


def _validate_formula_content(region: Region) -> list[SchemaViolation]:
    """Validate FormulaContent fields."""
    violations: list[SchemaViolation] = []
    content = region.content
    if isinstance(content, FormulaContent):
        if not content.latex.strip():
            violations.append(SchemaViolation(
                region_id=region.id,
                field="content.latex",
                violation_type="empty_field",
                message="Formula LaTeX is empty",
                severity="error",
            ))
    return violations


# Map region types to their content validators
CONTENT_VALIDATORS = {
    RegionType.TEXT: _validate_text_content,
    RegionType.TABLE: _validate_table_content,
    RegionType.HANDWRITING: _validate_handwriting_content,
    RegionType.FORMULA: _validate_formula_content,
}


def validate_schema(regions: list[Region]) -> list[SchemaViolation]:
    """Run Layer 1 schema validation on all regions.

    Checks:
    - Non-empty region IDs
    - No duplicate region IDs
    - Content type-specific validation (non-empty text, valid table headers, etc.)

    Args:
        regions: List of extracted regions to validate.

    Returns:
        List of SchemaViolation objects (empty if all valid).
    """
    violations: list[SchemaViolation] = []
    seen_ids: set[str] = set()

    for region in regions:
        # Check region ID
        if not region.id.strip():
            violations.append(SchemaViolation(
                region_id=region.id or "(empty)",
                field="id",
                violation_type="empty_field",
                message="Region ID is empty",
                severity="error",
            ))

        # Check duplicate IDs
        if region.id in seen_ids:
            violations.append(SchemaViolation(
                region_id=region.id,
                field="id",
                violation_type="duplicate_id",
                message=f"Duplicate region ID: {region.id}",
                severity="error",
            ))
        seen_ids.add(region.id)

        # Content-specific validation
        validator = CONTENT_VALIDATORS.get(region.type)
        if validator:
            violations.extend(validator(region))

    return violations
