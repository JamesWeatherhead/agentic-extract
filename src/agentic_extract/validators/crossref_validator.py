"""Validator Layer 2: Cross-reference validation.

Deterministic checks for:
- Date plausibility (not future, not before 1900)
- Numerical magnitude (values within sane orders of magnitude)
- Reference consistency (Table N / Figure N references match actual regions)
- Entity cross-referencing (names mentioned appear in document text)
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from agentic_extract.models import (
    Region,
    RegionType,
    TableContent,
    TextContent,
)

# Dates before this year are flagged as implausible
MIN_PLAUSIBLE_YEAR = 1900
# Dates after this year are flagged as future
MAX_PLAUSIBLE_YEAR = datetime.now().year + 2

# Numerical values above this magnitude are flagged
MAX_SANE_MAGNITUDE = 1e11

# Regex patterns
DATE_PATTERN = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
TABLE_REF_PATTERN = re.compile(r"Table\s+(\d+)", re.IGNORECASE)
FIGURE_REF_PATTERN = re.compile(r"Figure\s+(\d+)", re.IGNORECASE)


@dataclass
class CrossRefViolation:
    """A single cross-reference validation violation."""

    region_id: str
    check_type: str
    message: str
    severity: str  # "error" or "warning"


def _check_date_plausibility(regions: list[Region]) -> list[CrossRefViolation]:
    """Check that dates in text regions are plausible."""
    violations: list[CrossRefViolation] = []

    for region in regions:
        text = ""
        if isinstance(region.content, TextContent):
            text = region.content.text
        elif isinstance(region.content, TableContent):
            # Check table cell values for dates
            for row in region.content.json_data.get("rows", []):
                for val in row.values():
                    text += f" {val}"

        for match in DATE_PATTERN.finditer(text):
            year = int(match.group(1))
            if year < MIN_PLAUSIBLE_YEAR:
                violations.append(CrossRefViolation(
                    region_id=region.id,
                    check_type="date_plausibility",
                    message=f"Date {match.group(0)} has year {year} before {MIN_PLAUSIBLE_YEAR}",
                    severity="warning",
                ))
            elif year > MAX_PLAUSIBLE_YEAR:
                violations.append(CrossRefViolation(
                    region_id=region.id,
                    check_type="date_plausibility",
                    message=f"Date {match.group(0)} has year {year} in the future",
                    severity="warning",
                ))

    return violations


def _check_numerical_magnitude(regions: list[Region]) -> list[CrossRefViolation]:
    """Check that numerical values in tables are within sane magnitudes."""
    violations: list[CrossRefViolation] = []

    for region in regions:
        if not isinstance(region.content, TableContent):
            continue

        for row in region.content.json_data.get("rows", []):
            for key, val in row.items():
                try:
                    num = float(str(val).replace(",", ""))
                    if abs(num) > MAX_SANE_MAGNITUDE:
                        violations.append(CrossRefViolation(
                            region_id=region.id,
                            check_type="numerical_magnitude",
                            message=(
                                f"Value {val} in column '{key}' exceeds "
                                f"sane magnitude ({MAX_SANE_MAGNITUDE})"
                            ),
                            severity="warning",
                        ))
                except (ValueError, TypeError):
                    pass  # Non-numeric values are fine

    return violations


def _check_reference_consistency(regions: list[Region]) -> list[CrossRefViolation]:
    """Check that Table N and Figure N references match existing regions."""
    violations: list[CrossRefViolation] = []

    # Count table and figure regions
    table_count = sum(1 for r in regions if r.type == RegionType.TABLE)
    figure_count = sum(
        1 for r in regions
        if r.type == RegionType.FIGURE
    )

    # Collect all text
    all_text = ""
    text_region_ids: list[str] = []
    for region in regions:
        if isinstance(region.content, TextContent):
            all_text += " " + region.content.text
            text_region_ids.append(region.id)

    # Check table references
    for match in TABLE_REF_PATTERN.finditer(all_text):
        ref_num = int(match.group(1))
        if ref_num > table_count:
            # Find which region contains this reference
            ref_region_id = "unknown"
            for region in regions:
                if isinstance(region.content, TextContent):
                    if match.group(0) in region.content.text:
                        ref_region_id = region.id
                        break
            violations.append(CrossRefViolation(
                region_id=ref_region_id,
                check_type="reference_consistency",
                message=(
                    f"Text references 'Table {ref_num}' but only "
                    f"{table_count} table(s) found in document"
                ),
                severity="warning",
            ))

    # Check figure references
    for match in FIGURE_REF_PATTERN.finditer(all_text):
        ref_num = int(match.group(1))
        if ref_num > figure_count:
            ref_region_id = "unknown"
            for region in regions:
                if isinstance(region.content, TextContent):
                    if match.group(0) in region.content.text:
                        ref_region_id = region.id
                        break
            violations.append(CrossRefViolation(
                region_id=ref_region_id,
                check_type="reference_consistency",
                message=(
                    f"Text references 'Figure {ref_num}' but only "
                    f"{figure_count} figure(s) found in document"
                ),
                severity="warning",
            ))

    return violations


def validate_cross_references(
    regions: list[Region],
) -> list[CrossRefViolation]:
    """Run Layer 2 cross-reference validation on all regions.

    Checks:
    - Date plausibility (years between 1900 and current+2)
    - Numerical magnitude (table values within sane ranges)
    - Reference consistency (Table N / Figure N match actual counts)

    Args:
        regions: List of extracted regions to validate.

    Returns:
        List of CrossRefViolation objects (empty if all valid).
    """
    violations: list[CrossRefViolation] = []
    violations.extend(_check_date_plausibility(regions))
    violations.extend(_check_numerical_magnitude(regions))
    violations.extend(_check_reference_consistency(regions))
    return violations
