# tests/validators/test_crossref_validator.py
"""Tests for Validator Layer 2: cross-reference validation."""
import pytest

from agentic_extract.models import (
    BoundingBox,
    Region,
    RegionType,
    TableContent,
    TextContent,
)
from agentic_extract.validators.crossref_validator import (
    CrossRefViolation,
    validate_cross_references,
)


def _make_text_region(rid: str, text: str, page: int = 1) -> Region:
    return Region(
        id=rid,
        type=RegionType.TEXT,
        subtype=None,
        page=page,
        bbox=BoundingBox(x=0.0, y=0.0, w=1.0, h=0.5),
        content=TextContent(text=text, markdown=text),
        confidence=0.95,
        extraction_method="paddleocr_3.0",
    )


def _make_table_region(
    rid: str, headers: list[str], rows: list[dict], page: int = 1,
) -> Region:
    html = "<table><tr>" + "".join(f"<th>{h}</th>" for h in headers) + "</tr>"
    for row in rows:
        html += "<tr>" + "".join(f"<td>{row.get(h, '')}</td>" for h in headers) + "</tr>"
    html += "</table>"
    return Region(
        id=rid,
        type=RegionType.TABLE,
        subtype=None,
        page=page,
        bbox=BoundingBox(x=0.0, y=0.5, w=1.0, h=0.5),
        content=TableContent(
            html=html,
            json_data={"headers": headers, "rows": rows},
        ),
        confidence=0.92,
        extraction_method="docling",
    )


def test_crossref_violation_model():
    cv = CrossRefViolation(
        region_id="r1",
        check_type="date_plausibility",
        message="Date 2099-01-01 is in the future",
        severity="warning",
    )
    assert cv.check_type == "date_plausibility"
    assert cv.severity == "warning"


def test_valid_regions_pass():
    regions = [
        _make_text_region("r1", "As shown in Table 1, gene expression increased."),
        _make_table_region("t1", ["Gene", "Value"], [{"Gene": "BRCA1", "Value": "3.2"}]),
    ]
    violations = validate_cross_references(regions)
    assert violations == []


def test_future_date_flagged():
    regions = [
        _make_text_region("r1", "The experiment was conducted on 2099-12-31."),
    ]
    violations = validate_cross_references(regions)
    assert any(v.check_type == "date_plausibility" for v in violations)


def test_ancient_date_flagged():
    regions = [
        _make_text_region("r1", "Records from 1850-01-01 indicate treatment."),
    ]
    violations = validate_cross_references(regions)
    assert any(v.check_type == "date_plausibility" for v in violations)


def test_reasonable_date_passes():
    regions = [
        _make_text_region("r1", "Published on 2024-06-15 in Nature."),
    ]
    violations = validate_cross_references(regions)
    date_violations = [v for v in violations if v.check_type == "date_plausibility"]
    assert date_violations == []


def test_extreme_numerical_magnitude_flagged():
    regions = [
        _make_table_region(
            "t1",
            ["Gene", "P-value"],
            [{"Gene": "BRCA1", "P-value": "999999999999"}],
        ),
    ]
    violations = validate_cross_references(regions)
    assert any(v.check_type == "numerical_magnitude" for v in violations)


def test_reasonable_numerical_passes():
    regions = [
        _make_table_region(
            "t1",
            ["Gene", "P-value"],
            [{"Gene": "BRCA1", "P-value": "0.001"}],
        ),
    ]
    violations = validate_cross_references(regions)
    magnitude_violations = [v for v in violations if v.check_type == "numerical_magnitude"]
    assert magnitude_violations == []


def test_referenced_table_missing_flagged():
    """Text references 'Table 3' but no table region with that reference exists."""
    regions = [
        _make_text_region("r1", "As shown in Table 3, the results confirm..."),
        _make_table_region("t1", ["A"], [{"A": "1"}]),
    ]
    violations = validate_cross_references(regions)
    assert any(v.check_type == "reference_consistency" for v in violations)


def test_referenced_figure_missing_flagged():
    regions = [
        _make_text_region("r1", "See Figure 5 for the detailed results."),
    ]
    violations = validate_cross_references(regions)
    assert any(v.check_type == "reference_consistency" for v in violations)


def test_table_reference_present_passes():
    """When text references Table 1 and a table region exists, no violation."""
    regions = [
        _make_text_region("r1", "Results in Table 1 show improvement."),
        _make_table_region("table_1", ["Metric", "Value"], [{"Metric": "Acc", "Value": "0.95"}]),
    ]
    # The check looks for table/figure count vs references.
    # With 1 table region and reference to "Table 1", this should pass.
    violations = validate_cross_references(regions)
    ref_violations = [v for v in violations if v.check_type == "reference_consistency"]
    assert ref_violations == []
