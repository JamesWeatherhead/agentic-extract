# src/agentic_extract/models.py
"""Core Pydantic v2 data models for Agentic Extract.

Matches the JSON output schema from design doc Section 6.
All bounding box coordinates are normalized to [0, 1].
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


class BoundingBox(BaseModel):
    """Normalized bounding box with coordinates in [0, 1]."""

    x: float = Field(..., ge=0.0, le=1.0, description="Left edge, normalized")
    y: float = Field(..., ge=0.0, le=1.0, description="Top edge, normalized")
    w: float = Field(..., ge=0.0, le=1.0, description="Width, normalized")
    h: float = Field(..., ge=0.0, le=1.0, description="Height, normalized")


class RegionType(str, Enum):
    """Types of document regions detected by layout analysis."""

    TEXT = "text"
    TABLE = "table"
    FIGURE = "figure"
    HANDWRITING = "handwriting"
    FORMULA = "formula"
    FORM_FIELD = "form_field"


# --- Region Content Types ---


class TextContent(BaseModel):
    """Content for text regions."""

    text: str
    markdown: str


class TableContent(BaseModel):
    """Content for table regions."""

    html: str
    json_data: dict[str, Any] = Field(default_factory=dict)
    cell_bboxes: list[dict[str, Any]] = Field(default_factory=list)


class FigureContent(BaseModel):
    """Content for figure/chart regions."""

    description: str
    figure_type: str | None = None
    figure_json: dict[str, Any] = Field(default_factory=dict)


class HandwritingContent(BaseModel):
    """Content for handwriting regions."""

    text: str
    latex: str | None = None


class FormulaContent(BaseModel):
    """Content for formula/equation regions."""

    latex: str
    mathml: str | None = None


# Union type for region content
RegionContent = TextContent | TableContent | FigureContent | HandwritingContent | FormulaContent


class Region(BaseModel):
    """A detected and extracted document region."""

    id: str
    type: RegionType
    subtype: str | None = None
    page: int = Field(..., ge=1)
    bbox: BoundingBox
    content: RegionContent
    confidence: float = Field(..., ge=0.0, le=1.0)
    extraction_method: str
    model_agreement: str | None = None
    needs_review: bool = False
    review_reason: str | None = None


class DocumentMetadata(BaseModel):
    """Metadata about the processed document."""

    id: str
    source: str
    page_count: int = Field(..., ge=0)
    processing_timestamp: datetime
    approach: str
    total_confidence: float = Field(..., ge=0.0, le=1.0)
    processing_time_ms: int = Field(..., ge=0)


class ProcessingStage(BaseModel):
    """Timing information for a single processing stage."""

    stage: str
    duration_ms: int = Field(..., ge=0)


class AuditTrail(BaseModel):
    """Complete processing audit trail."""

    models_used: list[str]
    total_llm_calls: int = Field(..., ge=0)
    re_extractions: int = Field(..., ge=0)
    fields_flagged: int = Field(..., ge=0)
    processing_stages: list[ProcessingStage]


class ExtractionResult(BaseModel):
    """Top-level extraction result containing all outputs."""

    document: DocumentMetadata
    markdown: str
    regions: list[Region]
    extracted_entities: dict[str, Any] = Field(default_factory=dict)
    audit_trail: AuditTrail
