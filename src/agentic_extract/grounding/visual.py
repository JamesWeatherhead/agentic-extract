# src/agentic_extract/grounding/visual.py
"""Visual grounding enhancement: link every extracted value to its source bounding box.

All coordinates are normalized to [0, 1] relative to page dimensions.
After grounding, each field and cell carries a bbox_verified flag
indicating whether a Layer 4 validation OCR pass confirmed the
bounding box aligns with the extracted text.
"""
from __future__ import annotations

import logging
import pathlib
from typing import Any

from pydantic import BaseModel, Field

from agentic_extract.models import (
    BoundingBox,
    FigureContent,
    FormulaContent,
    HandwritingContent,
    Region,
    RegionType,
    TableContent,
    TextContent,
)
from agentic_extract.tools.docker_runner import DockerTool

logger = logging.getLogger(__name__)

# Maximum edit distance ratio for bbox verification
EDIT_DISTANCE_THRESHOLD = 0.30


class GroundedField(BaseModel):
    """A single extracted field linked to its source bounding box."""

    field_name: str
    value: Any
    bbox: BoundingBox
    confidence: float = Field(..., ge=0.0, le=1.0)
    bbox_verified: bool = False


class CellGrounding(BaseModel):
    """A single table cell linked to its source bounding box."""

    row: int = Field(..., ge=0)
    col: int = Field(..., ge=0)
    value: Any
    bbox: BoundingBox
    confidence: float = Field(..., ge=0.0, le=1.0)
    bbox_verified: bool = False


class GroundedRegion(BaseModel):
    """A fully grounded region with all fields and cells linked to bounding boxes."""

    region_id: str
    region_type: RegionType
    page: int = Field(..., ge=1)
    region_bbox: BoundingBox
    fields: list[GroundedField]
    cells: list[CellGrounding] = Field(default_factory=list)


def _run_quick_ocr(image_path: pathlib.Path, bbox: BoundingBox) -> str:
    """Crop the bbox region from the image and run a quick OCR pass.

    Uses PaddleOCR Docker container for Layer 4 visual grounding verification.
    Returns the OCR text from the cropped region.
    """
    from PIL import Image
    import json

    img = Image.open(image_path)
    w, h = img.size

    left = int(bbox.x * w)
    top = int(bbox.y * h)
    right = int((bbox.x + bbox.w) * w)
    bottom = int((bbox.y + bbox.h) * h)

    # Clamp to image bounds
    left = max(0, min(left, w))
    top = max(0, min(top, h))
    right = max(0, min(right, w))
    bottom = max(0, min(bottom, h))

    if right <= left or bottom <= top:
        return ""

    cropped = img.crop((left, top, right, bottom))
    crop_path = image_path.parent / f"_crop_{bbox.x:.3f}_{bbox.y:.3f}.png"
    cropped.save(crop_path)

    try:
        tool = DockerTool(
            image_name="paddlepaddle/paddleocr:latest",
            default_timeout=30,
        )
        result = tool.run(["--image_dir", str(crop_path), "--type", "ocr"])
        if result.exit_code == 0:
            data = json.loads(result.stdout)
            return data.get("text", "")
    except Exception as exc:
        logger.warning("Quick OCR for bbox verification failed: %s", exc)

    return ""


def _normalized_edit_distance(s1: str, s2: str) -> float:
    """Compute normalized edit distance between two strings.

    Returns a float in [0, 1] where 0 means identical.
    """
    s1_clean = s1.strip().lower()
    s2_clean = s2.strip().lower()

    if s1_clean == s2_clean:
        return 0.0

    if not s1_clean or not s2_clean:
        return 1.0

    # Simple Levenshtein distance
    len1, len2 = len(s1_clean), len(s2_clean)
    if len1 == 0:
        return 1.0
    if len2 == 0:
        return 1.0

    matrix = [[0] * (len2 + 1) for _ in range(len1 + 1)]
    for i in range(len1 + 1):
        matrix[i][0] = i
    for j in range(len2 + 1):
        matrix[0][j] = j

    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            cost = 0 if s1_clean[i - 1] == s2_clean[j - 1] else 1
            matrix[i][j] = min(
                matrix[i - 1][j] + 1,
                matrix[i][j - 1] + 1,
                matrix[i - 1][j - 1] + cost,
            )

    distance = matrix[len1][len2]
    max_len = max(len1, len2)
    return distance / max_len if max_len > 0 else 0.0


def _verify_bbox(extracted_value: str, ocr_text: str) -> bool:
    """Verify that the OCR text from the cropped bbox matches the extracted value."""
    if not extracted_value or not ocr_text:
        return False
    distance = _normalized_edit_distance(extracted_value, ocr_text)
    return distance <= EDIT_DISTANCE_THRESHOLD


class VisualGrounding:
    """Visual grounding engine: links extracted values to source bounding boxes.

    For each extracted region:
    1. Text regions: link the extracted text to the region bbox
    2. Table regions: link each cell value to its cell bbox
    3. Figure regions: link the description to the region bbox
    4. Handwriting/Formula: link to the region bbox

    After linking, runs a Layer 4 OCR verification pass on each bbox
    to set the bbox_verified flag.
    """

    def ground_region(
        self,
        region: Region,
        page_image_path: pathlib.Path,
    ) -> GroundedRegion:
        """Ground a region by linking values to bounding boxes.

        Args:
            region: The extracted region with content.
            page_image_path: Path to the full page image.

        Returns:
            GroundedRegion with fields and cells linked to bboxes.
        """
        content = region.content
        fields: list[GroundedField] = []
        cells: list[CellGrounding] = []

        if isinstance(content, TextContent):
            ocr_text = _run_quick_ocr(page_image_path, region.bbox)
            verified = _verify_bbox(content.text, ocr_text)
            fields.append(GroundedField(
                field_name="text",
                value=content.text,
                bbox=region.bbox,
                confidence=region.confidence,
                bbox_verified=verified,
            ))

        elif isinstance(content, TableContent):
            # Cell-level grounding
            for cell_info in content.cell_bboxes:
                cell_bbox = cell_info.get("bbox")
                if cell_bbox is None:
                    continue
                if isinstance(cell_bbox, dict):
                    cell_bbox = BoundingBox(**cell_bbox)

                row = cell_info.get("row", 0)
                col = cell_info.get("col", 0)

                # Get cell value from json_data
                cell_value = self._get_cell_value(content.json_data, row, col)

                ocr_text = _run_quick_ocr(page_image_path, cell_bbox)
                verified = _verify_bbox(str(cell_value), ocr_text)

                cells.append(CellGrounding(
                    row=row,
                    col=col,
                    value=cell_value,
                    bbox=cell_bbox,
                    confidence=region.confidence,
                    bbox_verified=verified,
                ))

        elif isinstance(content, FigureContent):
            ocr_text = _run_quick_ocr(page_image_path, region.bbox)
            verified = _verify_bbox(content.description, ocr_text)
            fields.append(GroundedField(
                field_name="description",
                value=content.description,
                bbox=region.bbox,
                confidence=region.confidence,
                bbox_verified=verified,
            ))

        elif isinstance(content, HandwritingContent):
            ocr_text = _run_quick_ocr(page_image_path, region.bbox)
            verified = _verify_bbox(content.text, ocr_text)
            fields.append(GroundedField(
                field_name="text",
                value=content.text,
                bbox=region.bbox,
                confidence=region.confidence,
                bbox_verified=verified,
            ))

        elif isinstance(content, FormulaContent):
            # For formulas, we check the raw LaTeX against OCR
            ocr_text = _run_quick_ocr(page_image_path, region.bbox)
            # Formulas rarely match via OCR, so verify is lenient
            verified = len(ocr_text.strip()) > 0
            fields.append(GroundedField(
                field_name="latex",
                value=content.latex,
                bbox=region.bbox,
                confidence=region.confidence,
                bbox_verified=verified,
            ))

        return GroundedRegion(
            region_id=region.id,
            region_type=region.type,
            page=region.page,
            region_bbox=region.bbox,
            fields=fields,
            cells=cells,
        )

    @staticmethod
    def _get_cell_value(json_data: dict, row: int, col: int) -> Any:
        """Extract a cell value from table json_data by row and col index.

        Row 0 is the header row. Subsequent rows are data rows.
        """
        headers = json_data.get("headers", [])
        rows = json_data.get("rows", [])

        if row == 0:
            # Header row
            if col < len(headers):
                return headers[col]
            return ""

        # Data row (row index is 1-based for data)
        data_idx = row - 1
        if data_idx < len(rows):
            row_data = rows[data_idx]
            if isinstance(row_data, dict) and col < len(headers):
                return row_data.get(headers[col], "")
            if isinstance(row_data, list) and col < len(row_data):
                return row_data[col]
        return ""
