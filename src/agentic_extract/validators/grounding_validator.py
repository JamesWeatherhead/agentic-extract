"""Validator Layers 4-5: Visual grounding check + confidence calibration.

Layer 4 (Visual Grounding):
  For each extracted value, crop the bounding box region from the original
  page image, run a quick OCR pass (PaddleOCR) on the crop, and compare
  with the extracted text. Significant divergence (high edit distance)
  indicates either the extraction or the bounding box is wrong.

Layer 5 (Confidence Calibration):
  Aggregate per-character, per-field, and per-region confidence scores.
  Apply temperature scaling calibration. Compute the final calibrated
  confidence per field. Apply decision thresholds:
    - confidence >= 0.90 -> ACCEPT
    - 0.70 <= confidence < 0.90 -> RE_EXTRACT
    - confidence < 0.70 -> FLAG
"""
from __future__ import annotations

import logging
import math
import pathlib
from dataclasses import dataclass
from enum import Enum

from PIL import Image

from agentic_extract.models import (
    Region,
    RegionType,
    TextContent,
    HandwritingContent,
)
from agentic_extract.specialists.text import PaddleOCRTool

logger = logging.getLogger(__name__)

# Decision thresholds (from design doc Section 7)
ACCEPT_THRESHOLD = 0.90
RE_EXTRACT_THRESHOLD = 0.70

# Confidence weight formula (from design doc Section 7)
OCR_WEIGHT = 0.3
VLM_WEIGHT = 0.4
VALIDATION_WEIGHT = 0.3

# Edit distance threshold for grounding check
GROUNDING_EDIT_DISTANCE_THRESHOLD = 0.3


class ValidationDecision(str, Enum):
    """Decision gate outcomes for validated fields."""

    ACCEPT = "accept"
    RE_EXTRACT = "re_extract"
    FLAG = "flag"


@dataclass
class GroundingResult:
    """Result from visual grounding check for a single region."""

    region_id: str
    extracted_text: str
    ocr_text: str
    edit_distance: int
    grounding_score: float


def _levenshtein_distance(s1: str, s2: str) -> int:
    """Compute Levenshtein edit distance between two strings."""
    if len(s1) < len(s2):
        return _levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)

    prev_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = prev_row[j + 1] + 1
            deletions = curr_row[j] + 1
            substitutions = prev_row[j] + (c1 != c2)
            curr_row.append(min(insertions, deletions, substitutions))
        prev_row = curr_row

    return prev_row[-1]


def _normalized_edit_distance(s1: str, s2: str) -> float:
    """Compute normalized edit distance in [0, 1]. 0 = identical."""
    if not s1 and not s2:
        return 0.0
    dist = _levenshtein_distance(s1, s2)
    max_len = max(len(s1), len(s2))
    return dist / max_len if max_len > 0 else 0.0


def _crop_region(
    page_image_path: pathlib.Path,
    bbox_x: float,
    bbox_y: float,
    bbox_w: float,
    bbox_h: float,
    output_path: pathlib.Path,
) -> pathlib.Path:
    """Crop a region from a page image using normalized bbox coordinates."""
    img = Image.open(page_image_path)
    w, h = img.size
    left = int(bbox_x * w)
    top = int(bbox_y * h)
    right = int((bbox_x + bbox_w) * w)
    bottom = int((bbox_y + bbox_h) * h)
    # Clamp to image bounds
    left = max(0, min(left, w))
    top = max(0, min(top, h))
    right = max(left + 1, min(right, w))
    bottom = max(top + 1, min(bottom, h))
    cropped = img.crop((left, top, right, bottom))
    cropped.save(output_path)
    return output_path


def _get_text_from_region(region: Region) -> str | None:
    """Extract text content from a region, if applicable."""
    if isinstance(region.content, TextContent):
        return region.content.text
    if isinstance(region.content, HandwritingContent):
        return region.content.text
    return None


def check_visual_grounding(
    regions: list[Region],
    page_images: dict[int, pathlib.Path],
    ocr_tool: PaddleOCRTool | None = None,
) -> list[GroundingResult]:
    """Layer 4: Visual grounding check.

    For each text-bearing region, crop the bounding box from the original
    page image, run OCR, and compare with the extracted text.

    Args:
        regions: Extracted regions to check.
        page_images: Mapping of page number to page image path.
        ocr_tool: PaddleOCR tool for verification OCR. Uses default if None.

    Returns:
        List of GroundingResult objects for text-bearing regions.
    """
    if ocr_tool is None:
        ocr_tool = PaddleOCRTool()

    results: list[GroundingResult] = []

    for region in regions:
        extracted_text = _get_text_from_region(region)
        if extracted_text is None:
            continue

        page_path = page_images.get(region.page)
        if page_path is None:
            continue

        try:
            # Crop the bounding box region
            crop_path = page_path.parent / f"grounding_crop_{region.id}.png"
            _crop_region(
                page_path,
                region.bbox.x, region.bbox.y,
                region.bbox.w, region.bbox.h,
                crop_path,
            )

            # Run verification OCR on the crop
            ocr_result = ocr_tool.extract(crop_path)
            ocr_text = ocr_result.text

            # Compare
            edit_dist = _levenshtein_distance(
                extracted_text.lower().strip(),
                ocr_text.lower().strip(),
            )
            norm_dist = _normalized_edit_distance(
                extracted_text.lower().strip(),
                ocr_text.lower().strip(),
            )
            grounding_score = max(0.0, 1.0 - norm_dist)

            results.append(GroundingResult(
                region_id=region.id,
                extracted_text=extracted_text,
                ocr_text=ocr_text,
                edit_distance=edit_dist,
                grounding_score=grounding_score,
            ))

            # Clean up crop
            crop_path.unlink(missing_ok=True)

        except Exception as exc:
            logger.warning(
                "Visual grounding check failed for region %s: %s",
                region.id, exc,
            )

    return results


def compute_weighted_confidence(
    ocr_confidence: float,
    vlm_confidence: float,
    validation_score: float,
) -> float:
    """Layer 5: Compute weighted confidence score.

    Formula from design doc Section 7:
        field_confidence = ocr_confidence * 0.3
                         + vlm_confidence * 0.4
                         + validation_score * 0.3

    Args:
        ocr_confidence: Raw OCR tool confidence [0, 1].
        vlm_confidence: VLM extraction confidence [0, 1].
        validation_score: Validation pass score [0, 1].

    Returns:
        Weighted confidence in [0, 1].
    """
    raw = (
        ocr_confidence * OCR_WEIGHT
        + vlm_confidence * VLM_WEIGHT
        + validation_score * VALIDATION_WEIGHT
    )
    return max(0.0, min(1.0, raw))


def calibrate_confidence(
    raw_confidence: float,
    temperature: float = 1.0,
) -> float:
    """Layer 5: Apply temperature scaling to calibrate confidence.

    Temperature scaling (Guo et al. 2017) adjusts the sharpness of
    confidence scores. Temperature > 1 softens scores toward 0.5;
    temperature < 1 sharpens scores toward 0 or 1.

    The temperature parameter is learned from a held-out validation
    set to minimize Expected Calibration Error (ECE < 0.05).

    Args:
        raw_confidence: Uncalibrated confidence in [0, 1].
        temperature: Calibration temperature (default 1.0 = no change).

    Returns:
        Calibrated confidence in [0, 1].
    """
    if temperature <= 0:
        raise ValueError("Temperature must be positive")

    # Convert confidence to logit
    # Clamp to avoid log(0) or log(inf)
    p = max(1e-7, min(1.0 - 1e-7, raw_confidence))
    logit = math.log(p / (1.0 - p))

    # Apply temperature scaling
    scaled_logit = logit / temperature

    # Convert back to probability
    calibrated = 1.0 / (1.0 + math.exp(-scaled_logit))
    return calibrated


def make_validation_decision(
    confidence: float,
    accept_threshold: float = ACCEPT_THRESHOLD,
    re_extract_threshold: float = RE_EXTRACT_THRESHOLD,
) -> ValidationDecision:
    """Layer 5: Apply decision thresholds.

    Decision gate from design doc Section 7:
        confidence >= 0.90 -> ACCEPT
        0.70 <= confidence < 0.90 -> RE_EXTRACT
        confidence < 0.70 -> FLAG

    Args:
        confidence: Calibrated confidence score.
        accept_threshold: Minimum for ACCEPT (default 0.90).
        re_extract_threshold: Minimum for RE_EXTRACT (default 0.70).

    Returns:
        ValidationDecision enum value.
    """
    if confidence >= accept_threshold:
        return ValidationDecision.ACCEPT
    elif confidence >= re_extract_threshold:
        return ValidationDecision.RE_EXTRACT
    else:
        return ValidationDecision.FLAG
