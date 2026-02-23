# src/agentic_extract/coordinator/quality.py
"""Document quality assessment: DPI, skew, degradation scoring.

Provides a quick assessment of page image quality to determine
whether enhancement (via DocEnTr) is needed before extraction.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from PIL import Image

from agentic_extract.coordinator.ingestion import PageImage

# Degradation threshold: above this, DocEnTr enhancement is triggered
ENHANCEMENT_THRESHOLD = 0.5

# DPI below this is considered low quality
LOW_DPI_THRESHOLD = 150

# Ideal DPI for document scanning
IDEAL_DPI = 300


@dataclass
class QualityAssessment:
    """Quality assessment results for a page image."""

    dpi: int
    skew_angle: float
    degradation_score: float
    needs_enhancement: bool


def _estimate_contrast_ratio(img_array: np.ndarray) -> float:
    """Estimate contrast ratio from grayscale image array.

    Returns a value in [0, 1] where 1 is maximum contrast.
    Low contrast (washed out or very dark) yields low scores.
    """
    if img_array.size == 0:
        return 0.0
    min_val = float(np.min(img_array))
    max_val = float(np.max(img_array))
    if max_val == 0:
        return 0.0
    return (max_val - min_val) / 255.0


def _estimate_noise_level(img_array: np.ndarray) -> float:
    """Estimate noise level from grayscale image array.

    Uses standard deviation of local pixel differences as a
    proxy for noise. Returns value in [0, 1].
    """
    if img_array.size < 4:
        return 0.0
    # Compute horizontal differences
    h_diff = np.abs(np.diff(img_array.astype(np.float32), axis=1))
    noise = float(np.mean(h_diff)) / 255.0
    return min(1.0, noise)


def _estimate_skew(img_array: np.ndarray) -> float:
    """Estimate skew angle using a simple edge-based heuristic.

    This is a lightweight approximation. For production quality,
    Surya or a dedicated deskew tool would be used.
    Returns angle in degrees.
    """
    # Simple heuristic: variance of row-wise mean brightness
    # Skewed documents show gradual brightness transitions
    # This returns a small angle estimate (good enough for assessment)
    if img_array.shape[0] < 10:
        return 0.0

    row_means = np.mean(img_array, axis=1)
    # Compute gradient of row means
    gradient = np.diff(row_means.astype(np.float64))
    if len(gradient) == 0:
        return 0.0

    # Estimate skew from gradient variance (heuristic)
    grad_std = float(np.std(gradient))
    # Map to approximate degrees (rough heuristic)
    angle = min(15.0, grad_std * 0.1)
    return round(angle, 2)


def assess_quality(page: PageImage) -> QualityAssessment:
    """Assess the quality of a page image.

    Evaluates DPI, skew angle, and degradation (contrast + noise).
    Sets needs_enhancement=True if degradation_score exceeds threshold.

    Args:
        page: PageImage with path to the image file.

    Returns:
        QualityAssessment with all metrics.
    """
    img = Image.open(page.image_path).convert("L")  # grayscale
    img_array = np.array(img)

    # DPI score: penalty for low DPI
    dpi_score = min(1.0, page.dpi / IDEAL_DPI)
    dpi_penalty = max(0.0, 1.0 - dpi_score)

    # Contrast: low contrast = degraded
    contrast = _estimate_contrast_ratio(img_array)
    contrast_penalty = max(0.0, 1.0 - contrast)

    # Noise estimation
    noise = _estimate_noise_level(img_array)

    # Skew estimation
    skew = _estimate_skew(img_array)
    skew_penalty = min(1.0, abs(skew) / 15.0)

    # Composite degradation score (weighted average)
    degradation_score = (
        dpi_penalty * 0.3
        + contrast_penalty * 0.35
        + noise * 0.25
        + skew_penalty * 0.1
    )
    degradation_score = min(1.0, max(0.0, degradation_score))

    return QualityAssessment(
        dpi=page.dpi,
        skew_angle=skew,
        degradation_score=round(degradation_score, 4),
        needs_enhancement=degradation_score > ENHANCEMENT_THRESHOLD,
    )
