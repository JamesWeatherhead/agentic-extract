# src/agentic_extract/specialists/visual_formula.py
"""Visual Specialist - Formula Mode: GOT-OCR 2.0 + pix2tex + voting.

Two independent formula-to-LaTeX tools run in parallel:
1. GOT-OCR 2.0 (580M params): unified OCR for formulas
2. pix2tex / LaTeX-OCR: dedicated formula conversion

When both succeed:
- If they agree: confidence boosted, high reliability
- If they disagree: higher-confidence result chosen, flagged for review
When one fails: the surviving tool's output is used.
"""
from __future__ import annotations

import json
import logging
import pathlib
from dataclasses import dataclass

from agentic_extract.models import (
    BoundingBox,
    FormulaContent,
    Region,
    RegionType,
)
from agentic_extract.tools.docker_runner import DockerTool

logger = logging.getLogger(__name__)

# Confidence boost when both tools agree
AGREEMENT_BOOST = 0.05


@dataclass
class GotOCRResult:
    """Raw GOT-OCR 2.0 formula extraction result."""

    latex: str
    confidence: float


@dataclass
class Pix2TexResult:
    """Raw pix2tex / LaTeX-OCR formula extraction result."""

    latex: str
    confidence: float


class GotOCRTool:
    """GOT-OCR 2.0 Docker wrapper for formula-to-LaTeX conversion.

    First model treating all optical signals uniformly (580M params).
    """

    IMAGE_NAME = "got-ocr2:latest"

    def __init__(
        self,
        image_name: str | None = None,
        volumes: dict[str, str] | None = None,
    ) -> None:
        self._image_name = image_name or self.IMAGE_NAME
        self._volumes = volumes or {}

    @staticmethod
    def _docker_tool(
        image_name: str, volumes: dict[str, str],
    ) -> DockerTool:
        return DockerTool(
            image_name=image_name,
            default_timeout=120,
            volumes=volumes,
        )

    def extract(self, image_path: pathlib.Path) -> GotOCRResult:
        """Run GOT-OCR 2.0 on a formula image.

        Args:
            image_path: Path to the formula region image.

        Returns:
            GotOCRResult with LaTeX string and confidence.

        Raises:
            RuntimeError: If GOT-OCR container fails.
        """
        tool = self._docker_tool(self._image_name, self._volumes)
        result = tool.run([
            "--input", str(image_path),
            "--task", "formula",
            "--format", "json",
        ])

        if result.exit_code != 0:
            raise RuntimeError(
                f"GOT-OCR failed (exit {result.exit_code}): {result.stderr}"
            )

        data = json.loads(result.stdout)
        return GotOCRResult(
            latex=data.get("latex", ""),
            confidence=data.get("confidence", 0.0),
        )


class Pix2TexTool:
    """pix2tex / LaTeX-OCR Docker wrapper for formula-to-LaTeX conversion.

    Primary open-source tool for handwritten equation conversion.
    """

    IMAGE_NAME = "pix2tex:latest"

    def __init__(
        self,
        image_name: str | None = None,
        volumes: dict[str, str] | None = None,
    ) -> None:
        self._image_name = image_name or self.IMAGE_NAME
        self._volumes = volumes or {}

    @staticmethod
    def _docker_tool(
        image_name: str, volumes: dict[str, str],
    ) -> DockerTool:
        return DockerTool(
            image_name=image_name,
            default_timeout=120,
            volumes=volumes,
        )

    def extract(self, image_path: pathlib.Path) -> Pix2TexResult:
        """Run pix2tex on a formula image.

        Args:
            image_path: Path to the formula region image.

        Returns:
            Pix2TexResult with LaTeX string and confidence.

        Raises:
            RuntimeError: If pix2tex container fails.
        """
        tool = self._docker_tool(self._image_name, self._volumes)
        result = tool.run(["--input", str(image_path), "--format", "json"])

        if result.exit_code != 0:
            raise RuntimeError(
                f"pix2tex failed (exit {result.exit_code}): {result.stderr}"
            )

        data = json.loads(result.stdout)
        return Pix2TexResult(
            latex=data.get("latex", ""),
            confidence=data.get("confidence", 0.0),
        )


class FormulaSpecialist:
    """Formula extraction specialist: GOT-OCR 2.0 + pix2tex with voting.

    Both tools run independently. Agreement boosts confidence,
    disagreement picks the higher-confidence result and flags for review.
    """

    def __init__(
        self,
        gotocr_tool: GotOCRTool | None = None,
        pix2tex_tool: Pix2TexTool | None = None,
    ) -> None:
        self.gotocr_tool = gotocr_tool or GotOCRTool()
        self.pix2tex_tool = pix2tex_tool or Pix2TexTool()

    def extract_sync(
        self,
        image_path: pathlib.Path,
        region_id: str,
        page: int,
        bbox: BoundingBox,
    ) -> Region:
        """Extract formula from a region image (synchronous).

        Both tools are CPU/GPU Docker containers, so they run
        synchronously. The async wrapper can be added at the
        orchestration layer using asyncio.to_thread.

        Args:
            image_path: Path to the cropped formula image.
            region_id: Unique identifier for this region.
            page: Page number (1-based).
            bbox: Normalized bounding box of this region.

        Returns:
            Region with FormulaContent populated.
        """
        got_result: GotOCRResult | None = None
        p2t_result: Pix2TexResult | None = None

        # Run GOT-OCR 2.0
        try:
            got_result = self.gotocr_tool.extract(image_path)
        except Exception as exc:
            logger.warning("GOT-OCR failed for %s: %s", region_id, exc)

        # Run pix2tex
        try:
            p2t_result = self.pix2tex_tool.extract(image_path)
        except Exception as exc:
            logger.warning("pix2tex failed for %s: %s", region_id, exc)

        # Both failed
        if got_result is None and p2t_result is None:
            return Region(
                id=region_id,
                type=RegionType.FORMULA,
                subtype=None,
                page=page,
                bbox=bbox,
                content=FormulaContent(latex="", mathml=None),
                confidence=0.0,
                extraction_method="none",
                needs_review=True,
                review_reason="Both GOT-OCR and pix2tex failed",
            )

        # Only one succeeded
        if got_result is None:
            return Region(
                id=region_id,
                type=RegionType.FORMULA,
                subtype=None,
                page=page,
                bbox=bbox,
                content=FormulaContent(latex=p2t_result.latex, mathml=None),
                confidence=p2t_result.confidence,
                extraction_method="pix2tex",
                needs_review=p2t_result.confidence < 0.90,
                review_reason=(
                    f"Formula confidence {p2t_result.confidence:.2f} below 0.90"
                    if p2t_result.confidence < 0.90
                    else None
                ),
            )

        if p2t_result is None:
            return Region(
                id=region_id,
                type=RegionType.FORMULA,
                subtype=None,
                page=page,
                bbox=bbox,
                content=FormulaContent(latex=got_result.latex, mathml=None),
                confidence=got_result.confidence,
                extraction_method="got-ocr",
                needs_review=got_result.confidence < 0.90,
                review_reason=(
                    f"Formula confidence {got_result.confidence:.2f} below 0.90"
                    if got_result.confidence < 0.90
                    else None
                ),
            )

        # Both succeeded: vote
        tools_agree = got_result.latex.strip() == p2t_result.latex.strip()

        if tools_agree:
            boosted = min(1.0, max(got_result.confidence, p2t_result.confidence) + AGREEMENT_BOOST)
            return Region(
                id=region_id,
                type=RegionType.FORMULA,
                subtype=None,
                page=page,
                bbox=bbox,
                content=FormulaContent(latex=got_result.latex, mathml=None),
                confidence=boosted,
                extraction_method="got-ocr + pix2tex (unanimous)",
                needs_review=boosted < 0.90,
                review_reason=(
                    f"Formula confidence {boosted:.2f} below 0.90"
                    if boosted < 0.90
                    else None
                ),
            )

        # Disagreement: pick higher confidence, flag for review
        if got_result.confidence >= p2t_result.confidence:
            winner_latex = got_result.latex
            winner_conf = got_result.confidence
        else:
            winner_latex = p2t_result.latex
            winner_conf = p2t_result.confidence

        return Region(
            id=region_id,
            type=RegionType.FORMULA,
            subtype=None,
            page=page,
            bbox=bbox,
            content=FormulaContent(latex=winner_latex, mathml=None),
            confidence=winner_conf,
            extraction_method="got-ocr + pix2tex (majority)",
            needs_review=True,
            review_reason=(
                f"GOT-OCR and pix2tex disagree: "
                f"GOT='{got_result.latex}' (conf={got_result.confidence:.2f}) vs "
                f"P2T='{p2t_result.latex}' (conf={p2t_result.confidence:.2f})"
            ),
        )
