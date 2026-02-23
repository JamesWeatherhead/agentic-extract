# src/agentic_extract/coordinator/reading_order.py
"""Reading order determination using Surya with geometric fallback.

Surya provides ML-based reading order detection. When unavailable,
a geometric heuristic (top-to-bottom, left-to-right with column
detection) is used as fallback.
"""
from __future__ import annotations

import json
import logging
import pathlib

from agentic_extract.coordinator.layout import LayoutRegion
from agentic_extract.tools.docker_runner import DockerTool

logger = logging.getLogger(__name__)


def fallback_reading_order(regions: list[LayoutRegion]) -> list[str]:
    """Geometric reading order: sort by page, then y, then x.

    This handles single-column and basic two-column layouts.
    Regions at similar y-positions (within 2% of page height)
    are sorted left-to-right.
    """
    Y_TOLERANCE = 0.02

    def sort_key(r: LayoutRegion) -> tuple[int, float, float]:
        # Round y to nearest tolerance band to group same-row items
        y_band = round(r.bbox.y / Y_TOLERANCE) * Y_TOLERANCE
        return (r.page, y_band, r.bbox.x)

    sorted_regions = sorted(regions, key=sort_key)
    return [r.region_id for r in sorted_regions]


class SuryaReadingOrder:
    """Surya reading order detection via Docker.

    Runs the Surya model to determine the correct reading
    order of detected regions on a page.
    """

    IMAGE_NAME = "surya-ocr:latest"

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

    def get_reading_order(
        self,
        image_path: pathlib.Path,
        region_ids: list[str],
    ) -> list[str]:
        """Run Surya to determine reading order.

        Args:
            image_path: Path to the page image.
            region_ids: List of region IDs to order.

        Returns:
            Ordered list of region IDs.

        Raises:
            RuntimeError: If Surya fails.
        """
        tool = self._docker_tool(self._image_name, self._volumes)
        result = tool.run(["--input", str(image_path), "--format", "json"])

        if result.exit_code != 0:
            raise RuntimeError(
                f"Surya reading order failed (exit {result.exit_code}): "
                f"{result.stderr}"
            )

        data = json.loads(result.stdout)
        return data.get("reading_order", region_ids)


def determine_reading_order(
    image_path: pathlib.Path,
    regions: list[LayoutRegion],
) -> list[str]:
    """Determine reading order, falling back to geometric sort on failure.

    Tries Surya first. If it fails (Docker not available, model error),
    falls back to the geometric heuristic.
    """
    region_ids = [r.region_id for r in regions]
    try:
        surya = SuryaReadingOrder()
        return surya.get_reading_order(image_path, region_ids)
    except Exception as exc:
        logger.warning("Surya reading order failed, using fallback: %s", exc)
        return fallback_reading_order(regions)
