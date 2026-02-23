# src/agentic_extract/coordinator/layout.py
"""Layout detection using DocLayout-YOLO in Docker.

Runs DocLayout-YOLO on page images to detect regions (text blocks,
tables, figures, formulas) and returns normalized bounding boxes
with region type classifications.
"""
from __future__ import annotations

import json
import pathlib
import uuid
from dataclasses import dataclass

from PIL import Image

from agentic_extract.models import BoundingBox, RegionType
from agentic_extract.tools.docker_runner import DockerTool


# DocLayout-YOLO class ID to RegionType mapping
YOLO_CLASS_MAP: dict[int, RegionType] = {
    0: RegionType.TEXT,          # text block
    1: RegionType.TABLE,         # table
    2: RegionType.FIGURE,        # figure
    3: RegionType.FORMULA,       # formula / equation
    4: RegionType.TEXT,          # caption (treat as text)
    5: RegionType.TEXT,          # list (treat as text)
    6: RegionType.TEXT,          # title / heading
    7: RegionType.TEXT,          # header
    8: RegionType.TEXT,          # footer
    9: RegionType.FORM_FIELD,   # form element
}


@dataclass
class LayoutRegion:
    """A region detected by layout analysis."""

    region_id: str
    region_type: RegionType
    bbox: BoundingBox
    confidence: float
    page: int


class DocLayoutYOLO:
    """DocLayout-YOLO wrapper for document layout detection.

    Runs the YOLO model inside a Docker container and parses
    the JSON output into LayoutRegion objects.
    """

    IMAGE_NAME = "doclayout-yolo:latest"

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

    def _map_class_id(self, class_id: int) -> RegionType:
        """Map a YOLO class ID to a RegionType."""
        return YOLO_CLASS_MAP.get(class_id, RegionType.TEXT)

    def detect(
        self,
        image_path: pathlib.Path,
        page_number: int,
    ) -> list[LayoutRegion]:
        """Run layout detection on a page image.

        Args:
            image_path: Path to the page image.
            page_number: 1-based page number.

        Returns:
            List of detected LayoutRegion objects.

        Raises:
            RuntimeError: If the Docker container fails.
        """
        img = Image.open(image_path)
        img_w, img_h = img.size

        tool = self._docker_tool(self._image_name, self._volumes)
        result = tool.run(["--input", str(image_path), "--format", "json"])

        if result.exit_code != 0:
            raise RuntimeError(
                f"Layout detection failed (exit {result.exit_code}): "
                f"{result.stderr}"
            )

        if not result.stdout.strip():
            return []

        raw_detections = json.loads(result.stdout)
        regions: list[LayoutRegion] = []

        for det in raw_detections:
            class_id = det["class_id"]
            conf = det["confidence"]
            x1, y1, x2, y2 = det["bbox"]

            # Normalize coordinates to [0, 1]
            nx = x1 / img_w
            ny = y1 / img_h
            nw = (x2 - x1) / img_w
            nh = (y2 - y1) / img_h

            # Clamp to valid range
            nx = max(0.0, min(1.0, nx))
            ny = max(0.0, min(1.0, ny))
            nw = max(0.0, min(1.0 - nx, nw))
            nh = max(0.0, min(1.0 - ny, nh))

            region_id = f"r_{page_number}_{uuid.uuid4().hex[:8]}"
            regions.append(
                LayoutRegion(
                    region_id=region_id,
                    region_type=self._map_class_id(class_id),
                    bbox=BoundingBox(x=nx, y=ny, w=nw, h=nh),
                    confidence=conf,
                    page=page_number,
                )
            )

        return regions


def detect_layout(
    image_path: pathlib.Path,
    page_number: int,
) -> list[LayoutRegion]:
    """Convenience function to detect layout on a single page."""
    detector = DocLayoutYOLO()
    return detector.detect(image_path, page_number)
