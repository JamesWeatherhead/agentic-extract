"""Document ingestion: detect file type and convert to page images.

Supports PDF (via pdf2image) and common image formats (PNG, JPEG, TIFF).
All pages are normalized to PNG for downstream processing.
"""
from __future__ import annotations

import pathlib
import shutil
from dataclasses import dataclass, field

from PIL import Image


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp"}
PDF_EXTENSIONS = {".pdf"}
SUPPORTED_EXTENSIONS = IMAGE_EXTENSIONS | PDF_EXTENSIONS


@dataclass
class PageImage:
    """A single page converted to an image."""

    page_number: int
    image_path: pathlib.Path
    width: int
    height: int
    dpi: int


@dataclass
class IngestionResult:
    """Result of document ingestion."""

    pages: list[PageImage]
    temp_dir: pathlib.Path
    source_file: pathlib.Path
    page_count: int


def _convert_pdf_to_images(
    pdf_path: pathlib.Path,
    output_dir: pathlib.Path,
    dpi: int = 300,
) -> list[pathlib.Path]:
    """Convert a PDF to a list of page images using pdf2image.

    Returns list of paths to the generated PNG files.
    """
    from pdf2image import convert_from_path

    images = convert_from_path(str(pdf_path), dpi=dpi)
    paths: list[pathlib.Path] = []
    for i, img in enumerate(images):
        out_path = output_dir / f"page_{i + 1}.png"
        img.save(out_path, "PNG")
        paths.append(out_path)
    return paths


def _get_dpi(img: Image.Image) -> int:
    """Extract DPI from image metadata, defaulting to 72."""
    info = img.info
    dpi_val = info.get("dpi", (72, 72))
    if isinstance(dpi_val, tuple):
        return int(dpi_val[0])
    return int(dpi_val)


def ingest(
    file_path: pathlib.Path,
    output_dir: pathlib.Path | None = None,
) -> IngestionResult:
    """Ingest a document file and produce page images.

    Args:
        file_path: Path to the input file (PDF or image).
        output_dir: Directory to write page images. Created if needed.

    Returns:
        IngestionResult with page images and metadata.

    Raises:
        FileNotFoundError: If file_path does not exist.
        ValueError: If the file type is not supported.
    """
    file_path = pathlib.Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    suffix = file_path.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type: {suffix}. "
            f"Supported: {sorted(SUPPORTED_EXTENSIONS)}"
        )

    if output_dir is None:
        output_dir = file_path.parent / f"ae_pages_{file_path.stem}"
    output_dir.mkdir(parents=True, exist_ok=True)

    pages: list[PageImage] = []

    if suffix in PDF_EXTENSIONS:
        image_paths = _convert_pdf_to_images(file_path, output_dir)
        for i, img_path in enumerate(image_paths):
            img = Image.open(img_path)
            dpi = _get_dpi(img)
            pages.append(
                PageImage(
                    page_number=i + 1,
                    image_path=img_path,
                    width=img.width,
                    height=img.height,
                    dpi=dpi,
                )
            )
    else:
        # Single image file
        img = Image.open(file_path)
        dpi = _get_dpi(img)
        out_path = output_dir / f"page_1.png"
        img.save(out_path, "PNG")
        pages.append(
            PageImage(
                page_number=1,
                image_path=out_path,
                width=img.width,
                height=img.height,
                dpi=dpi,
            )
        )

    return IngestionResult(
        pages=pages,
        temp_dir=output_dir,
        source_file=file_path,
        page_count=len(pages),
    )
