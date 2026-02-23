"""Tests for document ingestion (PDF and image handling)."""
import pathlib
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from agentic_extract.coordinator.ingestion import (
    IngestionResult,
    PageImage,
    ingest,
)


def test_page_image_dataclass():
    pi = PageImage(
        page_number=1,
        image_path=pathlib.Path("/tmp/page_1.png"),
        width=2550,
        height=3300,
        dpi=300,
    )
    assert pi.page_number == 1
    assert pi.dpi == 300


def test_ingestion_result_dataclass():
    result = IngestionResult(
        pages=[],
        temp_dir=pathlib.Path("/tmp/ae_test"),
        source_file=pathlib.Path("test.png"),
        page_count=0,
    )
    assert result.page_count == 0
    assert result.pages == []


def test_ingest_single_image(tmp_path: pathlib.Path):
    """Ingesting a single image should produce one page."""
    img = Image.new("RGB", (200, 300), "white")
    img_path = tmp_path / "scan.png"
    img.save(img_path, dpi=(150, 150))

    result = ingest(img_path, output_dir=tmp_path / "output")

    assert result.page_count == 1
    assert len(result.pages) == 1
    assert result.pages[0].page_number == 1
    assert result.pages[0].width == 200
    assert result.pages[0].height == 300
    assert result.pages[0].image_path.exists()
    assert result.source_file == img_path


def test_ingest_single_image_jpeg(tmp_path: pathlib.Path):
    """JPEG images should also work."""
    img = Image.new("RGB", (100, 100), "blue")
    img_path = tmp_path / "photo.jpg"
    img.save(img_path)

    result = ingest(img_path, output_dir=tmp_path / "output")
    assert result.page_count == 1
    assert result.pages[0].image_path.suffix == ".png"


@patch("agentic_extract.coordinator.ingestion._convert_pdf_to_images")
def test_ingest_pdf(mock_convert: MagicMock, tmp_path: pathlib.Path):
    """PDFs should be converted to page images via pdf2image."""
    # Create fake page images that the mock will "produce"
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    for i in range(3):
        img = Image.new("RGB", (612, 792), "white")
        img.save(output_dir / f"page_{i + 1}.png", dpi=(72, 72))

    mock_convert.return_value = [
        output_dir / "page_1.png",
        output_dir / "page_2.png",
        output_dir / "page_3.png",
    ]

    pdf_path = tmp_path / "document.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 fake pdf content")

    result = ingest(pdf_path, output_dir=output_dir)

    assert result.page_count == 3
    assert len(result.pages) == 3
    assert result.pages[0].page_number == 1
    assert result.pages[2].page_number == 3
    mock_convert.assert_called_once()


def test_ingest_unsupported_format(tmp_path: pathlib.Path):
    """Unsupported file types should raise ValueError."""
    bad_file = tmp_path / "data.csv"
    bad_file.write_text("a,b,c\n1,2,3")

    with pytest.raises(ValueError, match="Unsupported file type"):
        ingest(bad_file, output_dir=tmp_path / "output")


def test_ingest_missing_file(tmp_path: pathlib.Path):
    """Missing files should raise FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        ingest(tmp_path / "nonexistent.pdf", output_dir=tmp_path / "output")
