# tests/specialists/test_visual_figure.py
"""Tests for the Visual Specialist figure mode (FigEx2 + DECIMER + GelGenie + classifiers)."""
import json
import pathlib
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from PIL import Image

from agentic_extract.clients.vlm import VLMResponse
from agentic_extract.models import BoundingBox, FigureContent, Region, RegionType
from agentic_extract.specialists.visual_figure import (
    DecimerResult,
    DecimerTool,
    FigEx2Result,
    FigEx2Tool,
    FigureSpecialist,
    FigureTypeClassifier,
    GelGenieResult,
    GelGenieTool,
)
from agentic_extract.tools.docker_runner import ToolOutput


# --- FigEx2 Tool Tests ---

def test_figex2_result_dataclass():
    result = FigEx2Result(
        panel_paths=[pathlib.Path("/tmp/panel_a.png"), pathlib.Path("/tmp/panel_b.png")],
        panel_labels=["A", "B"],
        confidence=0.92,
    )
    assert len(result.panel_paths) == 2
    assert result.panel_labels == ["A", "B"]


@patch.object(FigEx2Tool, "_docker_tool")
def test_figex2_splits_panels(mock_tool: MagicMock, tmp_path: pathlib.Path):
    img = Image.new("RGB", (1000, 500), "white")
    img_path = tmp_path / "multi_panel.png"
    img.save(img_path)

    # Create fake panel images that FigEx2 would produce
    panel_a = tmp_path / "panel_0.png"
    panel_b = tmp_path / "panel_1.png"
    Image.new("RGB", (500, 500), "white").save(panel_a)
    Image.new("RGB", (500, 500), "white").save(panel_b)

    figex_output = json.dumps({
        "panels": [
            {"path": str(panel_a), "label": "A"},
            {"path": str(panel_b), "label": "B"},
        ],
        "confidence": 0.93,
    })
    mock_tool.return_value.run.return_value = ToolOutput(
        stdout=figex_output, stderr="", exit_code=0, duration_ms=1500,
    )

    tool = FigEx2Tool()
    result = tool.split(img_path, output_dir=tmp_path)

    assert len(result.panel_paths) == 2
    assert result.panel_labels == ["A", "B"]
    assert result.confidence == 0.93


@patch.object(FigEx2Tool, "_docker_tool")
def test_figex2_single_panel(mock_tool: MagicMock, tmp_path: pathlib.Path):
    """A single-panel figure should return one panel (the original)."""
    img = Image.new("RGB", (500, 400), "white")
    img_path = tmp_path / "single.png"
    img.save(img_path)

    figex_output = json.dumps({
        "panels": [{"path": str(img_path), "label": ""}],
        "confidence": 0.98,
    })
    mock_tool.return_value.run.return_value = ToolOutput(
        stdout=figex_output, stderr="", exit_code=0, duration_ms=800,
    )

    tool = FigEx2Tool()
    result = tool.split(img_path, output_dir=tmp_path)
    assert len(result.panel_paths) == 1


@patch.object(FigEx2Tool, "_docker_tool")
def test_figex2_handles_failure(mock_tool: MagicMock, tmp_path: pathlib.Path):
    img = Image.new("RGB", (100, 100), "white")
    img_path = tmp_path / "bad.png"
    img.save(img_path)

    mock_tool.return_value.run.return_value = ToolOutput(
        stdout="", stderr="FigEx2 error", exit_code=1, duration_ms=50,
    )

    tool = FigEx2Tool()
    with pytest.raises(RuntimeError, match="FigEx2 failed"):
        tool.split(img_path, output_dir=tmp_path)


# --- DECIMER Tool Tests ---

def test_decimer_result_dataclass():
    result = DecimerResult(
        smiles="CC(=O)OC1=CC=CC=C1C(=O)O",
        inchi="InChI=1S/C9H8O4/c...",
        confidence=0.91,
    )
    assert "CC(=O)" in result.smiles


@patch.object(DecimerTool, "_docker_tool")
def test_decimer_extracts_smiles(mock_tool: MagicMock, tmp_path: pathlib.Path):
    img = Image.new("RGB", (300, 300), "white")
    img_path = tmp_path / "molecule.png"
    img.save(img_path)

    decimer_output = json.dumps({
        "smiles": "CC(=O)OC1=CC=CC=C1C(=O)O",
        "inchi": "InChI=1S/C9H8O4/c1-6(10)13-8-5-3-2-4-7(8)9(11)12/h2-5H,1H3,(H,11,12)",
        "confidence": 0.94,
    })
    mock_tool.return_value.run.return_value = ToolOutput(
        stdout=decimer_output, stderr="", exit_code=0, duration_ms=2000,
    )

    tool = DecimerTool()
    result = tool.extract(img_path)
    assert "CC(=O)" in result.smiles
    assert result.inchi.startswith("InChI=")
    assert result.confidence == 0.94


@patch.object(DecimerTool, "_docker_tool")
def test_decimer_handles_failure(mock_tool: MagicMock, tmp_path: pathlib.Path):
    img = Image.new("RGB", (100, 100), "white")
    img_path = tmp_path / "bad_mol.png"
    img.save(img_path)

    mock_tool.return_value.run.return_value = ToolOutput(
        stdout="", stderr="DECIMER error", exit_code=1, duration_ms=50,
    )

    tool = DecimerTool()
    with pytest.raises(RuntimeError, match="DECIMER failed"):
        tool.extract(img_path)


# --- GelGenie Tool Tests ---

def test_gelgenie_result_dataclass():
    result = GelGenieResult(
        bands=[
            {"lane": 1, "position": 0.3, "intensity": 0.95},
            {"lane": 1, "position": 0.7, "intensity": 0.42},
        ],
        lane_count=2,
        confidence=0.88,
    )
    assert len(result.bands) == 2
    assert result.lane_count == 2


@patch.object(GelGenieTool, "_docker_tool")
def test_gelgenie_identifies_bands(mock_tool: MagicMock, tmp_path: pathlib.Path):
    img = Image.new("RGB", (400, 600), "white")
    img_path = tmp_path / "gel.png"
    img.save(img_path)

    gel_output = json.dumps({
        "bands": [
            {"lane": 1, "position": 0.25, "intensity": 0.90},
            {"lane": 2, "position": 0.25, "intensity": 0.85},
        ],
        "lane_count": 2,
        "confidence": 0.87,
    })
    mock_tool.return_value.run.return_value = ToolOutput(
        stdout=gel_output, stderr="", exit_code=0, duration_ms=1800,
    )

    tool = GelGenieTool()
    result = tool.extract(img_path)
    assert len(result.bands) == 2
    assert result.lane_count == 2
    assert result.confidence == 0.87


@patch.object(GelGenieTool, "_docker_tool")
def test_gelgenie_handles_failure(mock_tool: MagicMock, tmp_path: pathlib.Path):
    img = Image.new("RGB", (100, 100), "white")
    img_path = tmp_path / "bad_gel.png"
    img.save(img_path)

    mock_tool.return_value.run.return_value = ToolOutput(
        stdout="", stderr="GelGenie error", exit_code=1, duration_ms=50,
    )

    tool = GelGenieTool()
    with pytest.raises(RuntimeError, match="GelGenie failed"):
        tool.extract(img_path)


# --- Figure Type Classifier Tests ---

def test_classifier_molecular_by_keyword():
    clf = FigureTypeClassifier()
    assert clf.classify_deterministic("molecular structure of aspirin") == "molecular"


def test_classifier_gel_by_keyword():
    clf = FigureTypeClassifier()
    assert clf.classify_deterministic("gel electrophoresis results") == "gel"


def test_classifier_general_fallback():
    clf = FigureTypeClassifier()
    assert clf.classify_deterministic("overview diagram") == "general"


@pytest.mark.asyncio
async def test_classifier_ambiguous_uses_claude():
    clf = FigureTypeClassifier()
    mock_claude = AsyncMock()
    mock_claude.send_vision_request.return_value = VLMResponse(
        content={"figure_type": "molecular"},
        confidence=0.88,
        model="claude-opus-4-20250514",
        usage_tokens=100,
        duration_ms=800,
    )

    result = await clf.classify(
        caption="Figure 3: Results of the experiment",
        image_path=pathlib.Path("/tmp/test.png"),
        claude_client=mock_claude,
    )
    # Deterministic returns "general", but Claude overrides to "molecular"
    assert result == "molecular"


# --- Figure Specialist Integration Tests ---

@pytest.mark.asyncio
async def test_figure_specialist_molecular(tmp_path: pathlib.Path):
    img = Image.new("RGB", (300, 300), "white")
    img_path = tmp_path / "molecule_fig.png"
    img.save(img_path)

    mock_figex2 = MagicMock()
    mock_figex2.split.return_value = FigEx2Result(
        panel_paths=[img_path], panel_labels=[""], confidence=0.98,
    )

    mock_decimer = MagicMock()
    mock_decimer.extract.return_value = DecimerResult(
        smiles="CC(=O)O", inchi="InChI=1S/C2H4O2/c1-2(3)4/h1H3,(H,3,4)",
        confidence=0.92,
    )

    mock_codex = AsyncMock()
    mock_codex.send_vision_request.return_value = VLMResponse(
        content={"figure_type": "molecular_structure", "elements": ["C", "H", "O"]},
        confidence=0.89,
        model="gpt-4o",
        usage_tokens=200,
        duration_ms=1200,
    )

    mock_claude = AsyncMock()
    mock_claude.send_vision_request.return_value = VLMResponse(
        content={"description": "Acetic acid molecular structure"},
        confidence=0.91,
        model="claude-opus-4-20250514",
        usage_tokens=150,
        duration_ms=1000,
    )

    mock_classifier = MagicMock()
    mock_classifier.classify = AsyncMock(return_value="molecular")

    specialist = FigureSpecialist(
        figex2_tool=mock_figex2,
        decimer_tool=mock_decimer,
        gelgenie_tool=MagicMock(),
        classifier=mock_classifier,
        codex_client=mock_codex,
        claude_client=mock_claude,
    )

    region = await specialist.extract(
        image_path=img_path,
        region_id="f1",
        page=4,
        bbox=BoundingBox(x=0.1, y=0.2, w=0.8, h=0.5),
        caption="Figure 1: Molecular structure of acetic acid",
    )

    assert isinstance(region, Region)
    assert region.type == RegionType.FIGURE
    assert isinstance(region.content, FigureContent)
    assert "CC(=O)O" in region.content.figure_json.get("smiles", "")
    assert "decimer" in region.extraction_method


@pytest.mark.asyncio
async def test_figure_specialist_gel(tmp_path: pathlib.Path):
    img = Image.new("RGB", (400, 600), "white")
    img_path = tmp_path / "gel_fig.png"
    img.save(img_path)

    mock_figex2 = MagicMock()
    mock_figex2.split.return_value = FigEx2Result(
        panel_paths=[img_path], panel_labels=[""], confidence=0.97,
    )

    mock_gelgenie = MagicMock()
    mock_gelgenie.extract.return_value = GelGenieResult(
        bands=[{"lane": 1, "position": 0.3, "intensity": 0.9}],
        lane_count=1,
        confidence=0.86,
    )

    mock_classifier = MagicMock()
    mock_classifier.classify = AsyncMock(return_value="gel")

    specialist = FigureSpecialist(
        figex2_tool=mock_figex2,
        decimer_tool=MagicMock(),
        gelgenie_tool=mock_gelgenie,
        classifier=mock_classifier,
        codex_client=None,
        claude_client=None,
    )

    region = await specialist.extract(
        image_path=img_path,
        region_id="f2",
        page=5,
        bbox=BoundingBox(x=0.0, y=0.0, w=1.0, h=1.0),
        caption="Figure 2: Gel electrophoresis",
    )

    assert region.content.figure_json.get("lane_count") == 1
    assert "gelgenie" in region.extraction_method


@pytest.mark.asyncio
async def test_figure_specialist_general(tmp_path: pathlib.Path):
    img = Image.new("RGB", (500, 400), "white")
    img_path = tmp_path / "general_fig.png"
    img.save(img_path)

    mock_figex2 = MagicMock()
    mock_figex2.split.return_value = FigEx2Result(
        panel_paths=[img_path], panel_labels=[""], confidence=0.95,
    )

    mock_codex = AsyncMock()
    mock_codex.send_vision_request.return_value = VLMResponse(
        content={"figure_type": "diagram", "elements": ["box1", "arrow", "box2"]},
        confidence=0.85,
        model="gpt-4o",
        usage_tokens=180,
        duration_ms=1100,
    )

    mock_claude = AsyncMock()
    mock_claude.send_vision_request.return_value = VLMResponse(
        content={"description": "A workflow diagram showing data processing steps."},
        confidence=0.90,
        model="claude-opus-4-20250514",
        usage_tokens=120,
        duration_ms=900,
    )

    mock_classifier = MagicMock()
    mock_classifier.classify = AsyncMock(return_value="general")

    specialist = FigureSpecialist(
        figex2_tool=mock_figex2,
        decimer_tool=MagicMock(),
        gelgenie_tool=MagicMock(),
        classifier=mock_classifier,
        codex_client=mock_codex,
        claude_client=mock_claude,
    )

    region = await specialist.extract(
        image_path=img_path,
        region_id="f3",
        page=2,
        bbox=BoundingBox(x=0.1, y=0.1, w=0.8, h=0.6),
        caption="Figure 3: System overview",
    )

    assert isinstance(region.content, FigureContent)
    assert "workflow" in region.content.description.lower() or "diagram" in region.content.figure_type
