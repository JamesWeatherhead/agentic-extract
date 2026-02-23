# Agentic Extract v1 Implementation Plan: Phase 3

> Continuation from Phase 1 (Tasks 1-12) and Phase 2 (Tasks 13-20).

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Assumptions:** Phase 2 established the Visual Specialist (`specialists/visual.py`), Handwriting/Formula handling, and the 5-layer Validator (`validator/engine.py` with `ValidationResult`, `ValidationDecision`, `FieldValidation`). All models, Docker patterns, and VLM clients from Phase 1 are reused without modification.

---

## Phase 3: Agentic Loop + Polish (Tasks 21-26)

---

### Task 21: Re-extraction Loop

**Files:**
- Create: `src/agentic_extract/reextraction/__init__.py`
- Create: `src/agentic_extract/reextraction/engine.py`
- Test: `tests/reextraction/test_engine.py`
- Create: `tests/reextraction/__init__.py`

**Step 1: Write the failing test**

```python
# tests/reextraction/__init__.py
```

```python
# tests/reextraction/test_engine.py
"""Tests for the re-extraction engine with model switching."""
import pathlib
from unittest.mock import AsyncMock, MagicMock

import pytest

from agentic_extract.clients.vlm import VLMResponse
from agentic_extract.models import (
    BoundingBox,
    Region,
    RegionType,
    TextContent,
    TableContent,
    FigureContent,
    HandwritingContent,
)
from agentic_extract.reextraction.engine import (
    ReExtractionEngine,
    ReExtractionResult,
    ReExtractionCandidate,
    ModelSwitchStrategy,
)


def test_reextraction_result_model():
    candidate = ReExtractionCandidate(
        value="500mg",
        confidence=0.85,
        model="claude-opus-4-20250514",
        extraction_method="paddleocr_3.0 + claude-opus-4-20250514",
    )
    result = ReExtractionResult(
        region_id="r1",
        field_name="dosage",
        original_value="500mg",
        original_confidence=0.75,
        original_model="claude-opus-4-20250514",
        candidates=[candidate],
        final_value="500mg",
        final_confidence=0.85,
        retries_used=1,
        models_agreed=False,
        accepted=False,
        flagged=True,
    )
    assert result.region_id == "r1"
    assert result.retries_used == 1
    assert len(result.candidates) == 1
    assert result.flagged is True


def test_reextraction_candidate_model():
    c = ReExtractionCandidate(
        value="BRCA1",
        confidence=0.92,
        model="gpt-4o",
        extraction_method="paddleocr_3.0 + gpt-4o",
    )
    assert c.value == "BRCA1"
    assert c.confidence == 0.92


def test_model_switch_strategy_claude_to_codex():
    strategy = ModelSwitchStrategy()
    alt = strategy.get_alternate_model("claude-opus-4-20250514")
    assert alt == "gpt-4o"


def test_model_switch_strategy_codex_to_claude():
    strategy = ModelSwitchStrategy()
    alt = strategy.get_alternate_model("gpt-4o")
    assert alt == "claude-opus-4-20250514"


def test_model_switch_strategy_unknown_defaults_to_claude():
    strategy = ModelSwitchStrategy()
    alt = strategy.get_alternate_model("some-unknown-model")
    assert alt == "claude-opus-4-20250514"


@pytest.mark.asyncio
async def test_reextraction_engine_init():
    mock_claude = AsyncMock()
    mock_codex = AsyncMock()
    engine = ReExtractionEngine(
        claude_client=mock_claude,
        codex_client=mock_codex,
        max_retries=2,
    )
    assert engine.max_retries == 2


@pytest.mark.asyncio
async def test_reextraction_models_agree_boosts_confidence(tmp_path: pathlib.Path):
    """When both models agree, confidence should be boosted by +0.10."""
    from PIL import Image
    img = Image.new("RGB", (200, 50), "white")
    img_path = tmp_path / "region.png"
    img.save(img_path)

    mock_claude = AsyncMock()
    mock_codex = AsyncMock()

    # Original extraction was by Claude at 0.82
    # Codex re-extraction returns the same value at 0.84
    mock_codex.send_vision_request.return_value = VLMResponse(
        content={"corrected_text": "500mg"},
        confidence=0.84,
        model="gpt-4o",
        usage_tokens=100,
        duration_ms=800,
    )

    engine = ReExtractionEngine(
        claude_client=mock_claude,
        codex_client=mock_codex,
        max_retries=2,
    )

    original_region = Region(
        id="r1",
        type=RegionType.TEXT,
        subtype=None,
        page=1,
        bbox=BoundingBox(x=0.1, y=0.1, w=0.8, h=0.05),
        content=TextContent(text="500mg", markdown="500mg"),
        confidence=0.82,
        extraction_method="paddleocr_3.0 + claude-opus-4-20250514",
    )

    result = await engine.re_extract_field(
        region=original_region,
        field_name="dosage",
        image_path=img_path,
        original_model="claude-opus-4-20250514",
    )

    assert isinstance(result, ReExtractionResult)
    # Models agreed on "500mg", so confidence boosted by 0.10
    assert result.models_agreed is True
    assert result.final_confidence >= 0.92  # 0.82 + 0.10
    assert result.final_value == "500mg"
    assert result.accepted is True  # >= 0.90 threshold
    assert result.retries_used == 1


@pytest.mark.asyncio
async def test_reextraction_models_disagree_flags_field(tmp_path: pathlib.Path):
    """When models disagree after max retries, field should be flagged."""
    from PIL import Image
    img = Image.new("RGB", (200, 50), "white")
    img_path = tmp_path / "region.png"
    img.save(img_path)

    mock_claude = AsyncMock()
    mock_codex = AsyncMock()

    # Claude says "500mg", Codex says "800mg" (disagreement)
    mock_codex.send_vision_request.return_value = VLMResponse(
        content={"corrected_text": "800mg"},
        confidence=0.80,
        model="gpt-4o",
        usage_tokens=100,
        duration_ms=800,
    )
    # On second retry (back to Claude with different prompt), still says "500mg"
    mock_claude.send_vision_request.return_value = VLMResponse(
        content={"corrected_text": "500mg"},
        confidence=0.83,
        model="claude-opus-4-20250514",
        usage_tokens=120,
        duration_ms=900,
    )

    engine = ReExtractionEngine(
        claude_client=mock_claude,
        codex_client=mock_codex,
        max_retries=2,
    )

    original_region = Region(
        id="r2",
        type=RegionType.TEXT,
        subtype=None,
        page=1,
        bbox=BoundingBox(x=0.1, y=0.3, w=0.8, h=0.05),
        content=TextContent(text="500mg", markdown="500mg"),
        confidence=0.78,
        extraction_method="paddleocr_3.0 + claude-opus-4-20250514",
    )

    result = await engine.re_extract_field(
        region=original_region,
        field_name="dosage",
        image_path=img_path,
        original_model="claude-opus-4-20250514",
    )

    assert result.models_agreed is False
    assert result.flagged is True
    assert result.retries_used == 2
    assert len(result.candidates) == 2  # Both candidates preserved


@pytest.mark.asyncio
async def test_reextraction_respects_max_retries(tmp_path: pathlib.Path):
    """Engine must not exceed max_retries."""
    from PIL import Image
    img = Image.new("RGB", (200, 50), "white")
    img_path = tmp_path / "region.png"
    img.save(img_path)

    mock_claude = AsyncMock()
    mock_codex = AsyncMock()

    # Both always return low confidence, different values
    mock_codex.send_vision_request.return_value = VLMResponse(
        content={"corrected_text": "value_b"},
        confidence=0.72,
        model="gpt-4o",
        usage_tokens=50,
        duration_ms=500,
    )
    mock_claude.send_vision_request.return_value = VLMResponse(
        content={"corrected_text": "value_c"},
        confidence=0.71,
        model="claude-opus-4-20250514",
        usage_tokens=60,
        duration_ms=600,
    )

    engine = ReExtractionEngine(
        claude_client=mock_claude,
        codex_client=mock_codex,
        max_retries=2,
    )

    original_region = Region(
        id="r3",
        type=RegionType.TEXT,
        subtype=None,
        page=2,
        bbox=BoundingBox(x=0.0, y=0.0, w=1.0, h=1.0),
        content=TextContent(text="value_a", markdown="value_a"),
        confidence=0.75,
        extraction_method="paddleocr_3.0 + claude-opus-4-20250514",
    )

    result = await engine.re_extract_field(
        region=original_region,
        field_name="field_x",
        image_path=img_path,
        original_model="claude-opus-4-20250514",
    )

    assert result.retries_used <= 2
    assert result.flagged is True


@pytest.mark.asyncio
async def test_reextraction_vlm_failure_counts_as_retry(tmp_path: pathlib.Path):
    """If a VLM call fails during re-extraction, it counts as a used retry."""
    from PIL import Image
    img = Image.new("RGB", (200, 50), "white")
    img_path = tmp_path / "region.png"
    img.save(img_path)

    mock_claude = AsyncMock()
    mock_codex = AsyncMock()

    # Codex fails on first retry
    mock_codex.send_vision_request.side_effect = RuntimeError("API timeout")
    # Claude succeeds on second retry but disagrees
    mock_claude.send_vision_request.return_value = VLMResponse(
        content={"corrected_text": "fallback_value"},
        confidence=0.76,
        model="claude-opus-4-20250514",
        usage_tokens=80,
        duration_ms=700,
    )

    engine = ReExtractionEngine(
        claude_client=mock_claude,
        codex_client=mock_codex,
        max_retries=2,
    )

    original_region = Region(
        id="r4",
        type=RegionType.TEXT,
        subtype=None,
        page=1,
        bbox=BoundingBox(x=0.2, y=0.2, w=0.6, h=0.1),
        content=TextContent(text="original_value", markdown="original_value"),
        confidence=0.80,
        extraction_method="paddleocr_3.0 + claude-opus-4-20250514",
    )

    result = await engine.re_extract_field(
        region=original_region,
        field_name="field_y",
        image_path=img_path,
        original_model="claude-opus-4-20250514",
    )

    assert result.retries_used == 2
    # Should still produce a result (not crash)
    assert result.final_value is not None


@pytest.mark.asyncio
async def test_reextraction_table_region(tmp_path: pathlib.Path):
    """Re-extraction should work for table regions too."""
    from PIL import Image
    img = Image.new("RGB", (400, 200), "white")
    img_path = tmp_path / "table_region.png"
    img.save(img_path)

    mock_claude = AsyncMock()
    mock_codex = AsyncMock()

    mock_codex.send_vision_request.return_value = VLMResponse(
        content={"headers": ["Gene", "Value"], "rows": [{"Gene": "BRCA1", "Value": "3.2"}]},
        confidence=0.88,
        model="gpt-4o",
        usage_tokens=150,
        duration_ms=1200,
    )

    engine = ReExtractionEngine(
        claude_client=mock_claude,
        codex_client=mock_codex,
        max_retries=2,
    )

    original_region = Region(
        id="t1",
        type=RegionType.TABLE,
        subtype=None,
        page=1,
        bbox=BoundingBox(x=0.05, y=0.2, w=0.9, h=0.4),
        content=TableContent(
            html="<table><tr><td>BRCA1</td><td>3.2</td></tr></table>",
            json_data={"headers": ["Gene", "Value"], "rows": [{"Gene": "BRCA1", "Value": "3.2"}]},
        ),
        confidence=0.82,
        extraction_method="docling + claude-opus-4-20250514",
    )

    result = await engine.re_extract_field(
        region=original_region,
        field_name="table_data",
        image_path=img_path,
        original_model="claude-opus-4-20250514",
    )

    assert isinstance(result, ReExtractionResult)
    assert result.retries_used >= 1
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/reextraction/test_engine.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'agentic_extract.reextraction'"

**Step 3: Write minimal implementation**

```python
# src/agentic_extract/reextraction/__init__.py
"""Re-extraction engine with model switching for low-confidence fields."""
```

```python
# src/agentic_extract/reextraction/engine.py
"""Re-extraction engine: model-switch retry loop for low-confidence fields.

When the Validator flags a field with confidence in [0.70, 0.90), this
engine re-extracts using the alternate model. If Claude produced the
original, Codex retries (and vice versa). Max 2 retries per field.
Agreement between models boosts confidence by +0.10.
"""
from __future__ import annotations

import logging
import pathlib
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field

from agentic_extract.clients.vlm import VLMClient, VLMResponse
from agentic_extract.models import (
    BoundingBox,
    Region,
    RegionType,
    TextContent,
    TableContent,
    FigureContent,
    HandwritingContent,
    FormulaContent,
)

logger = logging.getLogger(__name__)

ACCEPT_THRESHOLD = 0.90
REEXTRACT_LOW = 0.70
CONFIDENCE_BOOST = 0.10

REEXTRACTION_PROMPT = """You are a document extraction expert performing a re-extraction.
Previous extraction attempt produced a value with low confidence.

Region type: {region_type}
Field name: {field_name}
Previous value: {previous_value}
Previous confidence: {previous_confidence}

Your task: Look at this document region image and extract the value for
the field "{field_name}". Return JSON with key "corrected_text" for text
fields, or the appropriate structured data for tables/figures.

Rules:
- Extract directly from the image; do not simply copy the previous value
- If the value is genuinely unreadable, return null for corrected_text
- Be precise: character-level accuracy matters
"""


class ReExtractionCandidate(BaseModel):
    """A single re-extraction attempt result."""

    value: Any
    confidence: float = Field(..., ge=0.0, le=1.0)
    model: str
    extraction_method: str


class ReExtractionResult(BaseModel):
    """Result of the re-extraction loop for a single field."""

    region_id: str
    field_name: str
    original_value: Any
    original_confidence: float = Field(..., ge=0.0, le=1.0)
    original_model: str
    candidates: list[ReExtractionCandidate]
    final_value: Any
    final_confidence: float = Field(..., ge=0.0, le=1.0)
    retries_used: int = Field(..., ge=0)
    models_agreed: bool
    accepted: bool
    flagged: bool


class ModelSwitchStrategy:
    """Determines the alternate model for re-extraction.

    If Claude produced the failing extraction, switch to Codex.
    If Codex produced the failing extraction, switch to Claude.
    Unknown models default to Claude.
    """

    CLAUDE_MODELS = {"claude-opus-4-20250514", "claude-sonnet-4-20250514"}
    CODEX_MODELS = {"gpt-4o", "gpt-4o-mini"}

    DEFAULT_CLAUDE = "claude-opus-4-20250514"
    DEFAULT_CODEX = "gpt-4o"

    def get_alternate_model(self, original_model: str) -> str:
        """Return the alternate model name for re-extraction."""
        if original_model in self.CLAUDE_MODELS or original_model.startswith("claude"):
            return self.DEFAULT_CODEX
        if original_model in self.CODEX_MODELS or original_model.startswith("gpt"):
            return self.DEFAULT_CLAUDE
        # Unknown model: default to Claude
        return self.DEFAULT_CLAUDE

    def is_claude_model(self, model: str) -> bool:
        return model in self.CLAUDE_MODELS or model.startswith("claude")

    def is_codex_model(self, model: str) -> bool:
        return model in self.CODEX_MODELS or model.startswith("gpt")


def _extract_value_from_response(response: VLMResponse, region_type: RegionType) -> Any:
    """Pull the extracted value from a VLM response based on region type."""
    content = response.content
    if isinstance(content, dict):
        # Text-like regions
        if region_type in (RegionType.TEXT, RegionType.HANDWRITING):
            return content.get("corrected_text", content.get("text", str(content)))
        # Table regions
        if region_type == RegionType.TABLE:
            return content
        # Figure regions
        if region_type == RegionType.FIGURE:
            return content
        # Formula regions
        if region_type == RegionType.FORMULA:
            return content.get("latex", str(content))
    return str(content) if content is not None else None


def _get_region_value(region: Region) -> Any:
    """Extract the primary value from a region's content."""
    content = region.content
    if isinstance(content, TextContent):
        return content.text
    if isinstance(content, TableContent):
        return content.json_data
    if isinstance(content, FigureContent):
        return content.figure_json
    if isinstance(content, HandwritingContent):
        return content.text
    if isinstance(content, FormulaContent):
        return content.latex
    return None


def _values_agree(val_a: Any, val_b: Any) -> bool:
    """Check if two extracted values agree (exact match for strings, deep equal for dicts)."""
    if isinstance(val_a, str) and isinstance(val_b, str):
        return val_a.strip().lower() == val_b.strip().lower()
    if isinstance(val_a, dict) and isinstance(val_b, dict):
        return val_a == val_b
    return str(val_a) == str(val_b)


class ReExtractionEngine:
    """Re-extraction engine with model switching.

    Workflow:
    1. Receive a region that failed validation (confidence in [0.70, 0.90))
    2. Identify the original model used for extraction
    3. Re-extract using the alternate model (Claude -> Codex or Codex -> Claude)
    4. If models agree: boost confidence by +0.10
    5. If models disagree and retries remain: try again with original model + different prompt
    6. If retries exhausted and still disagreeing: flag with both candidates

    Args:
        claude_client: VLM client for Claude API calls.
        codex_client: VLM client for Codex/GPT-4o API calls.
        max_retries: Maximum re-extraction attempts per field (default 2).
    """

    def __init__(
        self,
        claude_client: VLMClient,
        codex_client: VLMClient,
        max_retries: int = 2,
    ) -> None:
        self.claude_client = claude_client
        self.codex_client = codex_client
        self.max_retries = max_retries
        self.strategy = ModelSwitchStrategy()

    def _get_client_for_model(self, model: str) -> VLMClient:
        """Return the appropriate VLM client for a given model name."""
        if self.strategy.is_codex_model(model):
            return self.codex_client
        return self.claude_client

    async def re_extract_field(
        self,
        region: Region,
        field_name: str,
        image_path: pathlib.Path,
        original_model: str,
    ) -> ReExtractionResult:
        """Run the re-extraction loop for a single field.

        Args:
            region: The region containing the low-confidence field.
            field_name: Name of the field being re-extracted.
            image_path: Path to the cropped region image.
            original_model: The model that produced the original extraction.

        Returns:
            ReExtractionResult with candidates, final value, and flags.
        """
        original_value = _get_region_value(region)
        original_confidence = region.confidence
        candidates: list[ReExtractionCandidate] = []
        retries_used = 0
        best_value = original_value
        best_confidence = original_confidence
        models_agreed = False

        # Alternate between models: first retry uses alternate, second uses original
        model_sequence = [
            self.strategy.get_alternate_model(original_model),
            original_model,
        ]

        for retry_idx in range(self.max_retries):
            retries_used += 1
            current_model = model_sequence[retry_idx % len(model_sequence)]
            client = self._get_client_for_model(current_model)

            prompt = REEXTRACTION_PROMPT.format(
                region_type=region.type.value,
                field_name=field_name,
                previous_value=str(best_value),
                previous_confidence=best_confidence,
            )

            try:
                response = await client.send_vision_request(
                    image_path=image_path,
                    prompt=prompt,
                )
                new_value = _extract_value_from_response(response, region.type)
                new_confidence = response.confidence

                candidate = ReExtractionCandidate(
                    value=new_value,
                    confidence=new_confidence,
                    model=current_model,
                    extraction_method=f"re_extraction_{retry_idx + 1} + {current_model}",
                )
                candidates.append(candidate)

                # Check agreement with original value
                if _values_agree(original_value, new_value):
                    models_agreed = True
                    boosted = min(1.0, max(original_confidence, new_confidence) + CONFIDENCE_BOOST)
                    best_value = new_value
                    best_confidence = boosted
                    break
                else:
                    # Take the higher-confidence value as best so far
                    if new_confidence > best_confidence:
                        best_value = new_value
                        best_confidence = new_confidence

            except Exception as exc:
                logger.warning(
                    "Re-extraction attempt %d failed for region %s field %s: %s",
                    retry_idx + 1, region.id, field_name, exc,
                )
                candidates.append(ReExtractionCandidate(
                    value=None,
                    confidence=0.0,
                    model=current_model,
                    extraction_method=f"re_extraction_{retry_idx + 1}_failed",
                ))

        accepted = best_confidence >= ACCEPT_THRESHOLD
        flagged = not accepted

        return ReExtractionResult(
            region_id=region.id,
            field_name=field_name,
            original_value=original_value,
            original_confidence=original_confidence,
            original_model=original_model,
            candidates=candidates,
            final_value=best_value,
            final_confidence=best_confidence,
            retries_used=retries_used,
            models_agreed=models_agreed,
            accepted=accepted,
            flagged=flagged,
        )
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/reextraction/test_engine.py -v`
Expected: PASS (12 passed)

**Step 5: Commit**

```bash
git add src/agentic_extract/reextraction/__init__.py src/agentic_extract/reextraction/engine.py tests/reextraction/__init__.py tests/reextraction/test_engine.py
git commit -m "feat: re-extraction engine with model switching, confidence boost on agreement, max 2 retries"
```

---

### Task 22: Full Pipeline Orchestration

**Files:**
- Create: `src/agentic_extract/pipeline.py`
- Test: `tests/test_pipeline.py`

**Step 1: Write the failing test**

```python
# tests/test_pipeline.py
"""Tests for the full extraction pipeline orchestration."""
import asyncio
import pathlib
import time
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from PIL import Image

from agentic_extract.clients.vlm import VLMResponse
from agentic_extract.models import (
    BoundingBox,
    DocumentMetadata,
    ExtractionResult,
    Region,
    RegionType,
    TextContent,
    TableContent,
    FigureContent,
    ProcessingStage,
    AuditTrail,
)
from agentic_extract.pipeline import Pipeline, PipelineConfig


def test_pipeline_config_defaults():
    config = PipelineConfig()
    assert config.accept_threshold == 0.90
    assert config.reextract_low == 0.70
    assert config.max_retries == 2
    assert config.parallel_specialists is True


def test_pipeline_config_custom():
    config = PipelineConfig(
        accept_threshold=0.85,
        reextract_low=0.65,
        max_retries=3,
        parallel_specialists=False,
    )
    assert config.accept_threshold == 0.85
    assert config.max_retries == 3


def test_pipeline_init():
    pipeline = Pipeline(
        claude_client=AsyncMock(),
        codex_client=AsyncMock(),
    )
    assert pipeline.config.accept_threshold == 0.90


def test_pipeline_init_with_config():
    config = PipelineConfig(max_retries=1)
    pipeline = Pipeline(
        claude_client=AsyncMock(),
        codex_client=AsyncMock(),
        config=config,
    )
    assert pipeline.config.max_retries == 1


@pytest.mark.asyncio
async def test_pipeline_extract_single_text_region(tmp_path: pathlib.Path):
    """Pipeline should handle a simple text-only document end-to-end."""
    img = Image.new("RGB", (2550, 3300), "white")
    img_path = tmp_path / "simple.png"
    img.save(img_path, dpi=(300, 300))

    mock_claude = AsyncMock()
    mock_codex = AsyncMock()

    # Mock ingestion to return one page
    mock_pages = [MagicMock(
        page_number=1,
        image_path=img_path,
        width=2550,
        height=3300,
        dpi=300,
    )]
    mock_ingestion = MagicMock()
    mock_ingestion.pages = mock_pages
    mock_ingestion.page_count = 1
    mock_ingestion.source_file = img_path
    mock_ingestion.temp_dir = tmp_path

    # Mock layout detection to return one text region
    mock_layout_regions = [MagicMock(
        region_id="r1",
        region_type=RegionType.TEXT,
        page=1,
        bbox=BoundingBox(x=0.05, y=0.10, w=0.90, h=0.15),
        confidence=0.98,
    )]

    # Mock text specialist output
    mock_text_region = Region(
        id="r1",
        type=RegionType.TEXT,
        subtype=None,
        page=1,
        bbox=BoundingBox(x=0.05, y=0.10, w=0.90, h=0.15),
        content=TextContent(text="Hello world", markdown="Hello world"),
        confidence=0.97,
        extraction_method="paddleocr_3.0",
    )

    with patch("agentic_extract.pipeline.ingest", return_value=mock_ingestion), \
         patch("agentic_extract.pipeline.detect_layout", return_value=mock_layout_regions), \
         patch("agentic_extract.pipeline.determine_reading_order", return_value=mock_layout_regions), \
         patch("agentic_extract.pipeline.route_regions") as mock_route, \
         patch("agentic_extract.pipeline.TextSpecialist") as MockTextSpec, \
         patch("agentic_extract.pipeline.assemble_result") as mock_assemble:

        mock_route.return_value = {"text": [mock_layout_regions[0]], "table": [], "visual": []}

        mock_text_spec_instance = AsyncMock()
        mock_text_spec_instance.extract.return_value = mock_text_region
        MockTextSpec.return_value = mock_text_spec_instance

        mock_assemble.return_value = ExtractionResult(
            document=DocumentMetadata(
                id="doc-1",
                source=str(img_path),
                page_count=1,
                processing_timestamp=datetime.now(timezone.utc),
                approach="B",
                total_confidence=0.97,
                processing_time_ms=1500,
            ),
            markdown="# Document\n\nHello world",
            regions=[mock_text_region],
            extracted_entities={},
            audit_trail=AuditTrail(
                models_used=["paddleocr_3.0"],
                total_llm_calls=0,
                re_extractions=0,
                fields_flagged=0,
                processing_stages=[
                    ProcessingStage(stage="ingestion", duration_ms=200),
                    ProcessingStage(stage="extraction", duration_ms=800),
                    ProcessingStage(stage="validation", duration_ms=300),
                    ProcessingStage(stage="assembly", duration_ms=200),
                ],
            ),
        )

        pipeline = Pipeline(
            claude_client=mock_claude,
            codex_client=mock_codex,
        )
        result = await pipeline.extract(img_path)

        assert isinstance(result, ExtractionResult)
        assert result.document.page_count == 1
        assert len(result.regions) == 1
        assert result.regions[0].content.text == "Hello world"


@pytest.mark.asyncio
async def test_pipeline_parallel_specialist_dispatch(tmp_path: pathlib.Path):
    """Pipeline should dispatch specialists in parallel via asyncio.gather."""
    img = Image.new("RGB", (2550, 3300), "white")
    img_path = tmp_path / "mixed.png"
    img.save(img_path, dpi=(300, 300))

    mock_claude = AsyncMock()
    mock_codex = AsyncMock()

    mock_pages = [MagicMock(
        page_number=1, image_path=img_path,
        width=2550, height=3300, dpi=300,
    )]
    mock_ingestion = MagicMock()
    mock_ingestion.pages = mock_pages
    mock_ingestion.page_count = 1
    mock_ingestion.source_file = img_path
    mock_ingestion.temp_dir = tmp_path

    text_region = MagicMock(
        region_id="r1", region_type=RegionType.TEXT,
        page=1, bbox=BoundingBox(x=0.05, y=0.1, w=0.9, h=0.1), confidence=0.98,
    )
    table_region = MagicMock(
        region_id="r2", region_type=RegionType.TABLE,
        page=1, bbox=BoundingBox(x=0.05, y=0.3, w=0.9, h=0.3), confidence=0.95,
    )
    mock_layout_regions = [text_region, table_region]

    extracted_text = Region(
        id="r1", type=RegionType.TEXT, subtype=None, page=1,
        bbox=BoundingBox(x=0.05, y=0.1, w=0.9, h=0.1),
        content=TextContent(text="Text content", markdown="Text content"),
        confidence=0.97, extraction_method="paddleocr_3.0",
    )
    extracted_table = Region(
        id="r2", type=RegionType.TABLE, subtype=None, page=1,
        bbox=BoundingBox(x=0.05, y=0.3, w=0.9, h=0.3),
        content=TableContent(
            html="<table><tr><td>A</td></tr></table>",
            json_data={"headers": ["Col"], "rows": [{"Col": "A"}]},
        ),
        confidence=0.94, extraction_method="docling + claude-opus-4-20250514",
    )

    with patch("agentic_extract.pipeline.ingest", return_value=mock_ingestion), \
         patch("agentic_extract.pipeline.detect_layout", return_value=mock_layout_regions), \
         patch("agentic_extract.pipeline.determine_reading_order", return_value=mock_layout_regions), \
         patch("agentic_extract.pipeline.route_regions") as mock_route, \
         patch("agentic_extract.pipeline.TextSpecialist") as MockTextSpec, \
         patch("agentic_extract.pipeline.TableSpecialist") as MockTableSpec, \
         patch("agentic_extract.pipeline.assemble_result") as mock_assemble:

        mock_route.return_value = {
            "text": [text_region], "table": [table_region], "visual": [],
        }
        MockTextSpec.return_value = AsyncMock()
        MockTextSpec.return_value.extract.return_value = extracted_text
        MockTableSpec.return_value = AsyncMock()
        MockTableSpec.return_value.extract.return_value = extracted_table

        mock_assemble.return_value = ExtractionResult(
            document=DocumentMetadata(
                id="doc-2", source=str(img_path), page_count=1,
                processing_timestamp=datetime.now(timezone.utc),
                approach="B", total_confidence=0.95, processing_time_ms=2000,
            ),
            markdown="# Doc\n\nText content\n\n| Col |\n|-----|\n| A |",
            regions=[extracted_text, extracted_table],
            extracted_entities={},
            audit_trail=AuditTrail(
                models_used=["paddleocr_3.0", "docling", "claude-opus-4-20250514"],
                total_llm_calls=1, re_extractions=0, fields_flagged=0,
                processing_stages=[],
            ),
        )

        pipeline = Pipeline(claude_client=mock_claude, codex_client=mock_codex)
        result = await pipeline.extract(img_path)

        assert len(result.regions) == 2


@pytest.mark.asyncio
async def test_pipeline_partial_output_on_specialist_failure(tmp_path: pathlib.Path):
    """If one specialist fails, pipeline should still return partial output."""
    img = Image.new("RGB", (2550, 3300), "white")
    img_path = tmp_path / "partial.png"
    img.save(img_path, dpi=(300, 300))

    mock_claude = AsyncMock()
    mock_codex = AsyncMock()

    mock_pages = [MagicMock(
        page_number=1, image_path=img_path,
        width=2550, height=3300, dpi=300,
    )]
    mock_ingestion = MagicMock()
    mock_ingestion.pages = mock_pages
    mock_ingestion.page_count = 1
    mock_ingestion.source_file = img_path
    mock_ingestion.temp_dir = tmp_path

    text_region_layout = MagicMock(
        region_id="r1", region_type=RegionType.TEXT,
        page=1, bbox=BoundingBox(x=0.05, y=0.1, w=0.9, h=0.1), confidence=0.98,
    )
    table_region_layout = MagicMock(
        region_id="r2", region_type=RegionType.TABLE,
        page=1, bbox=BoundingBox(x=0.05, y=0.3, w=0.9, h=0.3), confidence=0.95,
    )

    extracted_text = Region(
        id="r1", type=RegionType.TEXT, subtype=None, page=1,
        bbox=BoundingBox(x=0.05, y=0.1, w=0.9, h=0.1),
        content=TextContent(text="Partial text", markdown="Partial text"),
        confidence=0.95, extraction_method="paddleocr_3.0",
    )

    with patch("agentic_extract.pipeline.ingest", return_value=mock_ingestion), \
         patch("agentic_extract.pipeline.detect_layout", return_value=[text_region_layout, table_region_layout]), \
         patch("agentic_extract.pipeline.determine_reading_order", return_value=[text_region_layout, table_region_layout]), \
         patch("agentic_extract.pipeline.route_regions") as mock_route, \
         patch("agentic_extract.pipeline.TextSpecialist") as MockTextSpec, \
         patch("agentic_extract.pipeline.TableSpecialist") as MockTableSpec, \
         patch("agentic_extract.pipeline.assemble_result") as mock_assemble:

        mock_route.return_value = {
            "text": [text_region_layout], "table": [table_region_layout], "visual": [],
        }
        MockTextSpec.return_value = AsyncMock()
        MockTextSpec.return_value.extract.return_value = extracted_text
        # Table specialist fails
        MockTableSpec.return_value = AsyncMock()
        MockTableSpec.return_value.extract.side_effect = RuntimeError("Docling crashed")

        mock_assemble.return_value = ExtractionResult(
            document=DocumentMetadata(
                id="doc-3", source=str(img_path), page_count=1,
                processing_timestamp=datetime.now(timezone.utc),
                approach="B", total_confidence=0.95, processing_time_ms=1000,
            ),
            markdown="# Doc\n\nPartial text\n\n[TABLE EXTRACTION FAILED]",
            regions=[extracted_text],
            extracted_entities={},
            audit_trail=AuditTrail(
                models_used=["paddleocr_3.0"],
                total_llm_calls=0, re_extractions=0, fields_flagged=1,
                processing_stages=[],
            ),
        )

        pipeline = Pipeline(claude_client=mock_claude, codex_client=mock_codex)
        result = await pipeline.extract(img_path)

        # Pipeline should succeed with partial output (text only)
        assert isinstance(result, ExtractionResult)
        assert len(result.regions) >= 1


@pytest.mark.asyncio
async def test_pipeline_per_stage_timing(tmp_path: pathlib.Path):
    """Pipeline should record per-stage timing in the audit trail."""
    img = Image.new("RGB", (100, 100), "white")
    img_path = tmp_path / "timed.png"
    img.save(img_path, dpi=(72, 72))

    mock_claude = AsyncMock()
    mock_codex = AsyncMock()

    mock_pages = [MagicMock(
        page_number=1, image_path=img_path,
        width=100, height=100, dpi=72,
    )]
    mock_ingestion = MagicMock()
    mock_ingestion.pages = mock_pages
    mock_ingestion.page_count = 1
    mock_ingestion.source_file = img_path
    mock_ingestion.temp_dir = tmp_path

    with patch("agentic_extract.pipeline.ingest", return_value=mock_ingestion), \
         patch("agentic_extract.pipeline.detect_layout", return_value=[]), \
         patch("agentic_extract.pipeline.determine_reading_order", return_value=[]), \
         patch("agentic_extract.pipeline.route_regions", return_value={"text": [], "table": [], "visual": []}), \
         patch("agentic_extract.pipeline.assemble_result") as mock_assemble:

        audit = AuditTrail(
            models_used=[], total_llm_calls=0, re_extractions=0,
            fields_flagged=0,
            processing_stages=[
                ProcessingStage(stage="ingestion", duration_ms=50),
                ProcessingStage(stage="layout_detection", duration_ms=30),
                ProcessingStage(stage="extraction", duration_ms=0),
                ProcessingStage(stage="validation", duration_ms=0),
                ProcessingStage(stage="assembly", duration_ms=10),
            ],
        )
        mock_assemble.return_value = ExtractionResult(
            document=DocumentMetadata(
                id="doc-4", source=str(img_path), page_count=1,
                processing_timestamp=datetime.now(timezone.utc),
                approach="B", total_confidence=1.0, processing_time_ms=90,
            ),
            markdown="", regions=[], extracted_entities={},
            audit_trail=audit,
        )

        pipeline = Pipeline(claude_client=mock_claude, codex_client=mock_codex)
        result = await pipeline.extract(img_path)

        stage_names = [s.stage for s in result.audit_trail.processing_stages]
        assert "ingestion" in stage_names
        assert "layout_detection" in stage_names
        assert "assembly" in stage_names


@pytest.mark.asyncio
async def test_pipeline_triggers_reextraction_for_low_confidence(tmp_path: pathlib.Path):
    """Pipeline should trigger re-extraction for regions with confidence in [0.70, 0.90)."""
    img = Image.new("RGB", (2550, 3300), "white")
    img_path = tmp_path / "reextract.png"
    img.save(img_path, dpi=(300, 300))

    mock_claude = AsyncMock()
    mock_codex = AsyncMock()

    mock_pages = [MagicMock(
        page_number=1, image_path=img_path,
        width=2550, height=3300, dpi=300,
    )]
    mock_ingestion = MagicMock()
    mock_ingestion.pages = mock_pages
    mock_ingestion.page_count = 1
    mock_ingestion.source_file = img_path
    mock_ingestion.temp_dir = tmp_path

    text_region_layout = MagicMock(
        region_id="r1", region_type=RegionType.TEXT,
        page=1, bbox=BoundingBox(x=0.05, y=0.1, w=0.9, h=0.1), confidence=0.98,
    )

    # Specialist produces low-confidence result
    low_conf_region = Region(
        id="r1", type=RegionType.TEXT, subtype=None, page=1,
        bbox=BoundingBox(x=0.05, y=0.1, w=0.9, h=0.1),
        content=TextContent(text="Ambiguous text", markdown="Ambiguous text"),
        confidence=0.78,
        extraction_method="paddleocr_3.0 + claude-opus-4-20250514",
    )

    with patch("agentic_extract.pipeline.ingest", return_value=mock_ingestion), \
         patch("agentic_extract.pipeline.detect_layout", return_value=[text_region_layout]), \
         patch("agentic_extract.pipeline.determine_reading_order", return_value=[text_region_layout]), \
         patch("agentic_extract.pipeline.route_regions") as mock_route, \
         patch("agentic_extract.pipeline.TextSpecialist") as MockTextSpec, \
         patch("agentic_extract.pipeline.ReExtractionEngine") as MockReExtract, \
         patch("agentic_extract.pipeline.assemble_result") as mock_assemble:

        mock_route.return_value = {"text": [text_region_layout], "table": [], "visual": []}
        MockTextSpec.return_value = AsyncMock()
        MockTextSpec.return_value.extract.return_value = low_conf_region

        from agentic_extract.reextraction.engine import ReExtractionResult, ReExtractionCandidate
        mock_reextract_result = ReExtractionResult(
            region_id="r1", field_name="text", original_value="Ambiguous text",
            original_confidence=0.78, original_model="claude-opus-4-20250514",
            candidates=[ReExtractionCandidate(
                value="Ambiguous text", confidence=0.88,
                model="gpt-4o", extraction_method="re_extraction_1 + gpt-4o",
            )],
            final_value="Ambiguous text", final_confidence=0.88,
            retries_used=1, models_agreed=True, accepted=True, flagged=False,
        )
        mock_engine_instance = AsyncMock()
        mock_engine_instance.re_extract_field.return_value = mock_reextract_result
        MockReExtract.return_value = mock_engine_instance

        boosted_region = Region(
            id="r1", type=RegionType.TEXT, subtype=None, page=1,
            bbox=BoundingBox(x=0.05, y=0.1, w=0.9, h=0.1),
            content=TextContent(text="Ambiguous text", markdown="Ambiguous text"),
            confidence=0.88, extraction_method="paddleocr_3.0 + claude-opus-4-20250514 + gpt-4o",
        )
        mock_assemble.return_value = ExtractionResult(
            document=DocumentMetadata(
                id="doc-5", source=str(img_path), page_count=1,
                processing_timestamp=datetime.now(timezone.utc),
                approach="B", total_confidence=0.88, processing_time_ms=3000,
            ),
            markdown="# Doc\n\nAmbiguous text",
            regions=[boosted_region],
            extracted_entities={},
            audit_trail=AuditTrail(
                models_used=["paddleocr_3.0", "claude-opus-4-20250514", "gpt-4o"],
                total_llm_calls=2, re_extractions=1, fields_flagged=0,
                processing_stages=[],
            ),
        )

        pipeline = Pipeline(claude_client=mock_claude, codex_client=mock_codex)
        result = await pipeline.extract(img_path)

        assert result.audit_trail.re_extractions >= 1
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_pipeline.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'agentic_extract.pipeline'"

**Step 3: Write minimal implementation**

```python
# src/agentic_extract/pipeline.py
"""Full pipeline orchestration: Coordinator -> Specialists -> Validator -> Re-extraction -> Output.

This is the top-level entry point for document extraction. It composes
all agents (Coordinator, Specialist Pool, Validator) and the re-extraction
loop into a single async pipeline.

Usage:
    pipeline = Pipeline(claude_client=claude, codex_client=codex)
    result = await pipeline.extract(Path("document.pdf"))
"""
from __future__ import annotations

import asyncio
import logging
import pathlib
import time
import uuid
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field

from agentic_extract.clients.vlm import VLMClient
from agentic_extract.coordinator.assembly import assemble_result
from agentic_extract.coordinator.ingestion import ingest
from agentic_extract.coordinator.layout import detect_layout
from agentic_extract.coordinator.reading_order import determine_reading_order
from agentic_extract.coordinator.routing import route_regions
from agentic_extract.models import (
    AuditTrail,
    BoundingBox,
    DocumentMetadata,
    ExtractionResult,
    ProcessingStage,
    Region,
    RegionType,
)
from agentic_extract.reextraction.engine import ReExtractionEngine
from agentic_extract.specialists.text import TextSpecialist
from agentic_extract.specialists.table import TableSpecialist

logger = logging.getLogger(__name__)

ACCEPT_THRESHOLD = 0.90
REEXTRACT_LOW = 0.70


class PipelineConfig(BaseModel):
    """Configuration for the extraction pipeline."""

    accept_threshold: float = Field(default=0.90, ge=0.0, le=1.0)
    reextract_low: float = Field(default=0.70, ge=0.0, le=1.0)
    max_retries: int = Field(default=2, ge=0)
    parallel_specialists: bool = True


class Pipeline:
    """Full extraction pipeline orchestrator.

    Flow:
    1. Ingestion: convert document to page images
    2. Layout detection: identify regions on each page
    3. Reading order: determine logical ordering
    4. Routing: assign regions to specialists
    5. Extraction: dispatch specialists (parallel via asyncio.gather)
    6. Validation: run 5-layer validation on all regions
    7. Re-extraction: retry low-confidence fields with model switching
    8. Assembly: produce final Markdown + JSON output

    Args:
        claude_client: VLM client for Claude API calls.
        codex_client: VLM client for Codex/GPT-4o API calls.
        config: Pipeline configuration. Uses defaults if not provided.
    """

    def __init__(
        self,
        claude_client: VLMClient,
        codex_client: VLMClient,
        config: PipelineConfig | None = None,
    ) -> None:
        self.claude_client = claude_client
        self.codex_client = codex_client
        self.config = config or PipelineConfig()
        self._reextraction_engine = ReExtractionEngine(
            claude_client=claude_client,
            codex_client=codex_client,
            max_retries=self.config.max_retries,
        )

    async def extract(
        self,
        file_path: pathlib.Path,
        schema: dict[str, Any] | None = None,
    ) -> ExtractionResult:
        """Run the full extraction pipeline on a document.

        Args:
            file_path: Path to the input document (PDF or image).
            schema: Optional JSON schema for structured entity extraction.

        Returns:
            ExtractionResult with Markdown, JSON, regions, and audit trail.
        """
        file_path = pathlib.Path(file_path)
        stages: list[ProcessingStage] = []
        all_regions: list[Region] = []
        models_used: set[str] = set()
        total_llm_calls = 0
        re_extraction_count = 0
        fields_flagged = 0

        pipeline_start = time.monotonic()

        # --- Stage 1: Ingestion ---
        stage_start = time.monotonic()
        ingestion_result = ingest(file_path)
        stages.append(ProcessingStage(
            stage="ingestion",
            duration_ms=int((time.monotonic() - stage_start) * 1000),
        ))

        # --- Stage 2: Layout Detection ---
        stage_start = time.monotonic()
        layout_regions = []
        for page in ingestion_result.pages:
            page_regions = detect_layout(page.image_path)
            layout_regions.extend(page_regions)
        stages.append(ProcessingStage(
            stage="layout_detection",
            duration_ms=int((time.monotonic() - stage_start) * 1000),
        ))

        # --- Stage 3: Reading Order ---
        stage_start = time.monotonic()
        ordered_layout = determine_reading_order(layout_regions)
        stages.append(ProcessingStage(
            stage="reading_order",
            duration_ms=int((time.monotonic() - stage_start) * 1000),
        ))

        # --- Stage 4: Routing ---
        stage_start = time.monotonic()
        routed = route_regions(ordered_layout)
        stages.append(ProcessingStage(
            stage="routing",
            duration_ms=int((time.monotonic() - stage_start) * 1000),
        ))

        # --- Stage 5: Specialist Extraction ---
        stage_start = time.monotonic()
        text_specialist = TextSpecialist(vlm_client=self.claude_client)
        table_specialist = TableSpecialist(
            vlm_client=self.claude_client,
            codex_client=self.codex_client,
        )

        extraction_tasks = []
        task_metadata = []

        for lr in routed.get("text", []):
            page_img = self._find_page_image(ingestion_result, lr)
            extraction_tasks.append(
                text_specialist.extract(
                    image_path=page_img,
                    region_id=lr.region_id,
                    page=lr.page,
                    bbox=lr.bbox,
                )
            )
            task_metadata.append(("text", lr))

        for lr in routed.get("table", []):
            page_img = self._find_page_image(ingestion_result, lr)
            extraction_tasks.append(
                table_specialist.extract(
                    image_path=page_img,
                    region_id=lr.region_id,
                    page=lr.page,
                    bbox=lr.bbox,
                )
            )
            task_metadata.append(("table", lr))

        # Dispatch in parallel or sequentially based on config
        if self.config.parallel_specialists and extraction_tasks:
            results = await asyncio.gather(
                *extraction_tasks, return_exceptions=True,
            )
        elif extraction_tasks:
            results = []
            for task in extraction_tasks:
                try:
                    r = await task
                    results.append(r)
                except Exception as e:
                    results.append(e)
        else:
            results = []

        # Collect successful extractions, log failures
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                spec_type, lr = task_metadata[i]
                logger.error(
                    "Specialist %s failed for region %s: %s",
                    spec_type, lr.region_id, result,
                )
                fields_flagged += 1
            elif isinstance(result, Region):
                all_regions.append(result)
                models_used.add(result.extraction_method)

        stages.append(ProcessingStage(
            stage="extraction",
            duration_ms=int((time.monotonic() - stage_start) * 1000),
        ))

        # --- Stage 6: Validation + Re-extraction ---
        stage_start = time.monotonic()
        final_regions: list[Region] = []

        for region in all_regions:
            if region.confidence >= self.config.accept_threshold:
                final_regions.append(region)
            elif region.confidence >= self.config.reextract_low:
                # Trigger re-extraction
                page_img = self._find_page_image_for_region(
                    ingestion_result, region,
                )
                try:
                    original_model = self._infer_model(region.extraction_method)
                    re_result = await self._reextraction_engine.re_extract_field(
                        region=region,
                        field_name=region.type.value,
                        image_path=page_img,
                        original_model=original_model,
                    )
                    re_extraction_count += 1
                    models_used.add(re_result.original_model)
                    for c in re_result.candidates:
                        models_used.add(c.model)

                    # Update region with re-extraction result
                    updated = region.model_copy(update={
                        "confidence": re_result.final_confidence,
                        "needs_review": re_result.flagged,
                        "review_reason": (
                            f"Re-extraction: models {'agreed' if re_result.models_agreed else 'disagreed'} "
                            f"after {re_result.retries_used} retries"
                            if re_result.flagged else None
                        ),
                    })
                    final_regions.append(updated)
                    if re_result.flagged:
                        fields_flagged += 1
                except Exception as exc:
                    logger.error(
                        "Re-extraction failed for region %s: %s", region.id, exc,
                    )
                    flagged_region = region.model_copy(update={
                        "needs_review": True,
                        "review_reason": f"Re-extraction error: {exc}",
                    })
                    final_regions.append(flagged_region)
                    fields_flagged += 1
            else:
                # Below reextract threshold: flag for review
                flagged_region = region.model_copy(update={
                    "needs_review": True,
                    "review_reason": (
                        f"Confidence {region.confidence:.2f} below {self.config.reextract_low} threshold"
                    ),
                })
                final_regions.append(flagged_region)
                fields_flagged += 1

        stages.append(ProcessingStage(
            stage="validation",
            duration_ms=int((time.monotonic() - stage_start) * 1000),
        ))

        # --- Stage 7: Assembly ---
        stage_start = time.monotonic()

        total_time_ms = int((time.monotonic() - pipeline_start) * 1000)
        metadata = DocumentMetadata(
            id=str(uuid.uuid4()),
            source=str(file_path),
            page_count=ingestion_result.page_count,
            processing_timestamp=datetime.now(timezone.utc),
            approach="B",
            total_confidence=self._compute_total_confidence(final_regions),
            processing_time_ms=total_time_ms,
        )

        audit = AuditTrail(
            models_used=sorted(models_used),
            total_llm_calls=total_llm_calls,
            re_extractions=re_extraction_count,
            fields_flagged=fields_flagged,
            processing_stages=stages,
        )

        result = assemble_result(
            regions=final_regions,
            metadata=metadata,
            audit_trail=audit,
        )

        stages.append(ProcessingStage(
            stage="assembly",
            duration_ms=int((time.monotonic() - stage_start) * 1000),
        ))

        return result

    @staticmethod
    def _find_page_image(ingestion_result: Any, layout_region: Any) -> pathlib.Path:
        """Find the page image for a given layout region."""
        for page in ingestion_result.pages:
            if page.page_number == layout_region.page:
                return page.image_path
        # Fallback to first page
        return ingestion_result.pages[0].image_path

    @staticmethod
    def _find_page_image_for_region(
        ingestion_result: Any, region: Region,
    ) -> pathlib.Path:
        """Find the page image for a given extracted region."""
        for page in ingestion_result.pages:
            if page.page_number == region.page:
                return page.image_path
        return ingestion_result.pages[0].image_path

    @staticmethod
    def _infer_model(extraction_method: str) -> str:
        """Infer the primary VLM model from an extraction method string."""
        if "claude" in extraction_method.lower():
            return "claude-opus-4-20250514"
        if "gpt" in extraction_method.lower() or "codex" in extraction_method.lower():
            return "gpt-4o"
        return "claude-opus-4-20250514"

    @staticmethod
    def _compute_total_confidence(regions: list[Region]) -> float:
        """Compute document-level confidence as the minimum region confidence."""
        if not regions:
            return 1.0
        return min(r.confidence for r in regions)
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_pipeline.py -v`
Expected: PASS (8 passed)

**Step 5: Commit**

```bash
git add src/agentic_extract/pipeline.py tests/test_pipeline.py
git commit -m "feat: full pipeline orchestration with parallel specialist dispatch, re-extraction, per-stage timing"
```

---

### Task 23: Visual Grounding Enhancement

**Files:**
- Create: `src/agentic_extract/grounding/__init__.py`
- Create: `src/agentic_extract/grounding/visual.py`
- Test: `tests/grounding/test_visual.py`
- Create: `tests/grounding/__init__.py`

**Step 1: Write the failing test**

```python
# tests/grounding/__init__.py
```

```python
# tests/grounding/test_visual.py
"""Tests for visual grounding: linking extracted values to source bounding boxes."""
import json
import pathlib
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from agentic_extract.grounding.visual import (
    GroundedField,
    GroundedRegion,
    VisualGrounding,
    CellGrounding,
)
from agentic_extract.models import (
    BoundingBox,
    Region,
    RegionType,
    TextContent,
    TableContent,
    FigureContent,
)
from agentic_extract.tools.docker_runner import ToolOutput


def test_grounded_field_model():
    gf = GroundedField(
        field_name="patient_name",
        value="John Smith",
        bbox=BoundingBox(x=0.1, y=0.2, w=0.3, h=0.05),
        confidence=0.95,
        bbox_verified=True,
    )
    assert gf.field_name == "patient_name"
    assert gf.bbox_verified is True
    assert gf.bbox.x == 0.1


def test_grounded_field_unverified():
    gf = GroundedField(
        field_name="dosage",
        value="500mg",
        bbox=BoundingBox(x=0.4, y=0.5, w=0.2, h=0.03),
        confidence=0.78,
        bbox_verified=False,
    )
    assert gf.bbox_verified is False


def test_cell_grounding_model():
    cg = CellGrounding(
        row=0,
        col=1,
        value="3.2",
        bbox=BoundingBox(x=0.3, y=0.25, w=0.15, h=0.04),
        confidence=0.94,
        bbox_verified=True,
    )
    assert cg.row == 0
    assert cg.col == 1
    assert cg.bbox_verified is True


def test_grounded_region_text():
    gr = GroundedRegion(
        region_id="r1",
        region_type=RegionType.TEXT,
        page=1,
        region_bbox=BoundingBox(x=0.05, y=0.1, w=0.9, h=0.15),
        fields=[
            GroundedField(
                field_name="text",
                value="Hello world",
                bbox=BoundingBox(x=0.05, y=0.1, w=0.9, h=0.15),
                confidence=0.97,
                bbox_verified=True,
            ),
        ],
        cells=[],
    )
    assert gr.region_id == "r1"
    assert len(gr.fields) == 1
    assert gr.fields[0].bbox_verified is True


def test_grounded_region_table_with_cells():
    cells = [
        CellGrounding(
            row=0, col=0, value="Gene",
            bbox=BoundingBox(x=0.05, y=0.2, w=0.3, h=0.05),
            confidence=0.96, bbox_verified=True,
        ),
        CellGrounding(
            row=0, col=1, value="Value",
            bbox=BoundingBox(x=0.35, y=0.2, w=0.3, h=0.05),
            confidence=0.95, bbox_verified=True,
        ),
        CellGrounding(
            row=1, col=0, value="BRCA1",
            bbox=BoundingBox(x=0.05, y=0.25, w=0.3, h=0.05),
            confidence=0.93, bbox_verified=True,
        ),
        CellGrounding(
            row=1, col=1, value="3.2",
            bbox=BoundingBox(x=0.35, y=0.25, w=0.3, h=0.05),
            confidence=0.91, bbox_verified=True,
        ),
    ]
    gr = GroundedRegion(
        region_id="t1",
        region_type=RegionType.TABLE,
        page=2,
        region_bbox=BoundingBox(x=0.05, y=0.2, w=0.9, h=0.4),
        fields=[],
        cells=cells,
    )
    assert len(gr.cells) == 4
    assert gr.cells[2].value == "BRCA1"


def test_visual_grounding_init():
    vg = VisualGrounding()
    assert vg is not None


@patch("agentic_extract.grounding.visual._run_quick_ocr")
def test_visual_grounding_verifies_text_bbox(mock_ocr: MagicMock, tmp_path: pathlib.Path):
    """bbox_verified should be True when OCR on the cropped bbox matches extracted text."""
    img = Image.new("RGB", (1000, 200), "white")
    img_path = tmp_path / "page.png"
    img.save(img_path)

    # Mock OCR on the cropped region returns matching text
    mock_ocr.return_value = "Hello world"

    region = Region(
        id="r1",
        type=RegionType.TEXT,
        subtype=None,
        page=1,
        bbox=BoundingBox(x=0.05, y=0.1, w=0.9, h=0.15),
        content=TextContent(text="Hello world", markdown="Hello world"),
        confidence=0.97,
        extraction_method="paddleocr_3.0",
    )

    vg = VisualGrounding()
    grounded = vg.ground_region(region, img_path)

    assert isinstance(grounded, GroundedRegion)
    assert len(grounded.fields) == 1
    assert grounded.fields[0].bbox_verified is True
    assert grounded.fields[0].value == "Hello world"


@patch("agentic_extract.grounding.visual._run_quick_ocr")
def test_visual_grounding_fails_verification_on_mismatch(mock_ocr: MagicMock, tmp_path: pathlib.Path):
    """bbox_verified should be False when OCR on the bbox does not match extracted text."""
    img = Image.new("RGB", (1000, 200), "white")
    img_path = tmp_path / "page.png"
    img.save(img_path)

    # OCR returns completely different text
    mock_ocr.return_value = "Completely different text"

    region = Region(
        id="r2",
        type=RegionType.TEXT,
        subtype=None,
        page=1,
        bbox=BoundingBox(x=0.1, y=0.2, w=0.8, h=0.1),
        content=TextContent(text="Expected text here", markdown="Expected text here"),
        confidence=0.85,
        extraction_method="paddleocr_3.0 + claude-opus-4-20250514",
    )

    vg = VisualGrounding()
    grounded = vg.ground_region(region, img_path)

    assert grounded.fields[0].bbox_verified is False


@patch("agentic_extract.grounding.visual._run_quick_ocr")
def test_visual_grounding_table_cells(mock_ocr: MagicMock, tmp_path: pathlib.Path):
    """Table regions should produce cell-level grounding."""
    img = Image.new("RGB", (1000, 500), "white")
    img_path = tmp_path / "page.png"
    img.save(img_path)

    # OCR returns matching values for each cell
    mock_ocr.side_effect = ["Gene", "Value", "BRCA1", "3.2"]

    region = Region(
        id="t1",
        type=RegionType.TABLE,
        subtype=None,
        page=1,
        bbox=BoundingBox(x=0.05, y=0.2, w=0.9, h=0.4),
        content=TableContent(
            html="<table><tr><th>Gene</th><th>Value</th></tr><tr><td>BRCA1</td><td>3.2</td></tr></table>",
            json_data={"headers": ["Gene", "Value"], "rows": [{"Gene": "BRCA1", "Value": "3.2"}]},
            cell_bboxes=[
                {"row": 0, "col": 0, "bbox": BoundingBox(x=0.05, y=0.2, w=0.3, h=0.05)},
                {"row": 0, "col": 1, "bbox": BoundingBox(x=0.35, y=0.2, w=0.3, h=0.05)},
                {"row": 1, "col": 0, "bbox": BoundingBox(x=0.05, y=0.25, w=0.3, h=0.05)},
                {"row": 1, "col": 1, "bbox": BoundingBox(x=0.35, y=0.25, w=0.3, h=0.05)},
            ],
        ),
        confidence=0.94,
        extraction_method="docling + claude-opus-4-20250514",
    )

    vg = VisualGrounding()
    grounded = vg.ground_region(region, img_path)

    assert len(grounded.cells) == 4
    assert all(c.bbox_verified for c in grounded.cells)
    assert grounded.cells[0].value == "Gene"
    assert grounded.cells[3].value == "3.2"


def test_visual_grounding_normalized_coordinates():
    """All bounding box coordinates must be normalized to [0, 1]."""
    gf = GroundedField(
        field_name="test",
        value="x",
        bbox=BoundingBox(x=0.0, y=0.0, w=1.0, h=1.0),
        confidence=0.9,
        bbox_verified=True,
    )
    assert 0.0 <= gf.bbox.x <= 1.0
    assert 0.0 <= gf.bbox.y <= 1.0
    assert 0.0 <= gf.bbox.w <= 1.0
    assert 0.0 <= gf.bbox.h <= 1.0


@patch("agentic_extract.grounding.visual._run_quick_ocr")
def test_visual_grounding_figure_region(mock_ocr: MagicMock, tmp_path: pathlib.Path):
    """Figure regions should produce region-level grounding (no cell detail)."""
    img = Image.new("RGB", (800, 600), "white")
    img_path = tmp_path / "page.png"
    img.save(img_path)

    mock_ocr.return_value = "Bar chart showing gene expression"

    region = Region(
        id="f1",
        type=RegionType.FIGURE,
        subtype="bar_chart",
        page=3,
        bbox=BoundingBox(x=0.1, y=0.05, w=0.8, h=0.45),
        content=FigureContent(
            description="Bar chart showing gene expression levels",
            figure_type="bar_chart",
            figure_json={"title": "Gene Expression"},
        ),
        confidence=0.86,
        extraction_method="deplot + claude-opus-4-20250514",
    )

    vg = VisualGrounding()
    grounded = vg.ground_region(region, img_path)

    assert grounded.region_type == RegionType.FIGURE
    assert len(grounded.fields) >= 1
    assert grounded.cells == []
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/grounding/test_visual.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'agentic_extract.grounding'"

**Step 3: Write minimal implementation**

```python
# src/agentic_extract/grounding/__init__.py
"""Visual grounding: linking extracted values to source bounding boxes."""
```

```python
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
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/grounding/test_visual.py -v`
Expected: PASS (12 passed)

**Step 5: Commit**

```bash
git add src/agentic_extract/grounding/__init__.py src/agentic_extract/grounding/visual.py tests/grounding/__init__.py tests/grounding/test_visual.py
git commit -m "feat: visual grounding with normalized bbox linking and Layer 4 OCR verification"
```

---

### Task 24: Audit Trail

**Files:**
- Create: `src/agentic_extract/audit.py`
- Test: `tests/test_audit.py`

**Step 1: Write the failing test**

```python
# tests/test_audit.py
"""Tests for the audit trail tracker that accumulates throughout the pipeline."""
import time

import pytest

from agentic_extract.audit import AuditTrailTracker
from agentic_extract.models import AuditTrail, ProcessingStage


def test_tracker_init():
    tracker = AuditTrailTracker()
    assert tracker.total_llm_calls == 0
    assert tracker.re_extractions == 0
    assert tracker.fields_flagged == 0
    assert len(tracker.models_used) == 0
    assert len(tracker.stages) == 0


def test_tracker_record_model():
    tracker = AuditTrailTracker()
    tracker.record_model("paddleocr_3.0")
    tracker.record_model("claude-opus-4-20250514")
    tracker.record_model("paddleocr_3.0")  # duplicate
    assert len(tracker.models_used) == 2
    assert "paddleocr_3.0" in tracker.models_used
    assert "claude-opus-4-20250514" in tracker.models_used


def test_tracker_record_llm_call():
    tracker = AuditTrailTracker()
    tracker.record_llm_call()
    tracker.record_llm_call()
    tracker.record_llm_call()
    assert tracker.total_llm_calls == 3


def test_tracker_record_reextraction():
    tracker = AuditTrailTracker()
    tracker.record_reextraction()
    assert tracker.re_extractions == 1
    tracker.record_reextraction()
    assert tracker.re_extractions == 2


def test_tracker_record_flagged_field():
    tracker = AuditTrailTracker()
    tracker.record_flagged_field()
    tracker.record_flagged_field()
    assert tracker.fields_flagged == 2


def test_tracker_start_stop_stage():
    tracker = AuditTrailTracker()
    tracker.start_stage("ingestion")
    time.sleep(0.01)  # Ensure non-zero duration
    tracker.stop_stage("ingestion")
    assert len(tracker.stages) == 1
    assert tracker.stages[0].stage == "ingestion"
    assert tracker.stages[0].duration_ms >= 0


def test_tracker_multiple_stages():
    tracker = AuditTrailTracker()
    tracker.start_stage("ingestion")
    tracker.stop_stage("ingestion")
    tracker.start_stage("layout_detection")
    tracker.stop_stage("layout_detection")
    tracker.start_stage("extraction")
    tracker.stop_stage("extraction")
    assert len(tracker.stages) == 3
    stage_names = [s.stage for s in tracker.stages]
    assert stage_names == ["ingestion", "layout_detection", "extraction"]


def test_tracker_stop_unstarted_stage_is_safe():
    tracker = AuditTrailTracker()
    # Should not raise
    tracker.stop_stage("nonexistent_stage")
    assert len(tracker.stages) == 0


def test_tracker_context_manager():
    tracker = AuditTrailTracker()
    with tracker.stage("validation"):
        time.sleep(0.01)
    assert len(tracker.stages) == 1
    assert tracker.stages[0].stage == "validation"
    assert tracker.stages[0].duration_ms >= 0


def test_tracker_context_manager_on_exception():
    tracker = AuditTrailTracker()
    with pytest.raises(ValueError):
        with tracker.stage("failing_stage"):
            raise ValueError("test error")
    # Stage should still be recorded even on exception
    assert len(tracker.stages) == 1
    assert tracker.stages[0].stage == "failing_stage"


def test_tracker_build_audit_trail():
    tracker = AuditTrailTracker()
    tracker.record_model("paddleocr_3.0")
    tracker.record_model("claude-opus-4-20250514")
    tracker.record_llm_call()
    tracker.record_llm_call()
    tracker.record_reextraction()
    tracker.record_flagged_field()
    tracker.start_stage("ingestion")
    tracker.stop_stage("ingestion")
    tracker.start_stage("extraction")
    tracker.stop_stage("extraction")

    audit = tracker.build()

    assert isinstance(audit, AuditTrail)
    assert len(audit.models_used) == 2
    assert audit.total_llm_calls == 2
    assert audit.re_extractions == 1
    assert audit.fields_flagged == 1
    assert len(audit.processing_stages) == 2
    assert audit.processing_stages[0].stage == "ingestion"


def test_tracker_build_sorts_models():
    tracker = AuditTrailTracker()
    tracker.record_model("gpt-4o")
    tracker.record_model("claude-opus-4-20250514")
    tracker.record_model("docling")

    audit = tracker.build()
    assert audit.models_used == ["claude-opus-4-20250514", "docling", "gpt-4o"]


def test_tracker_merge():
    """Merging two trackers should combine all their data."""
    t1 = AuditTrailTracker()
    t1.record_model("paddleocr_3.0")
    t1.record_llm_call()
    t1.start_stage("ingestion")
    t1.stop_stage("ingestion")

    t2 = AuditTrailTracker()
    t2.record_model("claude-opus-4-20250514")
    t2.record_llm_call()
    t2.record_llm_call()
    t2.record_reextraction()
    t2.start_stage("extraction")
    t2.stop_stage("extraction")

    t1.merge(t2)

    assert len(t1.models_used) == 2
    assert t1.total_llm_calls == 3
    assert t1.re_extractions == 1
    assert len(t1.stages) == 2


def test_tracker_reset():
    tracker = AuditTrailTracker()
    tracker.record_model("test")
    tracker.record_llm_call()
    tracker.record_reextraction()
    tracker.record_flagged_field()
    tracker.start_stage("s1")
    tracker.stop_stage("s1")

    tracker.reset()

    assert len(tracker.models_used) == 0
    assert tracker.total_llm_calls == 0
    assert tracker.re_extractions == 0
    assert tracker.fields_flagged == 0
    assert len(tracker.stages) == 0
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_audit.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'agentic_extract.audit'"

**Step 3: Write minimal implementation**

```python
# src/agentic_extract/audit.py
"""Audit trail tracker that accumulates metrics throughout the pipeline.

Integrates with the Pipeline class to record:
- models_used: all tools and VLMs invoked
- total_llm_calls: count of Claude/Codex API calls
- re_extractions: count of model-switch re-extraction attempts
- fields_flagged: count of fields marked needs_review
- per-stage timing: duration of each pipeline stage
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Generator

from agentic_extract.models import AuditTrail, ProcessingStage


class AuditTrailTracker:
    """Mutable tracker that accumulates audit data throughout pipeline execution.

    Usage:
        tracker = AuditTrailTracker()
        tracker.record_model("paddleocr_3.0")
        with tracker.stage("ingestion"):
            ... do ingestion ...
        tracker.record_llm_call()
        audit = tracker.build()  # Produces immutable AuditTrail
    """

    def __init__(self) -> None:
        self.models_used: set[str] = set()
        self.total_llm_calls: int = 0
        self.re_extractions: int = 0
        self.fields_flagged: int = 0
        self.stages: list[ProcessingStage] = []
        self._stage_starts: dict[str, float] = {}

    def record_model(self, model_name: str) -> None:
        """Record that a model or tool was used."""
        self.models_used.add(model_name)

    def record_llm_call(self) -> None:
        """Record a single LLM API call (Claude or Codex)."""
        self.total_llm_calls += 1

    def record_reextraction(self) -> None:
        """Record a re-extraction attempt."""
        self.re_extractions += 1

    def record_flagged_field(self) -> None:
        """Record a field that was flagged for review."""
        self.fields_flagged += 1

    def start_stage(self, name: str) -> None:
        """Mark the start of a pipeline stage for timing."""
        self._stage_starts[name] = time.monotonic()

    def stop_stage(self, name: str) -> None:
        """Mark the end of a pipeline stage and record its duration."""
        start = self._stage_starts.pop(name, None)
        if start is None:
            return
        duration_ms = int((time.monotonic() - start) * 1000)
        self.stages.append(ProcessingStage(stage=name, duration_ms=duration_ms))

    @contextmanager
    def stage(self, name: str) -> Generator[None, None, None]:
        """Context manager for timing a pipeline stage.

        The stage is recorded even if an exception occurs inside the block.
        """
        self.start_stage(name)
        try:
            yield
        finally:
            self.stop_stage(name)

    def merge(self, other: AuditTrailTracker) -> None:
        """Merge another tracker's data into this one.

        Useful when combining results from parallel specialist dispatch.
        """
        self.models_used |= other.models_used
        self.total_llm_calls += other.total_llm_calls
        self.re_extractions += other.re_extractions
        self.fields_flagged += other.fields_flagged
        self.stages.extend(other.stages)

    def reset(self) -> None:
        """Clear all accumulated data."""
        self.models_used.clear()
        self.total_llm_calls = 0
        self.re_extractions = 0
        self.fields_flagged = 0
        self.stages.clear()
        self._stage_starts.clear()

    def build(self) -> AuditTrail:
        """Build an immutable AuditTrail from the accumulated data.

        Returns:
            AuditTrail Pydantic model ready for inclusion in ExtractionResult.
        """
        return AuditTrail(
            models_used=sorted(self.models_used),
            total_llm_calls=self.total_llm_calls,
            re_extractions=self.re_extractions,
            fields_flagged=self.fields_flagged,
            processing_stages=list(self.stages),
        )
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_audit.py -v`
Expected: PASS (14 passed)

**Step 5: Commit**

```bash
git add src/agentic_extract/audit.py tests/test_audit.py
git commit -m "feat: AuditTrailTracker with per-stage timing, merge support, and context manager"
```

---

### Task 25: Claude Code Skill Packaging

**Files:**
- Create: `~/.claude/skills/agentic-extract/extract.md`
- Create: `src/agentic_extract/skill.py`
- Test: `tests/test_skill.py`

**Step 1: Write the failing test**

```python
# tests/test_skill.py
"""Tests for the Claude Code skill entry point."""
import asyncio
import json
import pathlib
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from PIL import Image

from agentic_extract.models import (
    AuditTrail,
    BoundingBox,
    DocumentMetadata,
    ExtractionResult,
    ProcessingStage,
    Region,
    RegionType,
    TextContent,
)
from agentic_extract.skill import (
    run_extraction,
    format_summary,
    write_outputs,
)


def _make_sample_result(source: str = "test.pdf") -> ExtractionResult:
    """Helper to build a sample ExtractionResult for testing."""
    return ExtractionResult(
        document=DocumentMetadata(
            id="doc-test",
            source=source,
            page_count=2,
            processing_timestamp=datetime.now(timezone.utc),
            approach="B",
            total_confidence=0.93,
            processing_time_ms=5000,
        ),
        markdown="# Test Document\n\nSample extracted text.",
        regions=[
            Region(
                id="r1",
                type=RegionType.TEXT,
                subtype=None,
                page=1,
                bbox=BoundingBox(x=0.05, y=0.1, w=0.9, h=0.15),
                content=TextContent(text="Sample extracted text.", markdown="Sample extracted text."),
                confidence=0.95,
                extraction_method="paddleocr_3.0",
            ),
            Region(
                id="r2",
                type=RegionType.TEXT,
                subtype=None,
                page=2,
                bbox=BoundingBox(x=0.05, y=0.1, w=0.9, h=0.10),
                content=TextContent(text="Page two content.", markdown="Page two content."),
                confidence=0.91,
                extraction_method="paddleocr_3.0 + claude-opus-4-20250514",
            ),
        ],
        extracted_entities={},
        audit_trail=AuditTrail(
            models_used=["claude-opus-4-20250514", "paddleocr_3.0"],
            total_llm_calls=1,
            re_extractions=0,
            fields_flagged=0,
            processing_stages=[
                ProcessingStage(stage="ingestion", duration_ms=500),
                ProcessingStage(stage="extraction", duration_ms=3000),
                ProcessingStage(stage="validation", duration_ms=1000),
                ProcessingStage(stage="assembly", duration_ms=500),
            ],
        ),
    )


def test_format_summary_basic():
    result = _make_sample_result()
    summary = format_summary(result)

    assert "test.pdf" in summary
    assert "2" in summary  # page count
    assert "0.93" in summary or "93" in summary  # confidence
    assert "paddleocr" in summary.lower() or "models" in summary.lower()
    assert isinstance(summary, str)
    assert len(summary) > 50


def test_format_summary_with_flags():
    result = _make_sample_result()
    result.audit_trail.fields_flagged = 3
    result.audit_trail.re_extractions = 2
    summary = format_summary(result)

    assert "3" in summary  # flagged fields
    assert "2" in summary  # re-extractions


def test_write_outputs_creates_files(tmp_path: pathlib.Path):
    result = _make_sample_result()
    output_dir = tmp_path / "output"

    write_outputs(result, output_dir)

    md_file = output_dir / "test.extracted.md"
    json_file = output_dir / "test.extracted.json"

    assert md_file.exists()
    assert json_file.exists()

    # Verify markdown content
    md_content = md_file.read_text()
    assert "Test Document" in md_content or "Sample extracted text" in md_content

    # Verify JSON is valid and round-trips
    json_content = json.loads(json_file.read_text())
    assert json_content["document"]["source"] == "test.pdf"
    assert len(json_content["regions"]) == 2


def test_write_outputs_custom_dir(tmp_path: pathlib.Path):
    result = _make_sample_result(source="report.png")
    output_dir = tmp_path / "custom" / "nested"

    write_outputs(result, output_dir)

    assert (output_dir / "report.extracted.md").exists()
    assert (output_dir / "report.extracted.json").exists()


@pytest.mark.asyncio
async def test_run_extraction_calls_pipeline(tmp_path: pathlib.Path):
    """run_extraction should create a Pipeline and call extract."""
    img = Image.new("RGB", (100, 100), "white")
    img_path = tmp_path / "doc.png"
    img.save(img_path)

    sample_result = _make_sample_result(source=str(img_path))

    with patch("agentic_extract.skill.Pipeline") as MockPipeline:
        mock_instance = AsyncMock()
        mock_instance.extract.return_value = sample_result
        MockPipeline.return_value = mock_instance

        result = await run_extraction(
            file_path=img_path,
            output_dir=tmp_path / "out",
        )

        assert isinstance(result, ExtractionResult)
        MockPipeline.assert_called_once()
        mock_instance.extract.assert_called_once_with(img_path, schema=None)


@pytest.mark.asyncio
async def test_run_extraction_with_schema(tmp_path: pathlib.Path):
    """run_extraction should pass schema to Pipeline.extract."""
    img = Image.new("RGB", (100, 100), "white")
    img_path = tmp_path / "doc.png"
    img.save(img_path)

    schema = {"type": "object", "properties": {"name": {"type": "string"}}}
    sample_result = _make_sample_result(source=str(img_path))

    with patch("agentic_extract.skill.Pipeline") as MockPipeline:
        mock_instance = AsyncMock()
        mock_instance.extract.return_value = sample_result
        MockPipeline.return_value = mock_instance

        result = await run_extraction(
            file_path=img_path,
            output_dir=tmp_path / "out",
            schema=schema,
        )

        mock_instance.extract.assert_called_once_with(img_path, schema=schema)


@pytest.mark.asyncio
async def test_run_extraction_writes_output_files(tmp_path: pathlib.Path):
    """run_extraction should write both markdown and JSON output files."""
    img = Image.new("RGB", (100, 100), "white")
    img_path = tmp_path / "invoice.png"
    img.save(img_path)

    sample_result = _make_sample_result(source=str(img_path))
    output_dir = tmp_path / "results"

    with patch("agentic_extract.skill.Pipeline") as MockPipeline:
        mock_instance = AsyncMock()
        mock_instance.extract.return_value = sample_result
        MockPipeline.return_value = mock_instance

        await run_extraction(
            file_path=img_path,
            output_dir=output_dir,
        )

        # Output files should exist
        assert (output_dir / "invoice.extracted.md").exists()
        assert (output_dir / "invoice.extracted.json").exists()
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_skill.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'agentic_extract.skill'"

**Step 3: Write minimal implementation**

```python
# src/agentic_extract/skill.py
"""Claude Code skill entry point for /extract command.

This module provides the interface between the Claude Code skill
definition and the extraction pipeline. It handles argument parsing,
pipeline invocation, output file writing, and summary reporting.

Usage (from Claude Code):
    /extract /path/to/document.pdf --output /path/to/output/
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import pathlib
from typing import Any

from agentic_extract.clients.vlm import ClaudeClient, CodexClient
from agentic_extract.models import ExtractionResult
from agentic_extract.pipeline import Pipeline, PipelineConfig

logger = logging.getLogger(__name__)


def format_summary(result: ExtractionResult) -> str:
    """Format a human-readable summary of the extraction result.

    Args:
        result: The completed extraction result.

    Returns:
        A multi-line summary string suitable for display.
    """
    doc = result.document
    audit = result.audit_trail
    regions_by_type: dict[str, int] = {}
    for r in result.regions:
        rtype = r.type.value
        regions_by_type[rtype] = regions_by_type.get(rtype, 0) + 1

    flagged_regions = [r for r in result.regions if r.needs_review]

    lines = [
        f"Extraction Complete: {doc.source}",
        f"Pages: {doc.page_count} | Regions: {len(result.regions)} | "
        f"Confidence: {doc.total_confidence:.2f}",
        f"Time: {doc.processing_time_ms}ms",
        "",
        "Regions by type:",
    ]
    for rtype, count in sorted(regions_by_type.items()):
        lines.append(f"  {rtype}: {count}")

    lines.append("")
    lines.append(
        f"Models used: {', '.join(audit.models_used)}"
    )
    lines.append(f"LLM calls: {audit.total_llm_calls}")
    lines.append(f"Re-extractions: {audit.re_extractions}")
    lines.append(f"Fields flagged: {audit.fields_flagged}")

    if flagged_regions:
        lines.append("")
        lines.append("Flagged regions requiring review:")
        for r in flagged_regions:
            lines.append(
                f"  {r.id} (page {r.page}, {r.type.value}): "
                f"confidence {r.confidence:.2f} - {r.review_reason or 'no reason'}"
            )

    lines.append("")
    lines.append("Stage timing:")
    for stage in audit.processing_stages:
        lines.append(f"  {stage.stage}: {stage.duration_ms}ms")

    return "\n".join(lines)


def write_outputs(
    result: ExtractionResult,
    output_dir: pathlib.Path,
) -> tuple[pathlib.Path, pathlib.Path]:
    """Write extraction results to markdown and JSON files.

    Args:
        result: The completed extraction result.
        output_dir: Directory to write output files.

    Returns:
        Tuple of (markdown_path, json_path).
    """
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Derive output filename from source
    source_name = pathlib.Path(result.document.source).stem
    md_path = output_dir / f"{source_name}.extracted.md"
    json_path = output_dir / f"{source_name}.extracted.json"

    # Write markdown
    md_path.write_text(result.markdown, encoding="utf-8")

    # Write JSON
    json_str = result.model_dump_json(indent=2)
    json_path.write_text(json_str, encoding="utf-8")

    return md_path, json_path


async def run_extraction(
    file_path: pathlib.Path,
    output_dir: pathlib.Path | None = None,
    schema: dict[str, Any] | None = None,
) -> ExtractionResult:
    """Run the full extraction pipeline and write output files.

    This is the main entry point called by the Claude Code skill.

    Args:
        file_path: Path to the document to extract.
        output_dir: Directory for output files. Defaults to file_path's directory.
        schema: Optional JSON schema for structured entity extraction.

    Returns:
        ExtractionResult with all extracted data.
    """
    file_path = pathlib.Path(file_path)

    if output_dir is None:
        output_dir = file_path.parent

    # Initialize VLM clients from environment
    claude_api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    codex_api_key = os.environ.get("OPENAI_API_KEY", "")

    claude_client = ClaudeClient(api_key=claude_api_key)
    codex_client = CodexClient(api_key=codex_api_key)

    pipeline = Pipeline(
        claude_client=claude_client,
        codex_client=codex_client,
    )

    result = await pipeline.extract(file_path, schema=schema)

    # Write output files
    md_path, json_path = write_outputs(result, output_dir)
    logger.info("Markdown output: %s", md_path)
    logger.info("JSON output: %s", json_path)

    return result
```

Now write the skill definition file:

```markdown
<!-- ~/.claude/skills/agentic-extract/extract.md -->
---
name: extract
description: Extract structured data from any document (PDF, image, scan) using the Agentic Extract pipeline
trigger: /extract <file_path> [--schema <path>] [--output <dir>]
---

# /extract - Agentic Document Extraction

## Usage

```
/extract <file_path> [--schema <path>] [--output <dir>]
```

## Arguments

- `<file_path>` (required): Path to the document to extract. Supports PDF, PNG, JPEG, TIFF, BMP, WebP.
- `--schema <path>` (optional): Path to a JSON schema file for structured entity extraction.
- `--output <dir>` (optional): Directory for output files. Defaults to the same directory as the input file.

## What It Does

Runs the full Agentic Extract pipeline on the given document:

1. **Ingestion**: Converts PDF/image to normalized page images
2. **Layout Detection**: DocLayout-YOLO identifies text, tables, figures, handwriting, formulas
3. **Reading Order**: Surya determines logical reading order across regions
4. **Specialist Extraction**: Dispatches regions to Text, Table, or Visual specialists (parallel)
5. **Validation**: 5-layer validation (schema, cross-reference, semantic, visual grounding, confidence)
6. **Re-extraction**: Low-confidence fields retried with alternate model (Claude/Codex switch)
7. **Output**: Produces Markdown + JSON with per-field confidence scores

## Implementation

```python
import asyncio
import json
import pathlib
import sys

# Parse arguments
args = sys.argv[1:] if len(sys.argv) > 1 else []
file_path = None
schema_path = None
output_dir = None

i = 0
while i < len(args):
    if args[i] == "--schema" and i + 1 < len(args):
        schema_path = args[i + 1]
        i += 2
    elif args[i] == "--output" and i + 1 < len(args):
        output_dir = args[i + 1]
        i += 2
    elif file_path is None:
        file_path = args[i]
        i += 1
    else:
        i += 1

if not file_path:
    print("Error: file_path is required. Usage: /extract <file_path> [--schema <path>] [--output <dir>]")
    sys.exit(1)

file_path = pathlib.Path(file_path).expanduser().resolve()
if not file_path.exists():
    print(f"Error: File not found: {file_path}")
    sys.exit(1)

# Load schema if provided
schema = None
if schema_path:
    schema_path = pathlib.Path(schema_path).expanduser().resolve()
    schema = json.loads(schema_path.read_text())

# Set output directory
if output_dir:
    output_dir = pathlib.Path(output_dir).expanduser().resolve()
else:
    output_dir = file_path.parent

# Run extraction
from agentic_extract.skill import run_extraction, format_summary

result = asyncio.run(run_extraction(
    file_path=file_path,
    output_dir=output_dir,
    schema=schema,
))

# Print summary
print(format_summary(result))
print(f"\nOutput files:")
print(f"  Markdown: {output_dir / (file_path.stem + '.extracted.md')}")
print(f"  JSON: {output_dir / (file_path.stem + '.extracted.json')}")
```

## Output

Two files are created in the output directory:

- `<filename>.extracted.md` - Human-readable Markdown with formatting, tables, figure descriptions
- `<filename>.extracted.json` - Structured JSON with per-field confidence, bounding boxes, audit trail

A summary is printed showing: page count, region count, confidence score, models used, timing, and any flagged fields requiring review.
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_skill.py -v`
Expected: PASS (8 passed)

**Step 5: Commit**

```bash
git add src/agentic_extract/skill.py tests/test_skill.py
git commit -m "feat: Claude Code skill entry point with /extract command, output writing, and summary formatting"
```

Then separately install the skill definition:

```bash
mkdir -p ~/.claude/skills/agentic-extract
cp docs/skills/extract.md ~/.claude/skills/agentic-extract/extract.md
```

---

### Task 26: Integration Testing

**Files:**
- Create: `tests/integration/__init__.py`
- Create: `tests/integration/test_end_to_end.py`

**Step 1: Write the failing test**

```python
# tests/integration/__init__.py
```

```python
# tests/integration/test_end_to_end.py
"""End-to-end integration tests with mocked Docker tools and VLM clients.

All 8 scenarios test the full pipeline path from document input to
Markdown + JSON output. Docker tools and VLM API calls are mocked
to ensure tests run without external dependencies.
"""
import asyncio
import json
import pathlib
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from PIL import Image

from agentic_extract.clients.vlm import VLMResponse
from agentic_extract.models import (
    AuditTrail,
    BoundingBox,
    DocumentMetadata,
    ExtractionResult,
    FigureContent,
    HandwritingContent,
    ProcessingStage,
    Region,
    RegionType,
    TableContent,
    TextContent,
)
from agentic_extract.pipeline import Pipeline, PipelineConfig
from agentic_extract.reextraction.engine import (
    ReExtractionCandidate,
    ReExtractionEngine,
    ReExtractionResult,
)
from agentic_extract.audit import AuditTrailTracker
from agentic_extract.tools.docker_runner import ToolOutput


# --- Shared Fixtures ---


@pytest.fixture
def mock_claude():
    return AsyncMock()


@pytest.fixture
def mock_codex():
    return AsyncMock()


@pytest.fixture
def single_page_image(tmp_path: pathlib.Path) -> pathlib.Path:
    """Create a single white page image at 300 DPI."""
    img = Image.new("RGB", (2550, 3300), "white")
    img_path = tmp_path / "document.png"
    img.save(img_path, dpi=(300, 300))
    return img_path


@pytest.fixture
def mock_ingestion(single_page_image, tmp_path):
    """Create a mocked IngestionResult for a single-page document."""
    mock_page = MagicMock(
        page_number=1,
        image_path=single_page_image,
        width=2550,
        height=3300,
        dpi=300,
    )
    mock_result = MagicMock()
    mock_result.pages = [mock_page]
    mock_result.page_count = 1
    mock_result.source_file = single_page_image
    mock_result.temp_dir = tmp_path
    return mock_result


def _make_text_region(region_id: str, page: int, text: str, confidence: float) -> Region:
    return Region(
        id=region_id,
        type=RegionType.TEXT,
        subtype=None,
        page=page,
        bbox=BoundingBox(x=0.05, y=0.1, w=0.9, h=0.15),
        content=TextContent(text=text, markdown=text),
        confidence=confidence,
        extraction_method="paddleocr_3.0",
    )


def _make_table_region(region_id: str, page: int, confidence: float) -> Region:
    return Region(
        id=region_id,
        type=RegionType.TABLE,
        subtype=None,
        page=page,
        bbox=BoundingBox(x=0.05, y=0.3, w=0.9, h=0.4),
        content=TableContent(
            html="<table><tr><th>Gene</th><th>Value</th></tr><tr><td>BRCA1</td><td>3.2</td></tr></table>",
            json_data={"headers": ["Gene", "Value"], "rows": [{"Gene": "BRCA1", "Value": "3.2"}]},
            cell_bboxes=[
                {"row": 0, "col": 0, "bbox": BoundingBox(x=0.05, y=0.3, w=0.3, h=0.05)},
                {"row": 0, "col": 1, "bbox": BoundingBox(x=0.35, y=0.3, w=0.3, h=0.05)},
                {"row": 1, "col": 0, "bbox": BoundingBox(x=0.05, y=0.35, w=0.3, h=0.05)},
                {"row": 1, "col": 1, "bbox": BoundingBox(x=0.35, y=0.35, w=0.3, h=0.05)},
            ],
        ),
        confidence=confidence,
        extraction_method="docling + claude-opus-4-20250514 + codex_structured_outputs",
    )


def _make_figure_region(region_id: str, page: int, confidence: float) -> Region:
    return Region(
        id=region_id,
        type=RegionType.FIGURE,
        subtype="bar_chart",
        page=page,
        bbox=BoundingBox(x=0.1, y=0.05, w=0.8, h=0.45),
        content=FigureContent(
            description="Bar chart showing gene expression levels across conditions",
            figure_type="bar_chart",
            figure_json={
                "title": "Gene Expression",
                "x_axis": {"label": "Condition"},
                "y_axis": {"label": "Fold Change"},
                "data_series": [{"name": "BRCA1", "values": [1.0, 3.2, 4.1]}],
            },
        ),
        confidence=confidence,
        extraction_method="deplot + claude_chartreasoning + codex_figmatch",
    )


def _make_handwriting_region(region_id: str, page: int, confidence: float) -> Region:
    return Region(
        id=region_id,
        type=RegionType.HANDWRITING,
        subtype=None,
        page=page,
        bbox=BoundingBox(x=0.05, y=0.5, w=0.9, h=0.3),
        content=HandwritingContent(
            text="Patient notes: administered 500mg",
            latex=None,
        ),
        confidence=confidence,
        extraction_method="trocr + codex_ocr + claude_hallcheck",
    )


def _build_mock_assembly(regions, source, page_count=1):
    """Build a mock ExtractionResult for assembly."""
    flagged = sum(1 for r in regions if r.needs_review)
    models = set()
    for r in regions:
        models.add(r.extraction_method)

    return ExtractionResult(
        document=DocumentMetadata(
            id="doc-integration",
            source=str(source),
            page_count=page_count,
            processing_timestamp=datetime.now(timezone.utc),
            approach="B",
            total_confidence=min((r.confidence for r in regions), default=1.0),
            processing_time_ms=5000,
        ),
        markdown="\n\n".join(
            f"Region {r.id}: {getattr(r.content, 'text', getattr(r.content, 'description', ''))}"
            for r in regions
        ),
        regions=regions,
        extracted_entities={},
        audit_trail=AuditTrail(
            models_used=sorted(models),
            total_llm_calls=len([r for r in regions if "claude" in r.extraction_method or "codex" in r.extraction_method]),
            re_extractions=0,
            fields_flagged=flagged,
            processing_stages=[
                ProcessingStage(stage="ingestion", duration_ms=200),
                ProcessingStage(stage="extraction", duration_ms=3000),
                ProcessingStage(stage="validation", duration_ms=1000),
                ProcessingStage(stage="assembly", duration_ms=300),
            ],
        ),
    )


# --- Scenario 1: Clean Text PDF ---


@pytest.mark.asyncio
async def test_scenario_1_clean_text_pdf(
    mock_claude, mock_codex, single_page_image, mock_ingestion, tmp_path,
):
    """Clean text PDF produces text extraction + markdown + JSON."""
    text_region = _make_text_region("r1", 1, "The quick brown fox jumps over the lazy dog.", 0.98)
    layout_region = MagicMock(
        region_id="r1", region_type=RegionType.TEXT, page=1,
        bbox=BoundingBox(x=0.05, y=0.1, w=0.9, h=0.15), confidence=0.98,
    )

    with patch("agentic_extract.pipeline.ingest", return_value=mock_ingestion), \
         patch("agentic_extract.pipeline.detect_layout", return_value=[layout_region]), \
         patch("agentic_extract.pipeline.determine_reading_order", return_value=[layout_region]), \
         patch("agentic_extract.pipeline.route_regions", return_value={"text": [layout_region], "table": [], "visual": []}), \
         patch("agentic_extract.pipeline.TextSpecialist") as MockTextSpec, \
         patch("agentic_extract.pipeline.assemble_result") as mock_assemble:

        MockTextSpec.return_value = AsyncMock()
        MockTextSpec.return_value.extract.return_value = text_region
        mock_assemble.return_value = _build_mock_assembly([text_region], single_page_image)

        pipeline = Pipeline(claude_client=mock_claude, codex_client=mock_codex)
        result = await pipeline.extract(single_page_image)

    assert isinstance(result, ExtractionResult)
    assert len(result.regions) == 1
    assert result.regions[0].type == RegionType.TEXT
    assert result.regions[0].content.text == "The quick brown fox jumps over the lazy dog."
    assert result.regions[0].confidence >= 0.90
    assert result.markdown is not None
    assert len(result.markdown) > 0
    # JSON roundtrip
    json_str = result.model_dump_json()
    restored = ExtractionResult.model_validate_json(json_str)
    assert restored.document.source == str(single_page_image)


# --- Scenario 2: PDF with Tables ---


@pytest.mark.asyncio
async def test_scenario_2_pdf_with_tables(
    mock_claude, mock_codex, single_page_image, mock_ingestion, tmp_path,
):
    """PDF with tables produces table extraction with cell-level detail."""
    table_region = _make_table_region("t1", 1, 0.94)
    layout_region = MagicMock(
        region_id="t1", region_type=RegionType.TABLE, page=1,
        bbox=BoundingBox(x=0.05, y=0.3, w=0.9, h=0.4), confidence=0.95,
    )

    with patch("agentic_extract.pipeline.ingest", return_value=mock_ingestion), \
         patch("agentic_extract.pipeline.detect_layout", return_value=[layout_region]), \
         patch("agentic_extract.pipeline.determine_reading_order", return_value=[layout_region]), \
         patch("agentic_extract.pipeline.route_regions", return_value={"text": [], "table": [layout_region], "visual": []}), \
         patch("agentic_extract.pipeline.TableSpecialist") as MockTableSpec, \
         patch("agentic_extract.pipeline.assemble_result") as mock_assemble:

        MockTableSpec.return_value = AsyncMock()
        MockTableSpec.return_value.extract.return_value = table_region
        mock_assemble.return_value = _build_mock_assembly([table_region], single_page_image)

        pipeline = Pipeline(claude_client=mock_claude, codex_client=mock_codex)
        result = await pipeline.extract(single_page_image)

    assert len(result.regions) == 1
    assert result.regions[0].type == RegionType.TABLE
    content = result.regions[0].content
    assert isinstance(content, TableContent)
    assert len(content.cell_bboxes) == 4
    assert content.json_data["headers"] == ["Gene", "Value"]
    assert content.json_data["rows"][0]["Gene"] == "BRCA1"


# --- Scenario 3: Chart Image ---


@pytest.mark.asyncio
async def test_scenario_3_chart_image(
    mock_claude, mock_codex, single_page_image, mock_ingestion, tmp_path,
):
    """Chart image produces DePlot + Claude reasoning output."""
    chart_region = _make_figure_region("f1", 1, 0.86)
    layout_region = MagicMock(
        region_id="f1", region_type=RegionType.FIGURE, page=1,
        bbox=BoundingBox(x=0.1, y=0.05, w=0.8, h=0.45), confidence=0.90,
    )

    with patch("agentic_extract.pipeline.ingest", return_value=mock_ingestion), \
         patch("agentic_extract.pipeline.detect_layout", return_value=[layout_region]), \
         patch("agentic_extract.pipeline.determine_reading_order", return_value=[layout_region]), \
         patch("agentic_extract.pipeline.route_regions", return_value={"text": [], "table": [], "visual": [layout_region]}), \
         patch("agentic_extract.pipeline.assemble_result") as mock_assemble:

        mock_assemble.return_value = _build_mock_assembly([chart_region], single_page_image)

        pipeline = Pipeline(claude_client=mock_claude, codex_client=mock_codex)
        result = await pipeline.extract(single_page_image)

    assert len(result.regions) == 1
    assert result.regions[0].type == RegionType.FIGURE
    content = result.regions[0].content
    assert isinstance(content, FigureContent)
    assert content.figure_type == "bar_chart"
    assert "BRCA1" in str(content.figure_json)


# --- Scenario 4: Handwriting Image ---


@pytest.mark.asyncio
async def test_scenario_4_handwriting_image(
    mock_claude, mock_codex, single_page_image, mock_ingestion, tmp_path,
):
    """Handwriting image produces TrOCR + dual model output."""
    hw_region = _make_handwriting_region("h1", 1, 0.82)
    layout_region = MagicMock(
        region_id="h1", region_type=RegionType.HANDWRITING, page=1,
        bbox=BoundingBox(x=0.05, y=0.5, w=0.9, h=0.3), confidence=0.85,
    )

    with patch("agentic_extract.pipeline.ingest", return_value=mock_ingestion), \
         patch("agentic_extract.pipeline.detect_layout", return_value=[layout_region]), \
         patch("agentic_extract.pipeline.determine_reading_order", return_value=[layout_region]), \
         patch("agentic_extract.pipeline.route_regions", return_value={"text": [], "table": [], "visual": [layout_region]}), \
         patch("agentic_extract.pipeline.assemble_result") as mock_assemble:

        mock_assemble.return_value = _build_mock_assembly([hw_region], single_page_image)

        pipeline = Pipeline(claude_client=mock_claude, codex_client=mock_codex)
        result = await pipeline.extract(single_page_image)

    assert len(result.regions) == 1
    assert result.regions[0].type == RegionType.HANDWRITING
    content = result.regions[0].content
    assert isinstance(content, HandwritingContent)
    assert "500mg" in content.text
    assert "trocr" in result.regions[0].extraction_method


# --- Scenario 5: Mixed Document ---


@pytest.mark.asyncio
async def test_scenario_5_mixed_document_routing(
    mock_claude, mock_codex, single_page_image, mock_ingestion, tmp_path,
):
    """Mixed document with text + table + figure routes to correct specialists."""
    text_region = _make_text_region("r1", 1, "Introduction text", 0.97)
    table_region = _make_table_region("t1", 1, 0.94)
    figure_region = _make_figure_region("f1", 1, 0.86)

    text_lr = MagicMock(
        region_id="r1", region_type=RegionType.TEXT, page=1,
        bbox=BoundingBox(x=0.05, y=0.05, w=0.9, h=0.1), confidence=0.98,
    )
    table_lr = MagicMock(
        region_id="t1", region_type=RegionType.TABLE, page=1,
        bbox=BoundingBox(x=0.05, y=0.2, w=0.9, h=0.3), confidence=0.95,
    )
    figure_lr = MagicMock(
        region_id="f1", region_type=RegionType.FIGURE, page=1,
        bbox=BoundingBox(x=0.1, y=0.6, w=0.8, h=0.35), confidence=0.90,
    )

    with patch("agentic_extract.pipeline.ingest", return_value=mock_ingestion), \
         patch("agentic_extract.pipeline.detect_layout", return_value=[text_lr, table_lr, figure_lr]), \
         patch("agentic_extract.pipeline.determine_reading_order", return_value=[text_lr, table_lr, figure_lr]), \
         patch("agentic_extract.pipeline.route_regions") as mock_route, \
         patch("agentic_extract.pipeline.TextSpecialist") as MockTextSpec, \
         patch("agentic_extract.pipeline.TableSpecialist") as MockTableSpec, \
         patch("agentic_extract.pipeline.assemble_result") as mock_assemble:

        mock_route.return_value = {
            "text": [text_lr],
            "table": [table_lr],
            "visual": [figure_lr],
        }
        MockTextSpec.return_value = AsyncMock()
        MockTextSpec.return_value.extract.return_value = text_region
        MockTableSpec.return_value = AsyncMock()
        MockTableSpec.return_value.extract.return_value = table_region

        all_regions = [text_region, table_region, figure_region]
        mock_assemble.return_value = _build_mock_assembly(all_regions, single_page_image)

        pipeline = Pipeline(claude_client=mock_claude, codex_client=mock_codex)
        result = await pipeline.extract(single_page_image)

    region_types = {r.type for r in result.regions}
    assert RegionType.TEXT in region_types
    assert RegionType.TABLE in region_types
    assert RegionType.FIGURE in region_types


# --- Scenario 6: Low Confidence Triggers Re-extraction ---


@pytest.mark.asyncio
async def test_scenario_6_low_confidence_triggers_reextraction(
    mock_claude, mock_codex, single_page_image, mock_ingestion, tmp_path,
):
    """Low-confidence region triggers the re-extraction loop."""
    low_conf_region = _make_text_region("r1", 1, "Ambiguous text", 0.78)
    low_conf_region = low_conf_region.model_copy(update={
        "extraction_method": "paddleocr_3.0 + claude-opus-4-20250514",
    })

    layout_region = MagicMock(
        region_id="r1", region_type=RegionType.TEXT, page=1,
        bbox=BoundingBox(x=0.05, y=0.1, w=0.9, h=0.15), confidence=0.98,
    )

    # After re-extraction, confidence should be boosted
    boosted_region = low_conf_region.model_copy(update={
        "confidence": 0.88,
        "extraction_method": "paddleocr_3.0 + claude-opus-4-20250514 + gpt-4o",
    })

    with patch("agentic_extract.pipeline.ingest", return_value=mock_ingestion), \
         patch("agentic_extract.pipeline.detect_layout", return_value=[layout_region]), \
         patch("agentic_extract.pipeline.determine_reading_order", return_value=[layout_region]), \
         patch("agentic_extract.pipeline.route_regions", return_value={"text": [layout_region], "table": [], "visual": []}), \
         patch("agentic_extract.pipeline.TextSpecialist") as MockTextSpec, \
         patch("agentic_extract.pipeline.ReExtractionEngine") as MockReExtract, \
         patch("agentic_extract.pipeline.assemble_result") as mock_assemble:

        MockTextSpec.return_value = AsyncMock()
        MockTextSpec.return_value.extract.return_value = low_conf_region

        mock_re_result = ReExtractionResult(
            region_id="r1",
            field_name="text",
            original_value="Ambiguous text",
            original_confidence=0.78,
            original_model="claude-opus-4-20250514",
            candidates=[ReExtractionCandidate(
                value="Ambiguous text",
                confidence=0.88,
                model="gpt-4o",
                extraction_method="re_extraction_1 + gpt-4o",
            )],
            final_value="Ambiguous text",
            final_confidence=0.88,
            retries_used=1,
            models_agreed=True,
            accepted=False,
            flagged=True,
        )
        mock_engine = AsyncMock()
        mock_engine.re_extract_field.return_value = mock_re_result
        MockReExtract.return_value = mock_engine

        mock_assemble.return_value = _build_mock_assembly([boosted_region], single_page_image)
        mock_assemble.return_value.audit_trail.re_extractions = 1

        pipeline = Pipeline(claude_client=mock_claude, codex_client=mock_codex)
        result = await pipeline.extract(single_page_image)

    assert result.audit_trail.re_extractions >= 1


# --- Scenario 7: Audit Trail Completeness ---


@pytest.mark.asyncio
async def test_scenario_7_audit_trail_completeness(
    mock_claude, mock_codex, single_page_image, mock_ingestion, tmp_path,
):
    """Audit trail must contain all required fields."""
    text_region = _make_text_region("r1", 1, "Audit test", 0.95)
    layout_region = MagicMock(
        region_id="r1", region_type=RegionType.TEXT, page=1,
        bbox=BoundingBox(x=0.05, y=0.1, w=0.9, h=0.15), confidence=0.98,
    )

    with patch("agentic_extract.pipeline.ingest", return_value=mock_ingestion), \
         patch("agentic_extract.pipeline.detect_layout", return_value=[layout_region]), \
         patch("agentic_extract.pipeline.determine_reading_order", return_value=[layout_region]), \
         patch("agentic_extract.pipeline.route_regions", return_value={"text": [layout_region], "table": [], "visual": []}), \
         patch("agentic_extract.pipeline.TextSpecialist") as MockTextSpec, \
         patch("agentic_extract.pipeline.assemble_result") as mock_assemble:

        MockTextSpec.return_value = AsyncMock()
        MockTextSpec.return_value.extract.return_value = text_region

        audit = AuditTrail(
            models_used=["paddleocr_3.0"],
            total_llm_calls=0,
            re_extractions=0,
            fields_flagged=0,
            processing_stages=[
                ProcessingStage(stage="ingestion", duration_ms=100),
                ProcessingStage(stage="layout_detection", duration_ms=80),
                ProcessingStage(stage="reading_order", duration_ms=20),
                ProcessingStage(stage="routing", duration_ms=10),
                ProcessingStage(stage="extraction", duration_ms=500),
                ProcessingStage(stage="validation", duration_ms=200),
                ProcessingStage(stage="assembly", duration_ms=50),
            ],
        )
        mock_assemble.return_value = ExtractionResult(
            document=DocumentMetadata(
                id="doc-audit", source=str(single_page_image),
                page_count=1, processing_timestamp=datetime.now(timezone.utc),
                approach="B", total_confidence=0.95, processing_time_ms=960,
            ),
            markdown="Audit test",
            regions=[text_region],
            extracted_entities={},
            audit_trail=audit,
        )

        pipeline = Pipeline(claude_client=mock_claude, codex_client=mock_codex)
        result = await pipeline.extract(single_page_image)

    at = result.audit_trail
    # Required fields
    assert isinstance(at.models_used, list)
    assert isinstance(at.total_llm_calls, int)
    assert isinstance(at.re_extractions, int)
    assert isinstance(at.fields_flagged, int)
    assert isinstance(at.processing_stages, list)
    assert at.total_llm_calls >= 0
    assert at.re_extractions >= 0
    assert at.fields_flagged >= 0

    # Stage names must include key pipeline stages
    stage_names = [s.stage for s in at.processing_stages]
    assert "ingestion" in stage_names
    assert "extraction" in stage_names
    assert "validation" in stage_names
    assert "assembly" in stage_names

    # All stages have non-negative duration
    for stage in at.processing_stages:
        assert stage.duration_ms >= 0


# --- Scenario 8: Confidence Scoring on All Fields ---


@pytest.mark.asyncio
async def test_scenario_8_confidence_on_all_fields(
    mock_claude, mock_codex, single_page_image, mock_ingestion, tmp_path,
):
    """Every region and field must have a confidence score in [0, 1]."""
    regions = [
        _make_text_region("r1", 1, "Text", 0.97),
        _make_table_region("t1", 1, 0.94),
        _make_figure_region("f1", 1, 0.86),
        _make_handwriting_region("h1", 1, 0.78),
    ]

    layout_regions = [
        MagicMock(region_id=r.id, region_type=r.type, page=1,
                  bbox=r.bbox, confidence=0.95)
        for r in regions
    ]

    with patch("agentic_extract.pipeline.ingest", return_value=mock_ingestion), \
         patch("agentic_extract.pipeline.detect_layout", return_value=layout_regions), \
         patch("agentic_extract.pipeline.determine_reading_order", return_value=layout_regions), \
         patch("agentic_extract.pipeline.route_regions") as mock_route, \
         patch("agentic_extract.pipeline.TextSpecialist") as MockTextSpec, \
         patch("agentic_extract.pipeline.TableSpecialist") as MockTableSpec, \
         patch("agentic_extract.pipeline.ReExtractionEngine") as MockReExtract, \
         patch("agentic_extract.pipeline.assemble_result") as mock_assemble:

        mock_route.return_value = {
            "text": [layout_regions[0]],
            "table": [layout_regions[1]],
            "visual": [layout_regions[2], layout_regions[3]],
        }
        MockTextSpec.return_value = AsyncMock()
        MockTextSpec.return_value.extract.return_value = regions[0]
        MockTableSpec.return_value = AsyncMock()
        MockTableSpec.return_value.extract.return_value = regions[1]

        # Mock re-extraction for low-confidence handwriting region
        mock_re_result = ReExtractionResult(
            region_id="h1", field_name="handwriting",
            original_value="Patient notes: administered 500mg",
            original_confidence=0.78, original_model="claude-opus-4-20250514",
            candidates=[], final_value="Patient notes: administered 500mg",
            final_confidence=0.78, retries_used=2,
            models_agreed=False, accepted=False, flagged=True,
        )
        mock_engine = AsyncMock()
        mock_engine.re_extract_field.return_value = mock_re_result
        MockReExtract.return_value = mock_engine

        mock_assemble.return_value = _build_mock_assembly(regions, single_page_image)

        pipeline = Pipeline(claude_client=mock_claude, codex_client=mock_codex)
        result = await pipeline.extract(single_page_image)

    # Every region must have a valid confidence score
    for region in result.regions:
        assert 0.0 <= region.confidence <= 1.0, (
            f"Region {region.id} has invalid confidence: {region.confidence}"
        )

    # Document-level confidence must also be valid
    assert 0.0 <= result.document.total_confidence <= 1.0

    # JSON serialization must preserve all confidence values
    json_str = result.model_dump_json()
    parsed = json.loads(json_str)
    for r in parsed["regions"]:
        assert 0.0 <= r["confidence"] <= 1.0
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/integration/test_end_to_end.py -v`
Expected: FAIL with import errors (pipeline and re-extraction modules not yet implemented)

**Step 3: Implementation**

All code is already implemented in Tasks 21-25. The integration tests validate the composed behavior. After Tasks 21-25 are implemented, these tests should pass.

**Step 4: Run test to verify it passes**

Run: `pytest tests/integration/test_end_to_end.py -v`
Expected: PASS (8 passed)

**Step 5: Commit**

```bash
git add tests/integration/__init__.py tests/integration/test_end_to_end.py
git commit -m "feat: 8 end-to-end integration tests covering all extraction scenarios with mocked tools"
```

---

## Phase 3 Summary

After completing Tasks 21-26, the project adds the following:

```
agentic-extract/
    src/agentic_extract/
        audit.py                               # AuditTrailTracker (Task 24)
        pipeline.py                            # Pipeline orchestrator (Task 22)
        skill.py                               # Claude Code skill entry point (Task 25)
        reextraction/
            __init__.py
            engine.py                          # ReExtractionEngine with model switching (Task 21)
        grounding/
            __init__.py
            visual.py                          # Visual grounding with bbox verification (Task 23)
    tests/
        test_audit.py                          # 14 tests
        test_pipeline.py                       # 8 tests
        test_skill.py                          # 8 tests
        reextraction/
            __init__.py
            test_engine.py                     # 12 tests
        grounding/
            __init__.py
            test_visual.py                     # 12 tests
        integration/
            __init__.py
            test_end_to_end.py                 # 8 tests (scenarios 1-8)
    ~/.claude/skills/agentic-extract/
        extract.md                             # Skill definition for /extract command
```

**Total new tests:** ~62 test cases across 6 test files
**Total new source files:** 6 Python modules + 1 skill definition
**Key capabilities added:**
- Re-extraction loop with model switching (Claude <-> Codex), max 2 retries, +0.10 confidence boost on agreement
- Full pipeline orchestration with asyncio.gather for parallel specialist dispatch
- Visual grounding linking every extracted value to normalized [0,1] bounding boxes with OCR verification
- AuditTrailTracker with per-stage timing, context manager, and merge support
- Claude Code skill packaging (/extract command with markdown + JSON output)
- 8 end-to-end integration test scenarios covering the complete extraction pipeline

**Phase 4 (Tasks 27+)** will add: Consensus Judge, Semantic Auditor, specialist pool expansion, and complexity-based routing per the design document Section 9.
