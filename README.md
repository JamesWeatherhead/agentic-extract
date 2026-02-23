# Agentic Extract

A 3-agent document extraction system that composes 12 open-source tools with dual-model (Claude + Codex) routing to produce Markdown and structured JSON from any document type. Deployed as a Claude Code skill.

**Status:** Early development (v0.1.0). Core data models, coordinator scaffolding, specialist stubs, and validation layers are implemented. Docker tool runner is functional. End-to-end pipeline is not yet operational.

## What It Does

Accepts PDFs, images, or URLs. Detects layout regions (text, tables, figures, charts, handwriting, formulas). Routes each region to a specialist that combines open-source OCR/detection tools with VLM enhancement. Validates every extracted field through a 5-layer pipeline. Returns both Markdown (human-readable) and JSON (machine-readable) with per-field confidence scores and full extraction provenance.

Covers document types that most extraction systems do not: molecular structures (SMILES/InChI), gel electrophoresis images, multi-panel scientific figures, handwritten math (LaTeX), and degraded scans.

## Architecture

Three agents, deterministically routed. The Coordinator is primarily rule-based code, not an LLM agent. LLM calls are reserved for genuinely ambiguous classification and for the extraction/validation layers where reasoning is required.

```
+------------------+
|  DOCUMENT INPUT  |   PDF / image / URL
+--------+---------+
         |
         v
+========================================+
|        COORDINATOR AGENT               |
|                                        |
|  DocLayout-YOLO    (layout detection)  |
|  Surya             (reading order)     |
|  DocEnTr           (scan enhancement)  |
|  pdf2image/PyMuPDF (ingestion)         |
|                                        |
|  Deterministic routing to specialists  |
|  (Claude call only for ambiguous regions)
+====+=========+=========+==============+
     |         |         |
     v         v         v
+--------+ +--------+ +--------+
|  TEXT  | | TABLE  | | VISUAL |   Specialist Pool
|        | |        | |        |
| Paddle | | Docling| | FigEx2 |   Open-source first pass
| OCR    | | HTML   | | DePlot |
|   +    | |   +    | | DECIMER|
| Claude | | Claude | | GelGenie
|        | | Codex  | | TrOCR  |
|        | |        | | pix2tex|
|        | |        | |   +    |
|        | |        | | Codex  |   VLM enhancement
|        | |        | | Claude |
+---+----+ +---+----+ +---+----+
    |          |          |
    +----------+----------+
               |
               v
+========================================+
|        VALIDATOR AGENT                 |
|                                        |
|  L1: Schema validation (deterministic) |
|  L2: Cross-reference checks            |
|  L3: Semantic validation (Claude)      |
|  L4: Visual grounding check            |
|  L5: Confidence calibration            |
|                                        |
|  >= 0.90  ->  ACCEPT                   |
|  0.70-0.89 -> RE-EXTRACT (alt model)  |
|  < 0.70  ->  FLAG FOR REVIEW           |
+========================================+
               |
     +---------+---------+
     v                   v
+-----------+     +-----------+
| MARKDOWN  |     |   JSON    |
| + visual  |     | + per-field|
| grounding |     | confidence |
+-----------+     +-----------+
```

### Key Design Choices

- **OCR-then-LLM pattern.** Open-source tools extract first; VLMs correct and enhance. This follows AWS ICCV 2025 research showing CER drops from 0.036 to 0.01 with the combined approach.
- **Benchmark-driven model routing.** Claude handles reasoning, hallucination-sensitive tasks, and chart interpretation. Codex handles figure matching, schema enforcement, and character-level OCR. Assignment is based on published benchmark results, not brand preference.
- **Null over hallucination.** Missing data is `null`, never fabricated. A null at 100% confidence beats a hallucinated value at 90%.
- **Confidence-based rejection with re-extraction.** Fields in the 0.70-0.90 range are re-extracted with the alternate model (if Claude failed, try Codex; vice versa). Max 2 retries per field.

## Tool Palette

12 open-source tools, each running in its own Docker container.

| #  | Tool                 | Role                                      | Key Metric                              |
|----|----------------------|-------------------------------------------|-----------------------------------------|
| 1  | Docling (IBM)        | PDF parsing, table extraction             | 97.9% on complex tables                 |
| 2  | DocLayout-YOLO       | Layout detection                          | Powers MinerU; best-in-class detection   |
| 3  | PaddleOCR 3.0        | OCR engine                                | 90+ languages; 8.3/10 Pragmile 2025     |
| 4  | Surya                | Reading order, text detection             | 90+ language support                     |
| 5  | GOT-OCR 2.0          | Unified OCR (formulas, tables, molecules) | 580M params; treats all signals uniformly|
| 6  | FigEx2               | Multi-panel figure splitting              | Cross-domain few-shot (Jan 2026)         |
| 7  | DePlot (Google)      | Chart-to-table extraction                 | 29.4% over prior SOTA on ChartQA        |
| 8  | DECIMER.ai           | Chemical structure recognition            | Nature Communications 2023               |
| 9  | GelGenie             | Gel electrophoresis band ID               | Nature Communications 2025               |
| 10 | TrOCR (Microsoft)    | Handwritten text recognition              | SOTA transformer HTR                     |
| 11 | DocEnTr              | Degraded scan enhancement                 | Transformer binarization/deblurring      |
| 12 | pix2tex / LaTeX-OCR  | Handwritten math to LaTeX                 | Primary open-source equation converter   |

## Model Routing

| Task                                    | Model        | Why                                                    |
|-----------------------------------------|--------------|--------------------------------------------------------|
| Table reasoning (merged cells)          | Claude       | Hallucination rate 0.09% (vs GPT-4o 0.15%)            |
| Chart interpretation                    | Claude       | ChartQA 90.8% (vs 85.7%); CharXiv ~60% (vs 47.1%)    |
| Cross-page reasoning                    | Claude       | 1M token context window                                |
| Semantic validation                     | Claude       | DocVQA 95.2% (vs 92.8%)                               |
| Identity document extraction            | Claude       | GPT-4o refuses due to content policy                   |
| Scientific figure matching              | Codex/GPT-4o | SciFIBench 75.4% (vs ~53%); AI2D 94.2% (vs 88.1%)    |
| Handwriting OCR verification            | Codex/GPT-4o | Edit distance 0.02 (vs 0.03); 3x token efficiency     |
| JSON schema enforcement                 | Codex/GPT-4o | Native Structured Outputs API                          |

When both models are used on the same region (complex figures, handwriting, ambiguous tables), disagreements are resolved by confidence-weighted voting.

## Project Structure

```
agentic-extract/
    src/agentic_extract/
        __init__.py
        models.py                     # Pydantic v2 data models (Section 6 schema)
        clients/
            vlm.py                    # Claude + Codex API client abstraction
        coordinator/
            ingestion.py              # PDF-to-image conversion
            layout.py                 # DocLayout-YOLO region detection
            reading_order.py          # Surya reading order
            quality.py                # DPI, skew, degradation assessment
            routing.py                # Deterministic region-to-specialist routing
            assembly.py               # Merge specialist outputs in reading order
        specialists/
            text.py                   # PaddleOCR + Claude
            table.py                  # Docling + Claude + Codex
            visual_chart.py           # DePlot + Claude
            visual_figure.py          # FigEx2 + DECIMER + GelGenie + Codex
            visual_formula.py         # GOT-OCR 2.0 + pix2tex
            visual_handwriting.py     # TrOCR + DocEnTr + dual-model
        validators/
            schema_validator.py       # Layer 1: JSON schema conformance
            crossref_validator.py     # Layer 2: Field consistency checks
            semantic_validator.py     # Layer 3: Claude semantic reasoning
            grounding_validator.py    # Layer 4: Bounding box verification
        reextraction/
            engine.py                 # Re-extraction loop with model switching
        tools/
            docker_runner.py          # Base Docker container execution
    tests/                            # Mirrors src/ structure
    docs/plans/
        2026-02-23-agentic-extract-design.md
        2026-02-23-agentic-extract-implementation.md
        2026-02-23-agentic-extract-implementation-phase3.md
    pyproject.toml
```

## Installation

Requires Python 3.11+ and Docker.

```bash
git clone https://github.com/JamesWeatherhead/agentic-extract.git
cd agentic-extract
pip install -e ".[dev]"
```

### API Keys

Set environment variables for VLM access:

```bash
export ANTHROPIC_API_KEY="your-key"
export OPENAI_API_KEY="your-key"
```

### Docker Images

Each open-source tool runs in its own container. Images must be pulled or built before use. See `docs/plans/2026-02-23-agentic-extract-design.md` Section 10 for the full image list.

## Usage

Not yet available as a CLI or skill. The target deployment is a Claude Code skill invocable via `/extract`.

Target interface:

```
/extract path/to/document.pdf
/extract path/to/document.pdf --schema invoice_schema.json
/extract https://arxiv.org/pdf/2401.12345
```

## Running Tests

```bash
pytest
```

All tests currently pass against the implemented scaffolding. Tests cover: data models, coordinator modules (ingestion, layout, reading order, quality, routing, assembly), specialist stubs, validator layers, re-extraction engine, VLM client, and Docker runner.

## Current Status

### Implemented

- Pydantic v2 data models matching the full output schema (Section 6 of design doc)
- Coordinator modules: ingestion, layout detection, reading order, quality assessment, routing, assembly
- Specialist stubs: text, table, visual (chart, figure, formula, handwriting)
- Validator layers: schema, cross-reference, semantic, visual grounding
- Re-extraction engine with model-switching logic
- VLM client abstraction (Claude + Codex)
- Docker tool runner (subprocess-based, no SDK dependency)
- Full test suite for all modules

### Not Yet Implemented

- Docker images for the 12 open-source tools (need to be built/pulled)
- Live API integration with Claude and Codex
- End-to-end pipeline wiring (coordinator dispatching to real specialists)
- Confidence calibration (temperature scaling on held-out data)
- Claude Code skill packaging (`/extract` command)
- Production benchmarking against target metrics

### Upgrade Path

The architecture is designed for incremental expansion from 3 agents (Approach B) to 7 agents (Approach C):

- Phase 4a: Consensus Judge agent for high-disagreement tiebreaking
- Phase 4b: Semantic Auditor agent for document-level coherence
- Phase 4c: Split Visual Specialist into Chart, Diagram, and Handwriting specialists
- Phase 4d: Complexity-based routing (simple/medium/complex cost scaling)

## Target Benchmarks

These are targets, not claims. The system has not yet been benchmarked against production data.

| Metric                                          | Target       |
|-------------------------------------------------|--------------|
| Text extraction (clean docs)                    | >= 99.5% CER |
| Table extraction (complex tables)               | >= 97% cell F1|
| Chart data extraction                           | >= 90% accuracy|
| Handwriting recognition (accepted fields)       | >= 95% WER   |
| Effective accuracy on accepted fields (conf >= 0.90) | >= 99.9% |
| Expected Calibration Error                      | < 0.05       |
| Latency (10-page paper, no retries)             | <= 25 seconds|
| Latency (10-page paper, with retries)           | <= 45 seconds|

## Technology Stack

| Component       | Technology                    |
|-----------------|-------------------------------|
| Language        | Python 3.11+                  |
| Build system    | Hatch                         |
| Models (primary)| Claude Opus 4.6 (1M context)  |
| Models (secondary)| GPT-4o / Codex (Structured Outputs) |
| Containers      | Docker (one per tool)         |
| Validation      | jsonschema, Pydantic v2       |
| Async           | asyncio                       |
| Testing         | pytest, pytest-asyncio        |

## Design Document

The full architecture, rationale, benchmark evidence, output schemas, and upgrade path are documented in:

`docs/plans/2026-02-23-agentic-extract-design.md` (~1,000 lines)

Every design decision cites specific benchmark numbers and research papers. See Appendix: Research Citations in the design doc.

## License

TBD
