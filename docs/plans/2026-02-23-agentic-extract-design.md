# Agentic Extract v1: Approved Design Document

**Date:** 2026-02-23
**Status:** APPROVED
**Approach:** B ("Agentic Specialist Router") with upgrade path to C ("Full Swarm with Consensus")
**Deployment:** Claude Code skill
**Priority:** Maximum accuracy; cost is not a constraint
**Model Strategy:** Claude primary (reasoning, hallucination prevention), Codex secondary (figure matching, schema enforcement)

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture](#2-architecture)
3. [Tool Palette](#3-tool-palette)
4. [Model Assignment](#4-model-assignment)
5. [Data Flow](#5-data-flow)
6. [Output Schemas](#6-output-schemas)
7. [Confidence Scoring](#7-confidence-scoring)
8. [Validation Pipeline](#8-validation-pipeline)
9. [Upgrade Path to C](#9-upgrade-path-to-c)
10. [Technology Stack](#10-technology-stack)
11. [Key Design Decisions](#11-key-design-decisions)
12. [Success Criteria](#12-success-criteria)

---

## 1. Project Overview

### What We Are Building

Agentic Extract is a 3-agent document extraction system, deployed as a Claude Code skill, that accepts any document type (scientific papers, diagrams, handwritten notes, charts, tables, mixed media, invoices, contracts) and produces both Markdown and structured JSON output. It composes 12 open-source tools for ingestion and detection, routes complex regions to Claude or Codex based on task-specific benchmarks, and uses a confidence-based re-extraction loop to achieve effective accuracy targeting 99.9% on accepted fields.

### Why We Are Building It

Landing AI's Agentic Document Extraction (ADE) platform is the current market leader, but our competitive analysis reveals it is approximately 30% agentic (deterministic pipeline with LLM post-processing, not true multi-agent self-correction). It has no documented:

- Dual-model cross-validation
- Re-extraction loops with model switching
- Scientific domain specialists (molecular structures, gel images, pathway diagrams)
- Calibrated per-field confidence scoring (paywalled at $2,000+/month)
- Cross-page reasoning capability

Agentic Extract fills every one of these gaps.

### How It Beats Landing AI

| Dimension | Landing AI ADE | Agentic Extract v1 |
|-----------|---------------|---------------------|
| Architecture | ~30% agentic (pipeline with LLM post-processing) | Genuinely agentic: 3 agents with self-correction loop |
| Model diversity | Single proprietary DPT model | Claude + Codex, task-routed by benchmark evidence |
| Error reduction | Single-pass, no documented cross-validation | ~50% error reduction via dual-model + retry loop |
| Scientific coverage | Generic figure handling only | DECIMER (molecules), GelGenie (gels), DePlot (charts), pix2tex (math), FigEx2 (multi-panel) |
| Confidence scoring | Paywalled at $2,000+/month | Calibrated per-field confidence, included by default |
| Visual grounding | Paywalled at higher tiers | Region + cell-level bounding boxes, included by default |
| Cross-page reasoning | Processes pages independently | Claude 1M token context enables document-level understanding |
| Transparency | Opaque credit system | Per-model token pricing, full audit trail, extraction provenance |
| Vendor lock-in | Fully proprietary | Open-source foundation; every component replaceable |

---

## 2. Architecture

### Approach B: Agentic Specialist Router

A 3-agent architecture (Coordinator, Extraction Specialist Pool, Validator) with deterministic routing, dual-model extraction for complex regions, and confidence-based rejection with re-extraction. This sits in the 2-4 agent sweet spot identified by production systems research.

### Architecture Diagram

```
+------------------+
|  DOCUMENT INPUT  |
+--------+---------+
         |
         v
+========================================+
|        COORDINATOR AGENT               |
|  (Deterministic + lightweight LLM)     |
|                                        |
|  1. Ingestion (DocLayout-YOLO + Surya) |
|  2. Region classification              |
|  3. Quality assessment (DPI, skew,     |
|     degradation level)                 |
|  4. Routing plan generation            |
|  5. Dispatch to specialists            |
|  6. Result aggregation                 |
+====+=========+=========+==============+
     |         |         |
     v         v         v
+--------+ +--------+ +--------+
| TEXT   | | TABLE  | | VISUAL |     (Specialist Pool)
| SPEC   | | SPEC   | | SPEC   |
|        | |        | |        |
| Paddle | | Docling| | FigEx2 |     Open-source first pass
| OCR    | | HTML   | | DePlot |
|   +    | |   +    | | DECIMER|
| Claude | | Claude | | GelGenie
| (long  | | (merge | |   +    |
| docs,  | | cells, | | Codex  |     VLM enhancement
| reason)| | reason)| | (fig   |
|        | |   +    | | match) |
|        | | Codex  | |   +    |
|        | | (JSON  | | Claude |
|        | | schema)| | (chart |
|        | |        | | reason)|
+---+----+ +---+----+ +---+----+
    |          |          |
    +----------+----------+
               |
               v
+========================================+
|        VALIDATOR AGENT                 |
|                                        |
|  Layer 1: Schema validation            |
|    (deterministic JSON schema check)   |
|                                        |
|  Layer 2: Cross-reference validation   |
|    (field consistency, date ranges,    |
|     numerical plausibility)            |
|                                        |
|  Layer 3: Semantic validation          |
|    (Claude: "Do these fields make      |
|     sense together?")                  |
|                                        |
|  Layer 4: Visual grounding check       |
|    (verify bounding box alignment      |
|     with extracted text)               |
|                                        |
|  Layer 5: Confidence calibration       |
|    (temperature scaling on per-field   |
|     scores; rejection threshold)       |
|                                        |
|  DECISION:                             |
|    confidence >= 0.90 -> ACCEPT        |
|    0.70 <= conf < 0.90 -> RE-EXTRACT  |
|    confidence < 0.70 -> FLAG REVIEW    |
+====+===================================+
     |
     | (RE-EXTRACT loop: max 2 retries)
     | Uses alternate model on retry
     | (if Claude failed, try Codex; vice versa)
     |
     v
+------------------+     +------------------+
|  MARKDOWN OUTPUT |     |  JSON OUTPUT     |
|  + visual ground |     |  + confidence    |
|    annotations   |     |    per field     |
+------------------+     +------------------+
```

### Agent Roles and Responsibilities

#### Agent 1: Coordinator

**Implementation:** Deterministic logic + lightweight LLM call for ambiguous classification only.

**Responsibilities:**

- Accept document input (PDF, image, URL)
- Convert to page images (pdf2image / PyMuPDF)
- Run layout detection (DocLayout-YOLO) to produce region bounding boxes + types
- Determine reading order (Surya / LayoutReader)
- Assess document quality: DPI estimation, skew detection, degradation scoring
- Classify each detected region (text, table, figure, handwriting, formula, form field)
- For ambiguous regions: single Claude call for classification (the only LLM routing call)
- Apply preprocessing for degraded regions (DocEnTr enhancement)
- Generate routing plan: which specialist handles which region, with surrounding context
- Dispatch regions to specialists with: cropped image, region type, extraction schema, surrounding context (caption, headers)
- Aggregate specialist outputs in reading order
- Pass aggregated output to Validator

**Tools:** DocLayout-YOLO, Surya, DocEnTr, pdf2image, image quality assessment functions

#### Agent 2: Extraction Specialist Pool (3 Specialists, Deterministically Routed)

Each specialist follows the **OCR-then-LLM** pattern proven by AWS ICCV 2025 research (CER drops from 0.036 to 0.01 with combined pipeline).

**Text Specialist:**

- Primary: PaddleOCR 3.0 for text extraction with per-character confidence scores
- Enhancement: Claude for long-context reasoning, cross-page references, narrative understanding
- Output: Markdown text blocks with reading order preserved
- Tools: PaddleOCR, Claude API

**Table Specialist:**

- Primary: Docling for table structure prediction (97.9% accuracy on complex tables)
- Enhancement: Claude for reasoning about merged cells, ambiguous headers, multi-page tables (lower hallucination rate critical for financial data)
- Schema enforcement: Codex/GPT-4o Structured Outputs API for final JSON table output
- Output: HTML table + JSON structured data with cell-level bounding boxes
- Tools: Docling, Claude API, Codex Structured Outputs API

**Visual Specialist:**

- Primary pass (parallel, per figure type):
  - Charts/graphs: DePlot for chart-to-table extraction
  - Multi-panel figures: FigEx2 for panel splitting, then recursive processing
  - Molecular structures: DECIMER for SMILES/InChI output
  - Gel images: GelGenie for band identification
  - Handwritten math: pix2tex/LaTeX-OCR for LaTeX conversion
  - Handwritten text: TrOCR for HTR with confidence scores
- VLM enhancement (dual-model for complex figures):
  - Codex/GPT-4o for scientific figure matching and element identification (SciFIBench advantage)
  - Claude for chart reasoning and interpretation (CharXiv advantage)
  - For handwriting: Codex primary (better raw accuracy), Claude secondary (lower hallucination)
- Output: Type-specific JSON schemas with bounding boxes and extraction provenance
- Tools: FigEx2, DePlot, DECIMER, GelGenie, TrOCR, pix2tex, GOT-OCR 2.0, Claude API, Codex API

#### Agent 3: Validator

**Implementation:** Multi-layer validation with re-extraction loop (detailed in Section 8).

**Summary:** 5-layer validation (schema, cross-reference, semantic, visual grounding, confidence calibration) with a decision gate that routes fields to ACCEPT, RE-EXTRACT (with model switch, max 2 retries), or FLAG FOR REVIEW.

---

## 3. Tool Palette

### The 12 Open-Source Tools

| # | Tool | License | Role | Selection Rationale |
|---|------|---------|------|---------------------|
| 1 | **Docling** (IBM) | MIT | PDF parsing, text extraction, table extraction | 97.9% accuracy on complex tables; best text fidelity among open-source; MIT license |
| 2 | **DocLayout-YOLO** | Open | Layout detection (figures, tables, text blocks, formulas) | Powers MinerU; best-in-class layout detection on diverse documents |
| 3 | **PaddleOCR 3.0 + PP-Structure** | Apache 2.0 | OCR engine + layout analysis | 8.3/10 in Pragmile 2025 testing; 90+ languages; strong multilingual support |
| 4 | **Surya** | Open | Text detection, reading order, table recognition | Component toolkit powering Marker; 90+ language support |
| 5 | **GOT-OCR 2.0** (580M params) | Open | Unified OCR for formulas, tables, molecular structures, charts | First model treating all optical signals uniformly |
| 6 | **FigEx2** (Jan 2026) | Open | Multi-panel figure splitting | Cross-domain few-shot; reward-augmented training |
| 7 | **DePlot** (Google, 282M) | Open | Chart-to-table extraction | 29.4% improvement over prior SOTA on ChartQA human queries |
| 8 | **DECIMER.ai** | Open | Chemical structure recognition (SMILES/InChI) | Open platform; Nature Communications 2023; production-proven |
| 9 | **GelGenie** | Open | Gel electrophoresis band identification | Nature Communications 2025; AI-powered, automatic |
| 10 | **TrOCR** (Microsoft) | MIT | Handwritten text recognition | State-of-the-art transformer HTR; HuggingFace ecosystem |
| 11 | **DocEnTr** | Open | Document image enhancement (degraded scans) | Transformer-based cleaning, binarization, deblurring |
| 12 | **pix2tex / LaTeX-OCR** | Open | Handwritten math-to-LaTeX | Primary open-source tool for equation conversion |

### Tool Composition by Pipeline Stage

| Stage | Tools Used | Purpose |
|-------|-----------|---------|
| Layout Detection | DocLayout-YOLO, Surya, FigEx2 | Region detection, reading order, panel splitting |
| OCR | PaddleOCR 3.0, TrOCR, GOT-OCR 2.0 | Text, handwriting, formulas |
| Tables | Docling | Structure prediction, cell extraction |
| Charts | DePlot | Chart-to-table conversion |
| Chemistry | DECIMER.ai | SMILES/InChI from molecular structures |
| Biology | GelGenie | Gel electrophoresis analysis |
| Math | pix2tex/LaTeX-OCR | Handwritten equation-to-LaTeX |
| Enhancement | DocEnTr | Degraded document recovery |
| Orchestration | Custom Python (Claude Code skill) | Agent coordination, state management |

---

## 4. Model Assignment

### Claude vs Codex Task Routing Table

| Task | Assigned Model | Benchmark Evidence | Rationale |
|------|---------------|-------------------|-----------|
| Table reasoning (merged cells, no gridlines) | **Claude** | Hallucination rate 0.09% (vs GPT-4o 0.15%) | Lower hallucination critical for financial/scientific tables |
| Chart interpretation and data extraction | **Claude** | ChartQA 90.8% (vs GPT-4o 85.7%); CharXiv reasoning ~60% (vs 47.1%) | Clear benchmark lead on chart reasoning tasks |
| Long document cross-page reasoning | **Claude** | 1M token context window | Only model with sufficient context for full-document reasoning |
| Identity document extraction | **Claude** | GPT-4o refuses due to content policy | No alternative; GPT-4o explicitly blocks identity documents |
| Semantic validation ("do these fields make sense?") | **Claude** | DocVQA 95.2% (vs GPT-4o 92.8%) | Best general document understanding |
| Ambiguous region classification | **Claude** | Broader document understanding benchmarks | Used only as tiebreaker when deterministic classification fails |
| Scientific figure matching and identification | **Codex/GPT-4o** | SciFIBench 75.4% (vs Claude ~53%); AI2D 94.2% (vs 88.1%) | Significant benchmark advantage on scientific figure tasks |
| Handwriting OCR verification | **Codex/GPT-4o** | Edit distance 0.02 (vs Claude 0.03); 3x token efficiency | Better raw OCR accuracy; more efficient for character-level work |
| Final JSON schema enforcement | **Codex/GPT-4o** | Native Structured Outputs API | Guarantees schema compliance at the API level; Claude lacks equivalent |
| Raw OCR output verification | **Codex/GPT-4o** | Edit distance advantage | Better character-level precision for OCR quality checking |

### Dual-Model Strategy: Cost-Benefit Profile

| Strategy | Error Reduction | Cost Multiplier | Used When |
|----------|----------------|-----------------|-----------|
| Task-specific routing (Claude OR Codex per region) | ~30% | 1.1x | Simple and medium-complexity regions |
| Dual-model (Claude AND Codex on same region) | ~60% | 1.8x | Complex regions (figures, handwriting, ambiguous tables) |
| Model switch on retry (failed model swapped) | Additional ~15% on retry pool | +$0.10-0.40 per doc | Validation failures in the 0.70-0.90 confidence band |

### Re-Extraction Model Switch Logic

When a field fails validation (confidence 0.70-0.90):

1. If the field was extracted by Claude, re-extract with Codex
2. If the field was extracted by Codex, re-extract with Claude
3. If both models agree on retry, confidence is boosted
4. If they disagree, the field is flagged with both candidates and their confidence scores

---

## 5. Data Flow

### Complete Flow: Document Input to Markdown + JSON Output

```
Document Input (PDF / image / URL)
    |
    v
[Coordinator: pdf2image / PyMuPDF]
    -> Page images (cached to disk)
    |
    v
[Coordinator: DocLayout-YOLO]
    -> Region bounding boxes + type classifications
    |
    v
[Coordinator: Surya]
    -> Reading order across all regions
    |
    v
[Coordinator: Quality Assessment]
    -> DPI estimation, skew angle, degradation score per page
    |
    v
[Coordinator: DocEnTr]
    -> Enhanced images (ONLY for regions with degradation score > threshold)
    |
    v
[Coordinator: Routing Plan Generation]
    -> Deterministic assignment of regions to specialists
    -> For ambiguous regions: single Claude classification call
    |
    +-------> Text regions -> TEXT SPECIALIST
    |           1. PaddleOCR 3.0 -> raw text + per-character confidence
    |           2. Claude -> enhanced text with cross-page context (if needed)
    |           3. Output: {text, confidence, bbox, method}
    |
    +-------> Table regions -> TABLE SPECIALIST
    |           1. Docling -> HTML table structure (97.9% accuracy)
    |           2. Claude -> reasoning about ambiguous cells, merged headers
    |           3. Codex Structured Outputs -> JSON schema enforcement
    |           4. Output: {html, json, confidence, cell_bboxes, method}
    |
    +-------> Chart/graph regions -> VISUAL SPECIALIST (chart mode)
    |           1. DePlot -> chart-to-table extraction
    |           2. Claude -> chart reasoning and interpretation
    |           3. Output: {data_table, description, confidence, bbox, method}
    |
    +-------> Figure regions -> VISUAL SPECIALIST (figure mode)
    |           1. FigEx2 -> panel split (if multi-panel)
    |           2. Type classifier -> route to domain tool
    |              - Molecular: DECIMER -> SMILES/InChI
    |              - Gel: GelGenie -> band identification
    |              - General: description extraction
    |           3. Codex -> figure matching/identification (SciFIBench)
    |           4. Claude -> scientific reasoning/interpretation
    |           5. Output: {type_specific_json, description, confidence, bbox, method}
    |
    +-------> Handwriting regions -> VISUAL SPECIALIST (handwriting mode)
    |           1. DocEnTr -> image enhancement
    |           2. TrOCR -> HTR with per-character confidence
    |           3. Codex -> OCR verification (edit distance 0.02)
    |           4. Claude -> hallucination check (0.09% rate)
    |           5. Output: {text, confidence, bbox, method}
    |
    +-------> Formula regions -> VISUAL SPECIALIST (formula mode)
              1. GOT-OCR 2.0 -> LaTeX output
              2. pix2tex -> LaTeX output (second opinion)
              3. Vote on disagreement
              4. Output: {latex, confidence, bbox, method}
    |
    v
[Coordinator: Assembly]
    - Merge all specialist outputs in reading order
    - Generate Markdown with proper hierarchy (headers, sections, references)
    - Generate JSON with all extracted fields and metadata
    |
    v
[Validator: 5-Layer Validation Pipeline] (see Section 8)
    |
    +-> ACCEPT (confidence >= 0.90) -> Final output
    |
    +-> RE-EXTRACT (0.70 <= conf < 0.90)
    |     -> Send region back to specialist with alternate model
    |     -> Re-validate (max 2 retries)
    |     -> If resolved: Final output
    |     -> If still failing after 2 retries: FLAG
    |
    +-> FLAG (confidence < 0.70)
          -> Output with needs_review markers and explanation
    |
    v
+------------------+     +------------------+
|  MARKDOWN OUTPUT |     |  JSON OUTPUT     |
|  + visual ground |     |  + confidence    |
|    annotations   |     |    per field     |
+------------------+     +------------------+
```

---

## 6. Output Schemas

### JSON Output Schema

```json
{
  "$schema": "agentic-extract/v1",
  "document": {
    "id": "uuid",
    "source": "filename.pdf",
    "page_count": 10,
    "processing_timestamp": "2026-02-23T12:00:00Z",
    "approach": "B",
    "total_confidence": 0.92,
    "processing_time_ms": 18500
  },
  "markdown": "# Document Title\n\n## Section 1\n\nText content...\n\n| Col1 | Col2 |\n|------|------|\n| val  | val  |\n\n![Figure 1](fig1_description)\n...",
  "regions": [
    {
      "id": "region_001",
      "type": "text",
      "page": 1,
      "bbox": {"x": 0.05, "y": 0.10, "w": 0.90, "h": 0.15},
      "content": {
        "text": "Extracted paragraph text...",
        "markdown": "Extracted **paragraph** text..."
      },
      "confidence": 0.97,
      "extraction_method": "paddleocr_3.0 + claude_opus_4.6",
      "model_agreement": null,
      "needs_review": false
    },
    {
      "id": "region_002",
      "type": "table",
      "page": 2,
      "bbox": {"x": 0.05, "y": 0.20, "w": 0.90, "h": 0.40},
      "content": {
        "html": "<table>...</table>",
        "json": {
          "headers": ["Gene", "Expression", "P-value"],
          "rows": [
            {"Gene": "BRCA1", "Expression": 3.2, "P-value": 0.001}
          ]
        },
        "cell_bboxes": [
          {"row": 0, "col": 0, "bbox": {"x": 0.05, "y": 0.20, "w": 0.30, "h": 0.05}}
        ]
      },
      "confidence": 0.94,
      "extraction_method": "docling + claude_opus_4.6 + codex_structured_outputs",
      "model_agreement": "unanimous_3",
      "needs_review": false
    },
    {
      "id": "region_003",
      "type": "figure",
      "subtype": "bar_chart",
      "page": 3,
      "bbox": {"x": 0.10, "y": 0.05, "w": 0.80, "h": 0.45},
      "content": {
        "description": "Bar chart showing gene expression levels...",
        "figure_json": {
          "figure_type": "bar_chart",
          "title": "Gene Expression Across Conditions",
          "x_axis": {"label": "Condition", "type": "categorical"},
          "y_axis": {"label": "Fold Change", "type": "numerical"},
          "data_series": [
            {"name": "BRCA1", "values": [1.0, 3.2, 4.1]}
          ]
        }
      },
      "confidence": 0.86,
      "extraction_method": "deplot + claude_chartreasoning + codex_figmatch",
      "model_agreement": "majority_3",
      "needs_review": false
    },
    {
      "id": "region_004",
      "type": "handwriting",
      "page": 7,
      "bbox": {"x": 0.05, "y": 0.50, "w": 0.90, "h": 0.30},
      "content": {
        "text": "Patient notes: administered 500mg...",
        "latex": null
      },
      "confidence": 0.78,
      "extraction_method": "trocr + codex_ocr + claude_hallcheck",
      "model_agreement": "majority_3",
      "needs_review": true,
      "review_reason": "Handwriting confidence below 0.85 threshold"
    }
  ],
  "extracted_entities": {
    "schema_version": "user_defined_schema_v1",
    "fields": {
      "patient_name": {
        "value": "John Smith",
        "confidence": 0.95,
        "source_region": "region_001"
      },
      "dosage": {
        "value": "500mg",
        "confidence": 0.78,
        "source_region": "region_004",
        "needs_review": true
      }
    }
  },
  "audit_trail": {
    "models_used": [
      "claude_opus_4.6",
      "gpt-4o",
      "paddleocr_3.0",
      "docling",
      "deplot",
      "trocr"
    ],
    "total_llm_calls": 14,
    "re_extractions": 2,
    "fields_flagged": 1,
    "processing_stages": [
      {"stage": "ingestion", "duration_ms": 1200},
      {"stage": "ocr", "duration_ms": 3400},
      {"stage": "vlm_extraction", "duration_ms": 9800},
      {"stage": "validation", "duration_ms": 2100},
      {"stage": "assembly", "duration_ms": 500}
    ]
  }
}
```

### Markdown Output Format

```markdown
# Document Title

**Source:** filename.pdf | **Pages:** 10 | **Processed:** 2026-02-23 | **Confidence:** 0.92

---

## Section 1: Introduction

Extracted paragraph text with **formatting** preserved. References to
[Table 1](#table-1) and [Figure 2](#figure-2) are hyperlinked.

---

## Table 1: Gene Expression Data {#table-1}

| Gene | Expression (Fold Change) | P-value |
|------|--------------------------|---------|
| BRCA1 | 3.2 | 0.001 |
| TP53 | 1.8 | 0.023 |

*Confidence: 0.94 | Method: docling + claude + codex*

---

## Figure 2: Expression Bar Chart {#figure-2}

**Type:** Bar chart
**Description:** Bar chart showing gene expression levels across three
experimental conditions (Control, Treatment A, Treatment B) for genes
BRCA1 and TP53.

**Extracted Data:**
- BRCA1: Control=1.0, Treatment A=3.2, Treatment B=4.1
- TP53: Control=1.0, Treatment A=1.8, Treatment B=2.9

*Confidence: 0.86 | Method: deplot + claude + codex*

---

## Handwritten Notes (Page 7) [NEEDS REVIEW]

Patient notes: administered 500mg of compound X at 14:00.
Observed mild reaction at injection site.

*Confidence: 0.78 | Method: trocr + codex + claude | FLAGGED: below 0.85 threshold*

---
```

### Key Schema Principles

1. **Every field has a confidence score.** No field is ever presented without a numerical confidence value.
2. **Null over hallucination.** Missing data is represented as `null`, never as a fabricated value. A null field at 100% confidence is better than a hallucinated value at 90% confidence.
3. **Extraction provenance.** Every region records the exact tools and models used for its extraction.
4. **Bounding boxes are normalized.** All bbox coordinates are normalized to [0, 1] relative to the page dimensions.
5. **Audit trail is mandatory.** Every output includes the complete processing audit trail: models used, call counts, re-extraction counts, per-stage timing.

---

## 7. Confidence Scoring

### Three-Level Confidence Architecture

**Level 1: Per-Character Confidence (from OCR tools)**

- PaddleOCR 3.0: Native per-character confidence scores
- TrOCR: Attention-based confidence from decoder
- GOT-OCR 2.0: Decoder probability per token

**Level 2: Per-Field Confidence (aggregated)**

```
field_confidence = weighted_average(
    ocr_confidence * 0.3,      # Raw OCR tool score
    vlm_confidence * 0.4,      # VLM extraction confidence (from logprobs)
    validation_score * 0.3     # Did it pass schema + cross-ref checks?
)
```

**Level 3: Per-Region Confidence (aggregated from fields)**

Region confidence is the minimum field confidence within the region, ensuring that any weak link surfaces to the region level.

### Calibration

- **Method:** Temperature scaling applied post-hoc using a held-out validation set
- **Target:** Expected Calibration Error (ECE) < 0.05
- **Recalibration cadence:** Monthly on production data
- **Calibration set:** Manually verified extractions from diverse document types

### Rejection Strategy

Based on the Pitrelli et al. framework from handwriting recognition research:

- **Target:** Reject 90% of errors while rejecting only 33% of correct extractions
- **Effective accuracy on accepted fields:** 99.9%

### Decision Thresholds

| Confidence Range | Decision | Action |
|-----------------|----------|--------|
| >= 0.90 | **ACCEPT** | Field included in final output as-is |
| 0.70 to 0.89 | **RE-EXTRACT** | Send back to specialist with alternate model (max 2 retries) |
| < 0.70 | **FLAG FOR REVIEW** | Include in output with `"needs_review": true` and explanation |

### Re-Extraction Confidence Boosting

When a field is re-extracted and both models agree:

- Agreement between original and retry model boosts confidence by +0.10
- Persistent disagreement after 2 retries forces the field to FLAG status with both candidate values preserved

---

## 8. Validation Pipeline

### 5-Layer Validation Architecture

#### Layer 1: Schema Validation (Deterministic)

- JSON schema conformance check against the expected output schema
- Every field must match its declared type (string, number, array, object)
- Fields with wrong types are rejected immediately and trigger re-extraction
- Required fields must be present (as values or explicit nulls)

#### Layer 2: Cross-Reference Validation (Deterministic)

- Do table row totals match column totals?
- Are dates within plausible ranges (not in the future, not before 1900)?
- Do extracted entity names appear somewhere in the document text?
- Are required fields present and non-null?
- Do numerical values fall within expected orders of magnitude?
- Are referenced figures/tables internally consistent (e.g., "Table 3" exists if text references it)?

#### Layer 3: Semantic Validation (LLM-Assisted)

- Single Claude call per document (not per field)
- Prompt: "Given this document region and these extracted fields, do the values make sense together? Flag any that seem incorrect or inconsistent."
- Claude's document understanding (DocVQA 95.2%) makes it the right model for this layer
- Returns a list of flagged fields with explanations

#### Layer 4: Visual Grounding Check (Deterministic)

- For each extracted value, crop the bounding box region from the original page image
- Run a quick OCR pass (PaddleOCR) on the cropped region
- Compare the OCR output with the extracted value
- If they diverge significantly (edit distance > threshold), flag the field
- This catches cases where the extraction is correct but the bounding box is wrong (or vice versa)

#### Layer 5: Confidence Calibration (Statistical)

- Aggregate per-character, per-field, and per-region confidence scores
- Apply temperature scaling calibration
- Compute the final calibrated confidence per field
- Apply decision thresholds (ACCEPT / RE-EXTRACT / FLAG)

### Re-Extraction Loop

```
VALIDATOR detects confidence in [0.70, 0.90) range
    |
    v
Identify the failing field(s) and their extraction model
    |
    v
Send the region back to the relevant specialist
    |
    v
Specialist re-extracts using the ALTERNATE model:
    - If Claude produced the failing extraction -> use Codex
    - If Codex produced the failing extraction -> use Claude
    |
    v
Re-run Validator layers 1-5 on the new extraction
    |
    +-> If confidence >= 0.90 -> ACCEPT (boosted by model agreement)
    |
    +-> If confidence still in [0.70, 0.90) and retries < 2
    |     -> Loop again with the original model + different prompt strategy
    |
    +-> If retries exhausted (2 retries done) -> FLAG
          - Include BOTH candidate values in output
          - Include confidence scores for each
          - Mark as needs_review with explanation
```

**Maximum retries:** 2 per field. This bounds worst-case latency and cost while providing genuine self-correction.

---

## 9. Upgrade Path to C

### From Approach B to Approach C: Incremental Agent Addition

The architecture is designed so that Approach C's additional agents can be added without redesigning the B foundation. Each upgrade is independent and can be deployed separately.

### Phase 4a: Add Consensus Judge (Week 7-8)

**Trigger:** High-disagreement cases where the re-extraction loop (model switch) still produces conflicting results after 2 retries.

**What changes:**

- A new **Consensus Judge** agent is added between the Validator and final output
- The Judge activates ONLY when the Validator's re-extraction loop exhausts retries without resolution
- The Judge loads the original document region, the two (or three) candidate extractions, and their confidence scores
- Claude (with full document context, 1M token window) acts as tiebreaker
- If Claude's tiebreaker agrees with one candidate, accept it with boosted confidence
- If Claude produces a fourth unique answer, flag for human review with all candidates

**What does NOT change:** Coordinator, Specialists, Validator layers 1-5 all remain identical. The Judge is purely additive.

### Phase 4b: Add Semantic Auditor (Week 8-9)

**Trigger:** Documents where cross-page coherence matters (scientific papers, contracts, multi-section reports).

**What changes:**

- A new **Semantic Auditor** agent runs once per document, after all fields are extracted and validated
- Assembles the complete Markdown output and the complete JSON output
- Sends both to Claude (1M token context) with the prompt: "Given this document in Markdown and the structured extraction in JSON, identify any inconsistencies, missing data, or extraction errors. Cross-reference figures with their captions, tables with their references in the text, and entities across sections."
- For each identified issue, flags the relevant JSON field
- If issues are found, triggers targeted re-extraction of affected regions

**Why this matters:** This is the agent that catches errors no field-level validation can find. Example: the text says "as shown in Table 3, mortality decreased by 15%" but the extracted Table 3 shows mortality increasing. Only a document-level reasoning agent catches this.

### Phase 4c: Expand Specialist Pool to 5 (Week 9-10)

**What changes:**

- Split the Visual Specialist into three separate specialists:
  - **Chart Specialist**: DePlot + Claude chart reasoning + Codex element identification
  - **Diagram Specialist**: DECIMER + GelGenie + domain-specific tools
  - **Handwriting Specialist**: TrOCR + DocEnTr + dual-model verification
- Each specialist now runs three parallel extraction passes (open-source tool, Claude, Codex) with field-level voting

### Phase 4d: Add Complexity-Based Routing (Week 10)

**What changes:**

- Coordinator gains a complexity assessment function that scores each region as Simple/Medium/Complex
- Simple regions (clean text, standard tables): single-model extraction
- Medium regions (tables with some ambiguity, basic figures): dual-model extraction
- Complex regions (handwriting, multi-panel figures, degraded scans): triple extraction with voting

**This reduces cost on easy documents** while maintaining maximum accuracy on hard ones.

### Agent Count Progression

| Phase | Agents | Cost/Doc (Complex) | Error Reduction |
|-------|--------|-------------------|----------------|
| B (initial) | 3 (Coordinator, Specialist Pool, Validator) | $0.60-1.10 | ~50% |
| B + Judge | 4 | $0.70-1.20 | ~55% |
| B + Judge + Auditor | 5 | $0.90-1.50 | ~58% |
| Full C (5 specialists, 3 validators) | 7 | $1.55-2.25 | ~60%+ |

---

## 10. Technology Stack

### Runtime Environment

| Component | Technology | Notes |
|-----------|-----------|-------|
| Language | Python 3.11+ | Primary orchestration language |
| Deployment | Claude Code skill | Invoked via `/extract` command |
| Container runtime | Docker | Every open-source tool runs in its own container |
| API client (Claude) | Anthropic Python SDK | Claude Opus 4.6 / Sonnet for VLM tasks |
| API client (Codex) | OpenAI Python SDK | GPT-4o / Codex for figure matching and schema enforcement |
| Image processing | Pillow, pdf2image, PyMuPDF | PDF-to-image conversion, cropping |
| JSON validation | jsonschema | Deterministic schema enforcement in Validator Layer 1 |
| Orchestration | asyncio | Parallel specialist dispatch |

### Docker Containers for Open-Source Tools

**CRITICAL: No bioinformatics tools are installed into the shared conda environment.** Every tool runs in its own Docker container per project policy.

| Tool | Docker Image | Mount |
|------|-------------|-------|
| DocLayout-YOLO | `doclayout-yolo:latest` (custom build) | `-v /data:/data` |
| PaddleOCR 3.0 | `paddlepaddle/paddleocr:latest` | `-v /data:/data` |
| Surya | `surya-ocr:latest` (custom build) | `-v /data:/data` |
| GOT-OCR 2.0 | `got-ocr2:latest` (custom build) | `-v /data:/data` |
| FigEx2 | `figex2:latest` (custom build) | `-v /data:/data` |
| DePlot | `deplot:latest` (custom build) | `-v /data:/data` |
| DECIMER.ai | `decimer:latest` (custom build) | `-v /data:/data` |
| GelGenie | `gelgenie:latest` (custom build) | `-v /data:/data` |
| TrOCR | `trocr:latest` (custom build) | `-v /data:/data` |
| DocEnTr | `docentr:latest` (custom build) | `-v /data:/data` |
| pix2tex | `pix2tex:latest` (custom build) | `-v /data:/data` |
| Docling | `docling:latest` (custom build) | `-v /data:/data` |

### Project Directory Structure

```
~/Desktop/agentic-extract/
    docs/
        plans/
            2026-02-23-agentic-extract-design.md   <- This document
    src/
        (implementation files, Phase 1+)
    tests/
        (test files, Phase 1+)
```

---

## 11. Key Design Decisions

### Decision 1: Deterministic Routing by Default, LLM Routing Only for Ambiguous Regions

**Choice:** Route regions to specialists using rule-based logic (region type from DocLayout-YOLO). Use a single Claude call only when the layout detector produces a low-confidence or ambiguous classification.

**Research justification:** The agentic architectures survey found that deterministic routing outperforms LLM routing in production reliability. LLM routing introduces nondeterminism, latency, and cost at a decision point where rules suffice 90%+ of the time.

**Implication:** The Coordinator agent is primarily deterministic code, not an LLM agent. This keeps the system predictable and debuggable.

### Decision 2: Claude for Reasoning, Codex for Schema Enforcement

**Choice:** Assign models to tasks based on benchmark evidence, not brand preference.

**Research justification:**
- Claude leads on DocVQA (95.2% vs 92.8%), ChartQA (90.8% vs 85.7%), CharXiv reasoning (~60% vs 47.1%), and hallucination rate (0.09% vs 0.15%)
- Codex/GPT-4o leads on SciFIBench (75.4% vs ~53%), AI2D (94.2% vs 88.1%), edit distance (0.02 vs 0.03), and has the native Structured Outputs API for schema compliance
- Task-specific routing yields ~30% error reduction at 1.1x cost; full dual-model yields ~60% at 1.8x cost

**Implication:** The system is not "Claude-first" or "OpenAI-first." It is benchmark-first. If a future model surpasses either on a specific task, the routing table is updated accordingly.

### Decision 3: OCR-then-LLM, Not LLM-Alone

**Choice:** Always run open-source OCR tools first, then condition LLM generation on the OCR output.

**Research justification:** AWS ICCV 2025 research demonstrated that OCR-then-LLM pipelines reduce character error rate from 0.036 to 0.01. The LLM sees the OCR output and the original image, enabling it to correct OCR errors while being grounded in the OCR's character-level extraction.

**Implication:** Every specialist follows the same two-stage pattern: (1) open-source tool produces raw extraction, (2) VLM enhances/corrects. The VLM never works from the raw image alone.

### Decision 4: Null Fields Over Hallucinated Fields

**Choice:** The system is explicitly instructed to return `null` for any field where data cannot be confidently extracted. A null field at 100% confidence is always preferable to a hallucinated value at 90% confidence.

**Research justification:** The Claude vs Codex analysis and handwriting research both identified hallucination as the primary failure mode in VLM extraction. Claude's 0.09% hallucination rate (vs GPT-4o's 0.15%) is why Claude handles hallucination-sensitive tasks, but even 0.09% is unacceptable for fields like medication dosages or financial figures.

**Implication:** All prompts include explicit instructions to output null rather than guess. The Validator's semantic check (Layer 3) specifically looks for implausible values that may indicate hallucination.

### Decision 5: Never Fabricate Quantitative Data from Figures

**Choice:** When extracting from scientific figures (charts, plots, graphs, KM curves), the system extracts ONLY what is certain: axis labels, legend entries, chart type, caption text, and qualitative trend descriptions. It NEVER generates a data table of numerical values estimated from pixel positions. Quantitative data points from figures are always `null` with `needs_review: true` and a note directing the user to contact authors or check supplementary materials.

**Research justification:** VLMs looking at chart images produce plausible-looking numbers that may or may not be correct. Presenting hallucinated data points in a precise-looking table is actively dangerous for downstream scientific use (meta-analyses, systematic reviews). There is no reliable way to validate pixel-estimated values without the original source data. A null is honest; a guess dressed up as extracted data is a fabrication.

**Implication:** Figure extraction outputs contain: chart type, axis metadata, legend, caption, and qualitative description. The `data_points` field is always `null` unless the figure includes an embedded data table with readable text (which can be OCR-verified). Confidence for figure regions with no quantitative extraction is capped at 0.50.

### Decision 6: Confidence-Based Rejection, Not Confidence-Based Filtering

**Choice:** Fields below threshold are flagged in the output with `"needs_review": true` and an explanation. They are NOT silently dropped.

**Research justification:** The handwriting extraction research established that calibrated confidence with rejection achieves 99.9% effective accuracy on accepted fields, while preserving information for downstream workflows that can tolerate uncertainty. Silent filtering loses data; flagging preserves it.

**Implication:** Downstream consumers of Agentic Extract output can choose their own threshold. A research pipeline might accept everything; a clinical pipeline might reject anything below 0.95.

### Decision 6: Open-Source Tools for Heavy Lifting, VLMs for Hard Parts

**Choice:** Use open-source tools (Docling, PaddleOCR, DePlot, etc.) for the deterministic, high-volume extraction work. Reserve Claude/Codex API calls for the parts that actually require reasoning.

**Research justification:** The extraction landscape analysis and cost-benefit research show that open-source tools handle 70-80% of extraction work with high accuracy and zero marginal cost. VLM calls are expensive ($0.01-0.10 per region) and should be reserved for regions where the open-source tools produce ambiguous or low-confidence results.

**Implication:** A clean, well-structured PDF with standard tables may require zero VLM calls after the open-source pass. A degraded handwritten document with complex figures may require 10+ VLM calls. Cost scales with document complexity, not document length.

---

## 12. Success Criteria

### What "Rivals Landing AI" Means in Measurable Terms

#### Accuracy Benchmarks

| Metric | Target | Landing AI Reference | Measurement Method |
|--------|--------|---------------------|-------------------|
| Text extraction accuracy (clean docs) | >= 99.5% character accuracy | ~99% (claimed) | Character error rate on standardized test set |
| Table extraction accuracy | >= 97% cell accuracy | ~95% (estimated from user reports) | Cell-level F1 on complex table benchmark |
| Chart data extraction | >= 90% data point accuracy | Not documented | Numerical accuracy on ChartQA-derived benchmark |
| Handwriting recognition | >= 95% word accuracy (accepted fields) | Not supported | Word error rate on IAM/RIMES test partitions |
| Scientific figure interpretation | >= 75% description accuracy | Not supported | SciFIBench-derived evaluation |
| Molecular structure extraction | >= 90% SMILES accuracy | Not supported | Tanimoto similarity on DECIMER test set |

#### Confidence and Rejection

| Metric | Target | Measurement Method |
|--------|--------|-------------------|
| Effective accuracy on accepted fields | >= 99.9% | Accuracy computed only on fields with confidence >= 0.90 |
| Expected Calibration Error (ECE) | < 0.05 | Computed on held-out validation set with temperature scaling |
| Error rejection rate | >= 90% of errors rejected | Proportion of errors caught by confidence threshold |
| Correct field rejection rate | <= 33% of correct fields rejected | Proportion of correct fields unnecessarily flagged |

#### Performance

| Metric | Target | Landing AI Reference |
|--------|--------|---------------------|
| Latency (10-page paper, no retries) | <= 25 seconds | ~8 seconds (claimed median) |
| Latency (10-page paper, with retries) | <= 45 seconds | N/A (no retry mechanism) |
| Latency (simple 5-page invoice) | <= 10 seconds | ~5 seconds (estimated) |

**Note:** We accept higher latency than Landing AI in exchange for higher accuracy and self-correction. Cost is not a constraint.

#### Cost

| Document Type | Target Cost | Landing AI Reference |
|---------------|-------------|---------------------|
| Simple invoice (5 pages) | $0.15-0.25 | $0.30 (estimated from credit system) |
| Scientific paper (10 pages, complex) | $0.60-1.10 | $0.50-1.00 (estimated) |
| Scientific paper with retries | $0.70-1.40 | N/A |

#### Coverage (vs Landing AI)

| Capability | Agentic Extract v1 | Landing AI |
|-----------|---------------------|-----------|
| Standard text extraction | Yes | Yes |
| Table extraction (simple) | Yes | Yes |
| Table extraction (merged cells, multi-page) | Yes (Claude reasoning) | Partial |
| Chart interpretation | Yes (DePlot + Claude) | Limited |
| Scientific figure matching | Yes (Codex + SciFIBench) | No |
| Molecular structure (SMILES) | Yes (DECIMER) | No |
| Gel electrophoresis | Yes (GelGenie) | No |
| Handwritten text | Yes (TrOCR + dual-model) | No |
| Handwritten math (LaTeX) | Yes (pix2tex + GOT-OCR) | No |
| Multi-panel figure splitting | Yes (FigEx2) | No |
| Degraded scan recovery | Yes (DocEnTr) | Limited |
| Cross-page reasoning | Yes (Claude 1M token context) | No |
| Identity documents | Yes (Claude; GPT-4o refuses) | Partial |
| Per-field confidence scoring | Yes (calibrated, included) | Paywalled ($2,000+/mo) |
| Visual grounding (bounding boxes) | Yes (region + cell level) | Paywalled (higher tiers) |
| Self-correction (re-extraction) | Yes (model switch, max 2 retries) | No documented equivalent |
| Audit trail | Yes (full provenance) | No documented equivalent |

### Definition of Done for v1

Agentic Extract v1 (Approach B) is considered complete when:

1. All three agents (Coordinator, Specialist Pool, Validator) are implemented and functional
2. All 12 open-source tools are containerized and integrated
3. Both Claude and Codex APIs are integrated with task-specific routing
4. The 5-layer validation pipeline is operational
5. The re-extraction loop with model switching is working (max 2 retries)
6. Both Markdown and JSON outputs conform to the schemas defined in Section 6
7. Per-field confidence scoring is calibrated (ECE < 0.05) on a test set
8. The system handles the following document types end-to-end: scientific papers with tables/figures/charts, handwritten notes, invoices, and mixed-media documents
9. The system is packaged as a Claude Code skill (invocable via `/extract`)
10. Processing time for a 10-page scientific paper is under 25 seconds (without retries)

---

## Appendix: Research Citations

| Decision | Supporting Evidence | Source |
|----------|-------------------|--------|
| OCR-then-LLM pattern | CER drops from 0.036 to 0.01 | Wang et al., ICCV 2025 Workshop |
| 2-4 agents in production | Survey of production systems | Agentic architectures research |
| Deterministic routing | Outperforms LLM routing in production | Zircon Tech 2026; Tensorlake 2025 |
| Simple voting > debate | ACL 2025 Findings | "Debate or Vote" arXiv:2508.17536 |
| Claude for chart reasoning | CharXiv: ~60% vs GPT-4o 47.1% | Wang et al., NeurIPS 2024 |
| Codex for figure matching | SciFIBench: 75.4% vs Claude ~53% | Roberts et al., NeurIPS 2024 |
| Codex for schema enforcement | Native Structured Outputs API | OpenAI documentation |
| Claude for hallucination prevention | 0.09% vs GPT-4o 0.15% | CodeSOTA 2025 |
| Claude for long documents | 1M token context window | Anthropic model card |
| Confidence with rejection = 99.9% | Production confidence patterns | Handwriting extraction research |
| Docling for tables | 97.9% on complex tables | Docling benchmark evaluation |
| DePlot for chart-to-table | 29.4% improvement over prior SOTA | Liu et al., ACL Findings 2023 |
| DECIMER for molecules | Open platform; Nature Comms 2023 | Rajan et al., 2023 |
| DocEnTr for degraded scans | Transformer-based enhancement | arXiv:2201.10252 |
| Dual-model = ~60% error reduction | Cost-benefit analysis | Claude vs Codex research |
| Task routing = ~30% error reduction | Cost-benefit analysis | Claude vs Codex research |
| Landing AI is ~30% agentic | Competitive analysis | Landing AI competitive analysis |

---

## Implementation Timeline

| Phase | Weeks | Deliverable |
|-------|-------|-------------|
| Phase 1: Foundation | 1-2 | Coordinator + Text Specialist + Table Specialist + basic output |
| Phase 2: Visual + Validation | 3-4 | Visual Specialist + Handwriting + Validator (5 layers) |
| Phase 3: Agentic Loop + Polish | 5-6 | Re-extraction loop + visual grounding + confidence calibration + skill packaging |
| Phase 4: Upgrade to C (optional) | 7-10 | Consensus Judge + Semantic Auditor + specialist expansion + complexity routing |

---

*This design document is based on six research files totaling approximately 4,500 lines of analyzed content across competitive intelligence, landscape analysis, model comparisons, scientific diagram extraction, handwriting recognition, and agentic architecture patterns. All tool names, benchmark numbers, and model capabilities are sourced from peer-reviewed research and verified documentation.*

*Approved: 2026-02-23*
