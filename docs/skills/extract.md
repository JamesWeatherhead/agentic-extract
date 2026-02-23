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
