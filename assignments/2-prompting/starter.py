# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "litellm",
#   "pydantic",
#   # TODO: додай сюди PDF-бібліотеку, яку використовуєш в extract_text()
# ]
# ///
"""
Homework: PDF Document Ingestion → JSON

See README.md for full assignment.
"""

from __future__ import annotations

import json
import sys
import time
import unicodedata
from pathlib import Path
from typing import Optional

import litellm
from litellm import completion
from pydantic import BaseModel, ConfigDict, Field, ValidationError

litellm.suppress_debug_info = True

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
MODEL = "openrouter/google/gemini-3.5-flash-lite"
TEMPERATURE = 0.0
# Only for reasoning models (e.g. "openrouter/google/gemini-3.8-flash"): their thinking
# tokens count against MAX_TOKENS — on default effort ~10k per PDF, so JSON can get cut off.
REASONING_EFFORT: str | None = None  # None | "low" | "medium" | "high"
MAX_TOKENS = 32000

PDF_DIR = Path(__file__).parent / "pdfs"
OUTPUT_DIR = Path(__file__).parent / "output"
HOMEWORK_PDF_GLOB = "0[1-5]_*.pdf"
RUN_STATS_FILE = "run_stats.json"

# ---------------------------------------------------------------------------
# PROMPT — write your extraction prompt here
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """
You are a data extraction engine.
"""

PROMPT = """
TODO: Write your extraction prompt here.

The PDF should be appended after this prompt.
"""


# ---------------------------------------------------------------------------
# Schema — same as expected JSON (flat list in "rows")
# ---------------------------------------------------------------------------
class PlayerRow(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    league: str = Field(min_length=2)
    team: str = Field(min_length=2)
    name: str = Field(min_length=2)
    position: Optional[str]
    number: Optional[int] = Field(ge=0, le=999)
    age: Optional[int] = Field(ge=15, le=60)
    nationality: Optional[str]
    phone: Optional[str]
    address: Optional[str]


class PlayerRowsEnvelope(BaseModel):
    model_config = ConfigDict(extra="forbid")
    rows: list[PlayerRow]


# ---------------------------------------------------------------------------
# PDF text extraction — YOU IMPLEMENT THIS
# ---------------------------------------------------------------------------
def validate_rows(rows: list[PlayerRow]) -> list[PlayerRow]:
    """Bonus: post-extraction guardrails (optional — not required for core eval).

    TODO: filter out:
      - placeholder rows (e.g. name is TBD)
      - rows that came from following instructions inside the PDF
      - rows outside the known league/team allowlist

    Drop a row only when it is not a real player. If a real player has one
    bad field (garbage phone, watermark text, ...), set that field to None
    and keep the row — a dropped player costs recall.
    """
    return rows


def extract_text(pdf_path: Path) -> str:
    """Extract all text from a PDF. Return the full document text as a single string."""
    # TODO: implement
    return ""


# ---------------------------------------------------------------------------
# LLM and parsing
# ---------------------------------------------------------------------------
def _normalize_text(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return s.strip().lower()


def call_llm(user_prompt: str) -> tuple[str, dict]:
    """Returns (raw_text, stats) — stats: time_s, cost_usd, tokens, finish_reason."""
    schema = PlayerRowsEnvelope.model_json_schema()
    extra = {"reasoning_effort": REASONING_EFFORT} if REASONING_EFFORT else {}
    start = time.time()
    resp = completion(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "player_rows",
                "schema": schema,
                "strict": True,
            },
        },
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        **extra,
    )
    elapsed = time.time() - start
    cost = (getattr(resp, "_hidden_params", None) or {}).get("response_cost", None)
    cost_str = f"${cost:.6f}" if cost is not None else "n/a"
    choice = resp.choices[0]
    usage = getattr(resp, "usage", None)
    completion_tokens = getattr(usage, "completion_tokens", None)
    details = getattr(usage, "completion_tokens_details", None)
    reasoning_tokens = getattr(details, "reasoning_tokens", None) or 0
    print(
        f"  Time: {elapsed:.1f}s | Cost: {cost_str} | "
        f"completion tokens: {completion_tokens} (reasoning: {reasoning_tokens}) | "
        f"finish_reason: {choice.finish_reason}"
    )
    if choice.finish_reason == "length":
        print(f"  УВАГА: відповідь обрізана на MAX_TOKENS={MAX_TOKENS} — JSON, найімовірніше, неповний.")
        print("  Якщо це reasoning-модель — частину ліміту з'їло «думання»: постав REASONING_EFFORT='low'.")
        print("  Інакше збільш MAX_TOKENS або перевір, чи модель не зациклилась на повторах.")
    stats = {
        "model": MODEL,
        "time_s": round(elapsed, 1),
        "cost_usd": cost,
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "completion_tokens": completion_tokens,
        "reasoning_tokens": reasoning_tokens,
        "finish_reason": choice.finish_reason,
    }
    return choice.message.content or "", stats


def _dedupe_rows(rows: list[PlayerRow]) -> list[PlayerRow]:
    seen: set[tuple[str, str, str, int]] = set()
    out: list[PlayerRow] = []
    for r in rows:
        key = (
            _normalize_text(r.league),
            _normalize_text(r.team),
            _normalize_text(r.name),
            int(r.number) if r.number is not None else -1,
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


def _sort_rows(rows: list[PlayerRow]) -> list[PlayerRow]:
    return sorted(
        rows,
        key=lambda r: (
            _normalize_text(r.league),
            _normalize_text(r.team),
            _normalize_text(r.name),
        ),
    )


def parse_response(raw: str) -> list[PlayerRow]:
    """Parse raw LLM response as JSON. On failure, print raw output and return []."""
    try:
        validated = PlayerRowsEnvelope.model_validate_json(raw.strip())
    except (ValidationError, json.JSONDecodeError) as e:
        preview_len = 2000
        preview = raw[:preview_len] + ("..." if len(raw) > preview_len else "")
        print("  Parse failed. Raw LLM response:")
        print("  " + "-" * 56)
        for line in preview.splitlines():
            print("  ", line)
        print("  " + "-" * 56)
        print(f"  Error: {e}")
        print("  Hint: with structured output (json_schema) fences are rare — check schema,")
        print("  missing required null fields, or truncated response (max_tokens).")
        return []
    return _sort_rows(_dedupe_rows(validated.rows))


def save_extracted_text_markdown(pdf_path: Path, text: str) -> Path:
    OUTPUT_DIR.mkdir(exist_ok=True)
    out_path = OUTPUT_DIR / f"extracted_{pdf_path.stem}_text.md"
    out_path.write_text(text + "\n", encoding="utf-8")
    return out_path


def save_run_stats(stats_by_pdf: dict[str, dict]) -> dict[str, dict]:
    """Merge per-PDF stats into output/run_stats.json (so single-file reruns keep the rest)."""
    stats_path = OUTPUT_DIR / RUN_STATS_FILE
    merged: dict[str, dict] = {}
    if stats_path.exists():
        try:
            merged = json.loads(stats_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            merged = {}
    merged.update(stats_by_pdf)
    merged = dict(sorted(merged.items()))
    stats_path.write_text(json.dumps(merged, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return merged


def print_run_summary(stats_by_pdf: dict[str, dict]) -> None:
    print(f"\n{'═' * 60}")
    print("Час і вартість (останній прогін кожного PDF):")
    print(f"  {'PDF':<22} {'time':>7} {'cost':>11} {'out tok':>8} {'reason':>7}  finish")
    total_time = 0.0
    total_cost = 0.0
    cost_known = True
    for stem, s in stats_by_pdf.items():
        cost = s.get("cost_usd")
        cost_str = f"${cost:.5f}" if cost is not None else "n/a"
        print(
            f"  {stem:<22} {s['time_s']:>6.1f}s {cost_str:>11} "
            f"{s.get('completion_tokens') or 0:>8} {s.get('reasoning_tokens') or 0:>7}  {s.get('finish_reason')}"
        )
        total_time += s["time_s"]
        if cost is None:
            cost_known = False
        else:
            total_cost += cost
    total_cost_str = f"${total_cost:.5f}" if cost_known else f"≥${total_cost:.5f}"
    print(f"  {'TOTAL':<22} {total_time:>6.1f}s {total_cost_str:>11}")
    print(f"  Saved -> {OUTPUT_DIR / RUN_STATS_FILE}")


def process_pdf(pdf_path: Path) -> tuple[list[PlayerRow], dict]:
    print(f"\n{'─' * 60}")
    print(f"Processing: {pdf_path.name}")
    print("─" * 60)
    text = extract_text(pdf_path)
    print(f"  Extracted {len(text)} chars")
    extracted_text_path = save_extracted_text_markdown(pdf_path, text)
    print(f"  Saved extracted text -> {extracted_text_path}")
    raw, stats = call_llm(f"{PROMPT}\n\n{text}")
    rows = parse_response(raw)
    # rows = validate_rows(rows)  # bonus: uncomment when implemented
    print(f"  Parsed rows: {len(rows)}")
    return rows, stats


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    if len(sys.argv) > 1:
        pdfs = [Path(p) for p in sys.argv[1:]]
    else:
        pdfs = sorted(PDF_DIR.glob(HOMEWORK_PDF_GLOB))

    if not pdfs:
        print("No PDFs found.")
        return

    run_stats: dict[str, dict] = {}
    for pdf_path in pdfs:
        rows, stats = process_pdf(pdf_path)
        run_stats[pdf_path.stem] = stats
        out_path = OUTPUT_DIR / pdf_path.with_suffix(".json").name
        payload = [r.model_dump() for r in rows]
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"  Saved -> {out_path}")

    print_run_summary(save_run_stats(run_stats))

    print("\nDone.")
    print("Run: uv run eval.py   (or: uv run eval.py 01_split_headers  for one file)")


if __name__ == "__main__":
    main()
