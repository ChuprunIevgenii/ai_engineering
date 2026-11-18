# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pydantic-ai-slim[openrouter]==2.6.*",
#   "sentence-transformers",
#   "numpy",
#   "pypdf",
#   "rank_bm25",
# ]
# ///
"""Ask stages of the ladder a question, as a given student, and see their tool calls.

  uv run ask.py --stage 2 "Хто викладає дисципліну ПЗ-204?"
  uv run ask.py --stage 2,4 --id q10        # same question on several stages; ids from data/questions.json
  uv run ask.py --stage 6 --id q08,q13      # several questions; validator retries are shown with ↩
  uv run ask.py --stage 5-naive --as S-1041 "Порівняй мій середній бал із середнім балом студента S-1002."
  uv run ask.py -i                          # interactive: the corpus loads once, then type questions

Interactive lines: `2,4 Хто веде ПЗ-204?`, `4 q14` (question id), `as S-1187` (switch student), empty line to quit.
Needs OPENROUTER_API_KEY. Package: `rag` (override with RAG_PKG=...).
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import os
import re
import sys
import time
from pathlib import Path

HW_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(HW_DIR))
PKG = os.environ.get("RAG_PKG", "rag")


class Session:
    def __init__(self, student: str | None):
        self.config = importlib.import_module(f"{PKG}.config")
        self.tools = importlib.import_module(f"{PKG}.tools")
        self.stages = importlib.import_module(f"{PKG}.stages")
        self.questions = {q["id"]: q for q in
                          json.loads(self.config.QUESTIONS_FILE.read_text(encoding="utf-8"))}
        self.student = student
        print("loading corpus…", file=sys.stderr)
        self.corpus = self.tools.Corpus.load()
        self.agents: dict = {}
        self.loop = asyncio.new_event_loop()   # one loop for the session: agents keep their HTTP client

    def ask(self, stage_list: str, text: str) -> None:
        from pydantic_ai import capture_run_messages
        from pydantic_ai.messages import ModelResponse, RetryPromptPart, TextPart, ToolCallPart

        q = self.questions.get(text.strip())
        question = q["q"] if q else text
        student = self.student or (q or {}).get("student") or self.config.DEFAULT_STUDENT
        print(f"\n? {question}   (student {student})")
        for stage in stage_list.split(","):
            if stage not in self.stages.STAGES:
                print(f"unknown stage {stage!r}; known: {self.stages.STAGES}")
                continue
            agent = self.agents.setdefault(stage, self.stages.build_agent(stage))
            deps = self.tools.Deps(self.corpus, student_id=student)
            t0 = time.time()
            res, error = None, None
            with capture_run_messages() as messages:
                try:
                    res = self.loop.run_until_complete(self.stages.run(agent, stage, question, deps))
                except NotImplementedError as e:
                    print(f"\n[stage {stage}] TODO not implemented: {e}")
                    continue
                except Exception as e:   # noqa: BLE001 - show the trace of a failed run, it is what you debug
                    error = e
            cost, calls = 0.0, 0
            final = messages[-1] if res else None
            print(f"\n[stage {stage}]")
            for m in messages:
                for part in m.parts:
                    if isinstance(part, ToolCallPart):
                        print(f"  → {part.tool_name}({json.dumps(part.args_as_dict(), ensure_ascii=False)})")
                    elif isinstance(part, RetryPromptPart):
                        print(f"  ↩ {'tool ' + part.tool_name if part.tool_name else 'validator'}: {part.content}")
                    elif isinstance(part, TextPart) and isinstance(m, ModelResponse) and m is not final \
                            and part.content.strip():
                        print(f"  ✎ {part.content[:300]}")
                if isinstance(m, ModelResponse):
                    calls += 1
                    cost += float((m.provider_details or {}).get("cost") or 0)
            print(f"\n{res.output}\n" if res else f"\n{type(error).__name__}: {error}\n")
            print(f"  LLM calls {calls} · ${cost:.4f} · {time.time() - t0:.1f}s")


def interactive(s: Session) -> None:
    while True:
        try:
            line = input("\nask> ").strip()
        except EOFError:
            return
        if not line:
            return
        if m := re.fullmatch(r"as\s+(S-\d{4})", line, flags=re.I):
            s.student = m.group(1).upper()
            print(f"student {s.student}")
            continue
        stage_list, _, text = line.partition(" ")
        if not text:
            print("format: <stages> <question or id>, e.g. `2,4 q10`")
            continue
        s.ask(stage_list, text)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("question", nargs="?")
    p.add_argument("--stage", help="stage or comma list: 0 1 2 3 4 5 5-naive 6")
    p.add_argument("--as", dest="student", default=None, help="logged-in student id (default S-1041)")
    p.add_argument("--id", help="question id or comma list from data/questions.json: q10 or q08,q13")
    p.add_argument("-i", "--interactive", action="store_true")
    args = p.parse_args()
    if not os.environ.get("OPENROUTER_API_KEY"):
        sys.exit("OPENROUTER_API_KEY is not set.")
    if not args.interactive and not (args.stage and (args.question or args.id)):
        p.error("give --stage and a question (or --id), or use -i")
    ids = [i.strip() for i in args.id.split(",")] if args.id else []
    known = {q["id"] for q in json.loads(importlib.import_module(f"{PKG}.config").QUESTIONS_FILE.read_text("utf-8"))}
    if unknown := [i for i in ids if i not in known]:
        p.error(f"unknown question id {', '.join(unknown)}; ids are in data/questions.json, e.g. q10")
    s = Session(args.student)
    if args.interactive:
        interactive(s)
    for qid in ids:
        s.ask(args.stage, qid)
    if args.question and not ids:
        s.ask(args.stage, args.question)


if __name__ == "__main__":
    main()
