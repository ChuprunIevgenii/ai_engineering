# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pydantic-ai-slim[openrouter]==2.6.*",
#   "sentence-transformers",
#   "numpy",
#   "pypdf",
#   "rank_bm25",
#   "gradio>=6,<7",
# ]
# ///
"""Web chat over the ladder: pick a stage and a logged-in student, ask, see the tool calls.

  uv run app.py                      # http://127.0.0.1:7860
  uv run app.py --stage 4 --as S-1187 --port 7861

Every message is a separate question (no conversation history), exactly as eval and ask.py run it.
Needs OPENROUTER_API_KEY. Package: `rag` (override with RAG_PKG=...).
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

import gradio as gr
from pydantic_ai import capture_run_messages
from pydantic_ai.messages import (ModelRequest, ModelResponse, RetryPromptPart, TextPart, ToolCallPart,
                                  ToolReturnPart, UserPromptPart)

HW_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(HW_DIR))
PKG = os.environ.get("RAG_PKG", "rag")
config = importlib.import_module(f"{PKG}.config")
tools = importlib.import_module(f"{PKG}.tools")
stages = importlib.import_module(f"{PKG}.stages")

QUESTIONS = json.loads(config.QUESTIONS_FILE.read_text(encoding="utf-8"))
MAX_TOOL_TEXT = 3000


def block(title: str, text: str) -> gr.ChatMessage:
    if len(text) > MAX_TOOL_TEXT:
        text = text[:MAX_TOOL_TEXT] + f"\n… ще {len(text) - MAX_TOOL_TEXT} символів"
    return gr.ChatMessage(content=f"```text\n{text}\n```", metadata={"title": title, "status": "done"})


def trace(messages: list, stage: str, failed: bool = False) -> tuple[list[gr.ChatMessage], int, float]:
    """Tool calls, tool results, validator retries (and the rejected drafts) of one run as collapsible blocks."""
    out, calls, llm_calls, cost = [], {}, 0, 0.0
    for i, m in enumerate(messages):
        if isinstance(m, ModelResponse):
            llm_calls += 1
            cost += float((m.provider_details or {}).get("cost") or 0)
            calls.update({p.tool_call_id: p for p in m.parts if isinstance(p, ToolCallPart)})
            draft = "".join(p.content for p in m.parts if isinstance(p, TextPart)).strip()
            if draft and (failed or i < len(messages) - 1):
                out.append(block("✎ чернетка відповіді", draft))
            continue
        if not isinstance(m, ModelRequest):
            continue
        for p in m.parts:
            if isinstance(p, UserPromptPart) and stage == "1" and i == 0:
                out.append(block("📄 контекст, який підставив пошук", str(p.content)))
            elif isinstance(p, ToolReturnPart):
                call = calls.get(p.tool_call_id)
                args = json.dumps(call.args_as_dict(), ensure_ascii=False) if call else ""
                out.append(block(f"🔧 {p.tool_name}({args})", p.model_response_str()))
            elif isinstance(p, RetryPromptPart):
                who = f"tool {p.tool_name}" if p.tool_name else "валідатор"
                out.append(block(f"↩️ повтор: {who}", p.model_response()))
    return out, llm_calls, cost


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", default="5", choices=stages.STAGES, help="stage selected on start (default 5)")
    p.add_argument("--as", dest="student", default=config.DEFAULT_STUDENT, help="student selected on start")
    p.add_argument("--port", type=int, default=7860)
    args = p.parse_args()
    if not os.environ.get("OPENROUTER_API_KEY"):
        sys.exit("OPENROUTER_API_KEY is not set.")

    print("loading corpus…", file=sys.stderr)
    corpus = tools.Corpus.load()
    students = sorted(corpus.students)
    agents: dict = {}

    async def respond(message: str, history, student: str, stage: str):
        agent = agents.setdefault(stage, stages.build_agent(stage))
        deps = tools.Deps(corpus, student_id=student)
        t0 = time.time()
        with capture_run_messages() as messages:
            try:
                res = await stages.run(agent, stage, message, deps)
            except NotImplementedError as e:
                return f"Сходинка {stage}: TODO ще не реалізовано — {e}"
            except Exception as e:   # local debugging UI: show what broke instead of a generic error
                cause = f"\n\nОстання причина: {e.__cause__}" if e.__cause__ else ""
                msgs, _, _ = trace(messages, stage, failed=True)
                return [*msgs, gr.ChatMessage(content=f"Сходинка {stage} впала: {type(e).__name__}: {e}{cause}")]
        msgs, llm_calls, cost = trace(messages, stage)
        msgs.append(gr.ChatMessage(content=str(res.output)))
        msgs.append(gr.ChatMessage(
            content=f"_сходинка {stage} · студент {student} · LLM-викликів {llm_calls} · "
                    f"${cost:.4f} · {time.time() - t0:.1f} с_"))
        return msgs

    demo = gr.ChatInterface(
        respond,
        chatbot=gr.Chatbot(height=650, label="Чат"),
        additional_inputs=[
            gr.Dropdown(students, value=args.student, label="Залогінений студент"),
            gr.Dropdown(stages.STAGES, value=args.stage, label="Сходинка"),
        ],
        additional_inputs_accordion=gr.Accordion("Сходинка і студент", open=True),
        examples=[[q["q"], q.get("student", config.DEFAULT_STUDENT)] for q in QUESTIONS],
        example_labels=[f"{q['id']} · {q['type']}" for q in QUESTIONS],
        cache_examples=False,
        title="ВНУ RAG — ручний тест сходинок",
        description=f"Пакет `{PKG}` · модель `{config.MODEL}`. Кожне повідомлення — окреме питання, без історії.",
        analytics_enabled=False,
    )
    demo.launch(server_name="127.0.0.1", server_port=args.port)


if __name__ == "__main__":
    main()
