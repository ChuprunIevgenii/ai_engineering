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
"""
RAG homework 5 validation and the stage ladder.

Run from the homework directory:
  uv run eval/eval.py                      # offline checks: structure, data, every TODO (no API key needed)
  uv run eval/eval.py --stage 2            # offline checks + stage 2 on all questions, 2 runs (needs key)
  uv run eval/eval.py --stage all          # every stage whose TODOs are done; stage 5 adds the ACL attacks
  uv run eval/eval.py --stage 1 --repeats 1
  uv run eval/eval.py --table              # the accumulated table from results/ladder.json + failed questions

Design:
  - Imports the solution package `rag` (override with RAG_PKG). Importing it has no side effects.
  - Offline checks call the tools directly with a stand-in ctx on a BM25-only corpus: no model, no key.
    Expected values are computed here from data/, on inputs that are not in questions.json.
  - A stage runs only when the TODOs it needs pass; otherwise it is reported as [SKIP] and its column
    in results/ladder.json is left untouched. A re-run of a stage replaces its column.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import inspect
import json
import os
import random
import re
import sys
import time
import typing
from pathlib import Path
from types import SimpleNamespace

HW_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HW_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import harness  # noqa: E402

PKG = os.environ.get("RAG_PKG", "rag")
SUBMODULES = ["config", "ingest", "search", "catalog", "api", "tools", "stages", "validator"]
DATA = HW_DIR / "data"
CODE_RE = re.compile(r"[А-ЯІЇЄҐ]{2}-\d{3}")
PERSONAS = ["S-1041", "S-1187", "S-2210"]
OTHER_STUDENT = "S-1002"

EXPECTED_TOOLS = {"0": set(), "1": set(), "2": {"search"}, "3": {"search_policies", "search_pages"}}
EXPECTED_TOOLS["4"] = EXPECTED_TOOLS["3"] | {"get_course", "get_staff", "query_catalog"}
EXPECTED_TOOLS["5"] = EXPECTED_TOOLS["4"] | {"my_profile"}
EXPECTED_TOOLS["5-naive"] = EXPECTED_TOOLS["4"] | {"get_student"}
EXPECTED_TOOLS["6"] = EXPECTED_TOOLS["5"]

TODO_OF = {"search": "TODO 2", "policy_metadata": "TODO 3a", "search_policies": "TODO 3b",
           "search_pages": "TODO 3c", "normalize_code": "TODO 4a", "filter_courses": "TODO 4b",
           "get_course": "TODO 4c", "get_staff": "TODO 4d", "query_catalog": "TODO 4e",
           "my_profile": "TODO 5", "validator": "bonus"}
STAGE_NEEDS = {"0": [], "1": [], "2": ["search"],
               "3": ["policy_metadata", "search_policies", "search_pages"]}
STAGE_NEEDS["4"] = STAGE_NEEDS["3"] + ["normalize_code", "filter_courses", "get_course", "get_staff", "query_catalog"]
STAGE_NEEDS["5"] = STAGE_NEEDS["4"] + ["my_profile"]
STAGE_NEEDS["5-naive"] = STAGE_NEEDS["4"]
STAGE_NEEDS["6"] = STAGE_NEEDS["5"] + ["validator"]


class Report:
    def __init__(self) -> None:
        self.passed = self.failed = self.skipped = 0
        self.failures: list[str] = []
        self.todo: set[str] = set()
        self.broken: set[str] = set()

    def ok(self, name: str, msg: str) -> None:
        self.passed += 1
        print(f"    [PASS] {name}: {msg}")

    def fail(self, name: str, msg: str) -> None:
        self.failed += 1
        self.broken.add(name)
        self.failures.append(f"{name}: {msg}")
        print(f"    [FAIL] {name}: {msg}")

    def skip(self, name: str, msg: str) -> None:
        self.skipped += 1
        print(f"    [SKIP] {name}: {msg}")

    def check(self, name: str, condition: bool, msg: str) -> None:
        self.ok(name, msg) if condition else self.fail(name, msg)

    def guard(self, name: str, fn) -> None:
        """Run the checks of one TODO; NotImplementedError -> [SKIP] TODO, any other exception -> [FAIL]."""
        try:
            fn()
        except NotImplementedError:
            self.todo.add(name)
            self.skip(name, f"{TODO_OF.get(name, 'TODO')} not implemented")
        except Exception as e:  # noqa: BLE001
            self.fail(name, f"raised {type(e).__name__}: {e}"[:300])

    def missing(self, stage: str) -> list[str]:
        return [n for n in STAGE_NEEDS[stage] if n in self.todo or n in self.broken]


def load_solution():
    sol = importlib.import_module(PKG)
    for sub in SUBMODULES:
        setattr(sol, sub, importlib.import_module(f"{PKG}.{sub}"))
    return sol


# --------------------------------------------------------------------------- ground truth from data/


def read_fields(path: Path) -> dict[str, str]:
    return dict(re.findall(r"^([^\n:#]+): (.+)$", path.read_text(encoding="utf-8"), flags=re.M))


def read_cards() -> dict[str, dict]:
    cards = {}
    for p in sorted((DATA / "pages" / "courses").glob("*.md")):
        f = read_fields(p)
        cards[p.stem] = {"faculty": re.search(r"\((\w+)\)", f["Факультет"]).group(1), "year": int(f["Курс"]),
                         "semester": f["Семестр"], "kind": f["Тип"], "credits": int(f["Кредити ЄКТС"]),
                         "teacher": f["Викладач"], "room": f["Аудиторія"]}
    return cards


def read_staff() -> dict[str, dict]:
    out = {}
    for p in sorted((DATA / "pages" / "staff").glob("*.md")):
        title = p.read_text(encoding="utf-8").splitlines()[0].lstrip("# ").strip()
        names = [w for w in title.split() if w[:1].isupper() and not w.endswith(".")]
        out[p.stem] = {"surname": names[0], "first": names[1], "email": read_fields(p)["Електронна пошта"]}
    return out


def literal_values(hint) -> set:
    """Values of every Literal inside a type hint: Literal["a", "b"] | None -> {"a", "b"}."""
    if typing.get_origin(hint) is typing.Literal:
        return set(typing.get_args(hint))
    return set().union(*(literal_values(a) for a in typing.get_args(hint)))


def teacher_matches(full: str, query: str) -> bool:
    names, words = full.lower().split(), query.lower().replace(".", " ").split()
    return 0 < len(words) <= len(names) and all(n.startswith(w) for n, w in zip(names, words))


def expected_rows(cards: dict, **flt) -> list[str]:
    def keep(c):
        for k, v in flt.items():
            if k == "teacher":
                if not teacher_matches(c["teacher"], v):
                    return False
            elif c[k] != v:
                return False
        return True
    return sorted(code for code, c in cards.items() if keep(c))


def question_codes(questions: list[dict]) -> set[str]:
    out = set()
    for q in questions:
        out |= set(CODE_RE.findall(q["q"])) | set(q.get("sources") or []) | set(q.get("answer_list") or [])
    return out


# --------------------------------------------------------------------------- data checks


def check_data(r: Report, questions: list[dict]) -> None:
    ids = [q["id"] for q in questions]
    r.check("data", len(ids) == len(set(ids)), f"{len(ids)} questions, ids unique")
    r.check("data", all(q["type"] in harness.TYPES for q in questions), "every type is one of the table rows")
    r.check("data", all(q.get("must") or q.get("reference") for q in questions), "every question has must or reference")
    known = {m["doc_id"] for m in json.loads((DATA / "policies" / "meta.json").read_text(encoding="utf-8"))}
    for kind in ("courses", "staff", "news"):
        known |= {p.stem for p in (DATA / "pages" / kind).glob("*.md")}
    known |= {f"api:{s}" for s in json.loads((DATA / "api" / "students.json").read_text(encoding="utf-8"))}
    bad = sorted({s for q in questions for s in q.get("sources") or [] if s not in known})
    r.check("data", not bad, f"expected sources exist in data/ {bad or ''}".strip())
    agg = [q for q in questions if q["type"] == "aggregation"]
    r.check("data", all(str(len(q["answer_list"])) in q["must"][0] for q in agg),
            f"aggregation answers equal len(answer_list) ({len(agg)} questions)")
    r.check("data", any(q.get("acl") for q in questions), "an ACL question carries the forbid list for the attacks")


# --------------------------------------------------------------------------- behavioral checks (offline)


def ctx_for(sol, corpus, student: str = "S-1041"):
    return SimpleNamespace(deps=sol.tools.Deps(corpus, student_id=student))


def labels_shown(ctx, out: str) -> bool:
    """The model can cite a source only by a label it sees, so every label a tool records must be in its output."""
    return all(f"[{label}]" in out for label in ctx.deps.seen)


def check_ready_stages(sol, corpus, r: Report, questions: list[dict]) -> None:
    from pydantic_ai.messages import ModelResponse, TextPart
    from pydantic_ai.models.function import FunctionModel

    for stage, want in EXPECTED_TOOLS.items():
        got = {f.__name__ for f in sol.stages.stage_tools(stage)}
        r.check("stages", got == want, f"stage {stage} tools = {sorted(want) or '[]'}"
                + ("" if got == want else f" (got {sorted(got)})"))

    prompts: list[str] = []

    def fake(messages, info):
        prompts.append(" ".join(str(getattr(p, "content", "")) for m in messages for p in m.parts))
        return ModelResponse(parts=[TextPart("ok")])

    q = questions[0]
    loop = asyncio.new_event_loop()
    try:
        for stage in ("0", "1"):
            deps = sol.tools.Deps(corpus, student_id=q["student"])
            res = loop.run_until_complete(sol.stages.run(sol.stages.build_agent(stage, model=FunctionModel(fake)),
                                                         stage, q["q"], deps))
            r.check(f"stage {stage}", res.output == "ok", "runs end to end on a stand-in model")
        r.check("stage 1", "с. " in prompts[-1] and len(prompts[-1]) > 500,
                "question and top-k chunks with page labels go into one prompt")
        r.check("stage 0", "с. " not in prompts[0], "closed book: no context in the prompt")
    finally:
        loop.close()


def check_fusion(sol, r: Report) -> None:
    """60 pages, one has the code; the stand-in dense ranker puts that page last. RRF must keep the BM25 match:
    pages without any query term have no BM25 rank and get no BM25 bonus."""
    import numpy as np

    Chunk = sol.ingest.Chunk
    chunks = [Chunk(f"p{i}", "course", f"ЗЗ-{100 + i}", f"ЗЗ-{100 + i} Дисципліна {i}", None,
                    f"Курс {i % 4 + 1}. Дисципліна про предмет номер {i}.") for i in range(60)]
    target = chunks[37].source
    idx = sol.search.Index(chunks, dense=False)

    class Encoder:
        def encode(self, texts, normalize_embeddings=True):
            return np.ones((len(texts), 2), dtype="float32")

    idx.encoder = Encoder()
    idx.emb = np.array([[1.0, 1.0] if c.source != target else [-1.0, -1.0] for c in chunks], dtype="float32")
    got = [c.source for c in idx.search(target, k=5)]
    r.check("hybrid", target in got, f"an exact BM25 match survives fusion with a dense ranker that dislikes it"
            + ("" if target in got else f" (top-5 {got})"))


def check_search(sol, corpus, r: Report, labels: dict) -> None:
    ctx = ctx_for(sol, corpus)
    out = sol.tools.search(ctx, "консультації викладача кафедри програмної інженерії")
    seen = ctx.deps.seen
    r.check("search", isinstance(out, str) and bool(seen), "returns text and records what it returned in deps.seen")
    r.check("search", all(f"[{lab}]" in out for lab in seen), "every recorded label is shown as [label] in the output")
    r.check("search", all(lab in labels for lab in seen), "labels are the corpus citation labels (Chunk.label())")
    ctx = ctx_for(sol, corpus)
    sol.tools.search(ctx, "розмір академічної стипендії і рейтинг")
    r.check("search", any(", с. " in lab for lab in ctx.deps.seen), "policy hits carry the page: «Назва, с. N»")


def check_policies(sol, corpus, r: Report, labels: dict) -> None:
    metas = json.loads((DATA / "policies" / "meta.json").read_text(encoding="utf-8"))
    by_id = {m["doc_id"]: m for m in metas}
    superseded = {m["doc_id"] for m in metas if m.get("superseded_by")}

    def meta_fn():
        for m in metas:
            d = sol.ingest.policy_metadata(m)
            if d.get("effective_from") != m["effective_from"] or d.get("superseded_by") != m.get("superseded_by"):
                raise AssertionError(f"policy_metadata({m['doc_id']}) = {d}")
        r.ok("policy_metadata", f"effective_from and superseded_by for all {len(metas)} policies")
        chunks = [c for c in corpus.index.chunks if c.kind == "policy"]
        bad = [c.id for c in chunks if c.meta.get("superseded_by") != by_id[c.source].get("superseded_by")]
        r.check("policy_metadata", not bad, f"every policy chunk carries its document's metadata {bad[:3] or ''}".strip())

    r.guard("policy_metadata", meta_fn)

    def policies_fn():
        query = "повторне складання іспиту, комісія і строки"
        ctx = ctx_for(sol, corpus)
        out = sol.tools.search_policies(ctx, query)
        src = {labels.get(lab) for lab in ctx.deps.seen}
        r.check("search_policies", bool(src) and src <= set(by_id), "returns policy chunks only")
        r.check("search_policies", not src & superseded, "default: no chunks from superseded editions")
        r.check("search_policies", any(by_id[s]["effective_from"] in out for s in src),
                "validity date (effective_from) is in the output")
        ctx = ctx_for(sol, corpus)
        sol.tools.search_policies(ctx, "заборгованість перескладання")
        n = sum(len(t.split("\n")) for lab, t in ctx.deps.seen.items() if labels.get(lab) in by_id)
        r.check("search_policies", n >= 4, f"superseded editions dropped before ranking: {n} current chunks "
                "(a filter applied to the top-k leaves 2-3 here: the 2024 edition took the slots)")
        ctx = ctx_for(sol, corpus)
        out = sol.tools.search_policies(ctx, query, include_superseded=True)
        old = {labels.get(lab) for lab in ctx.deps.seen} & superseded
        r.check("search_policies", bool(old), "include_superseded=True: the older edition comes back")
        r.check("search_policies", bool(old) and all(by_id[s]["effective_from"] in out for s in old)
                and re.search(r"замін|втрат|нечинн|скасов|superseded", out, re.I) is not None,
                "the older edition is marked as replaced, with its date")

    r.guard("search_policies", policies_fn)

    def pages_fn():
        kind_of = {c.source: c.kind for c in corpus.index.chunks}
        ctx = ctx_for(sol, corpus)
        sol.tools.search_pages(ctx, "консультації Гордієнко пятниця")
        src = {labels.get(lab) for lab in ctx.deps.seen}
        r.check("search_pages", bool(src) and not src & set(by_id), "returns faculty pages only, no policies")
        news = next(c for c in corpus.index.chunks if c.kind == "news")
        ctx = ctx_for(sol, corpus)
        sol.tools.search_pages(ctx, news.title)
        kinds = {kind_of.get(labels.get(lab)) for lab in ctx.deps.seen} | {kind_of.get(s) for s in src}
        r.check("search_pages", {"staff", "news"} <= kinds,
                f"staff pages and news come back too, not only course cards (got {sorted(k for k in kinds if k)})")

    r.guard("search_pages", pages_fn)


def check_exact(sol, corpus, r: Report, questions: list[dict]) -> None:
    rng = random.Random(len(questions))
    cards = read_cards()
    in_set = question_codes(questions)
    codes = rng.sample(sorted(set(cards) - in_set), 4)
    variants = lambda c: [c, c.lower().replace("-", " "), f"  {c}  ", c.replace("-", "–"), c.replace("-", "")]

    def norm_fn():
        bad = [(v, sol.catalog.normalize_code(v)) for c in codes for v in variants(c)
               if sol.catalog.normalize_code(v) != c]
        r.check("normalize_code", not bad, f"case, spaces and dashes -> canonical code on {codes} {bad[:2] or ''}".strip())

    r.guard("normalize_code", norm_fn)

    def course_fn():
        for c in codes[:2]:
            outs = [sol.tools.get_course(ctx_for(sol, corpus), v) for v in variants(c)]
            r.check("get_course", len(set(outs)) == 1, f"{c}: every spelling returns the same card")
            card = cards[c]
            r.check("get_course", card["room"] in outs[0] and card["teacher"] in outs[0],
                    f"{c}: the card has room {card['room']} and the teacher")
        ctx = ctx_for(sol, corpus)
        sol.tools.get_course(ctx, codes[2])
        r.check("get_course", bool(ctx.deps.seen), "records the card in deps.seen for citations")
        missing = next(f"ПЗ-{n}" for n in range(999, 100, -1) if f"ПЗ-{n}" not in cards)
        out = sol.tools.get_course(ctx_for(sol, corpus), missing)
        r.check("get_course", bool(out.strip()) and "Аудиторія" not in out, f"{missing}: honest «not in catalog»")
        searched = []
        corpus.index.search = lambda *a, **k: searched.append(a) or type(corpus.index).search(corpus.index, *a, **k)
        try:
            sol.tools.get_course(ctx_for(sol, corpus), codes[3])
        finally:
            del corpus.index.search
        r.check("get_course", not searched, "exact lookup, not search: a code sits in the prerequisites of dozens "
                "of other cards, so top-1 of a search is not the card")

    r.guard("get_course", course_fn)

    def staff_fn():
        staff = read_staff()
        mentioned = {s for q in questions for s in q.get("sources") or []}
        picks = rng.sample(sorted(set(staff) - mentioned), 3)
        for sid in picks:
            s = staff[sid]
            for name in (s["surname"], f"{s['first']} {s['surname']}"):
                out = sol.tools.get_staff(ctx_for(sol, corpus), name)
                r.check("get_staff", s["email"] in out, f"«{name}» -> page with {s['email']}")
        s = next(staff[sid] for sid in picks if len(staff[sid]["surname"]) >= 6)
        genitive = s["surname"][:-1] + "а" if s["surname"].endswith("о") else s["surname"] + "а"
        out = sol.tools.get_staff(ctx_for(sol, corpus), f"{genitive} {s['first'][0]}.")
        r.check("get_staff", s["email"] in out, f"«{genitive} {s['first'][0]}.» (case ending, initial) -> {s['email']}")
        out = sol.tools.get_staff(ctx_for(sol, corpus), "Неіснуюченко")
        r.check("get_staff", "@" not in out, "unknown surname: no page, no e-mail")
        by_surname: dict[str, list[dict]] = {}
        for sid in sorted(set(staff) - mentioned):
            by_surname.setdefault(staff[sid]["surname"], []).append(staff[sid])
        same = rng.choice([v for v in by_surname.values() if len(v) > 1])
        s, others = same[0], [o["email"] for o in same[1:]]
        out = sol.tools.get_staff(ctx_for(sol, corpus), f"{s['surname']} {s['first']}")
        r.check("get_staff", s["email"] in out and not any(e in out for e in others),
                f"«{s['surname']} {s['first']}»: only this page, not the namesakes' "
                "(a stem cut too short also matches other first names and patronymics)")

    r.guard("get_staff", staff_fn)

    teachers = sorted({c["teacher"] for c in cards.values()})
    asked = json.dumps(questions, ensure_ascii=False)
    namesakes = [t for t in teachers if t.split()[0] not in asked
                 and sum(o.split()[0] == t.split()[0] for o in teachers) > 1]
    surname, first = rng.choice(namesakes).split()[:2]
    surnames = sorted({t.split()[0] for t in teachers})
    inside = [a for a in surnames if a not in asked and any(a != b and a.lower() in b.lower() for b in surnames)]
    filters = [dict(faculty="ФГН", year=2, semester="осінній"), dict(faculty="ФІТ", kind="обов'язкова", credits=5),
               dict(teacher=surname), dict(teacher=f"{surname} {first}"), dict(teacher=f"{surname} {first[0]}."),
               *[dict(teacher=a) for a in inside[:1]],
               dict(year=4, kind="вибіркова", semester="весняний")]
    answer_lists = {tuple(sorted(q["answer_list"])) for q in questions if q.get("answer_list")}

    def filter_fn():
        for f in filters:
            want = expected_rows(cards, **f)
            assert want and tuple(want) not in answer_lists, f"eval filter {f} must differ from the question set"
            got = [c.code for c in sol.catalog.filter_courses(corpus.catalog, **f)]
            r.check("filter_courses", got == want, f"{f} -> {len(want)} courses, sorted by code"
                    + ("" if got == want else f" (got {len(got)})"))
        got = sol.catalog.filter_courses(corpus.catalog)
        r.check("filter_courses", len(got) == len(cards), f"no filters -> all {len(cards)} courses")

    r.guard("filter_courses", filter_fn)

    def catalog_fn():
        for f in filters[:2]:
            want = expected_rows(cards, **f)
            ctx = ctx_for(sol, corpus)
            out = sol.tools.query_catalog(ctx, **f)
            r.check("query_catalog", re.search(rf"(?<!\d){len(want)}(?!\d)", out) is not None
                    and all(c in out for c in want), f"{f}: the count {len(want)} and every code are in the output")
            r.check("query_catalog", bool(ctx.deps.seen), "records the result in deps.seen for citations")
            r.check("query_catalog", labels_shown(ctx, out), "the output shows the [label] it records under")
        params = set(inspect.signature(sol.tools.query_catalog).parameters) - {"ctx"}
        r.check("query_catalog", {"faculty", "year", "semester", "kind", "teacher", "credits"} <= params,
                "typed filters: faculty, year, semester, kind, teacher, credits")
        hints = typing.get_type_hints(sol.tools.query_catalog)
        want = {"faculty": {"ФІТ", "ЕКФ", "ФГН"}, "semester": {"осінній", "весняний"},
                "kind": {"вибіркова", "обов'язкова"}}
        loose = [k for k, v in want.items() if literal_values(hints.get(k)) != v]
        r.check("query_catalog", not loose, "faculty, semester, kind are Literal with the catalog values: the model "
                "cannot pass «Економічний факультет»" + (f" (free text: {loose})" if loose else ""))

    r.guard("query_catalog", catalog_fn)


def check_profile(sol, corpus, r: Report) -> None:
    students = json.loads((DATA / "api" / "students.json").read_text(encoding="utf-8"))

    def profile_fn():
        params = list(inspect.signature(sol.tools.my_profile).parameters)
        r.check("my_profile", params == ["ctx"], f"no id parameter: the model cannot choose whose data (got {params})")
        for sid in PERSONAS:
            ctx = ctx_for(sol, corpus, sid)
            out = sol.tools.my_profile(ctx)
            others = [s for s, rec in students.items() if s != sid and (s in out or rec["name"] in out)]
            r.check("my_profile", (sid in out or students[sid]["name"] in out) and not others,
                    f"{sid}: own record only, identity from ctx.deps.student_id")
        r.check("my_profile", bool(ctx.deps.seen) and labels_shown(ctx, out),
                "records the profile in deps.seen and the output shows that [label]")

    r.guard("my_profile", profile_fn)
    ctx = ctx_for(sol, corpus)
    out = sol.tools.get_student(ctx, OTHER_STUDENT.lower())
    r.check("get_student", OTHER_STUDENT in out, "naive tool (stage 5-naive) returns any id it is given — by design")


def check_validator(sol, corpus, r: Report) -> None:
    def fn():
        n = 23
        deps = sol.tools.Deps(corpus)
        deps.question = "Скільки дисциплін на четвертому курсі?"
        sol.tools.record(deps, "каталог", f"Знайдено дисциплін: {n}")
        sol.tools.record(deps, "ПЗ-224 Тестування програмного забезпечення", "Аудиторія: 2.118")
        check = sol.validator.check_answer

        def passes(answer: str, msg: str) -> None:
            got = check(answer, deps)
            r.check("validator", got == [], msg if got == [] else f"{msg}\n        answer: {answer!r}\n        got: {got}")

        passes(f"Їх {n} [каталог].", "grounded answer passes")
        r.check("validator", bool(check(f"Їх {n + 1} [каталог].", deps)), "a number no tool returned -> problem")
        r.check("validator", bool(check(f"Їх {n} [Положення про невідоме].", deps)),
                "citation of a source not returned in this run -> problem")
        r.check("validator", bool(check(f"Їх {n} [каталог], форма: https://vnu-portal-login.example/x", deps)),
                "link outside the allowlist -> problem")
        r.check("validator", bool(check(f"Їх {n} [каталог], форма: https://evil-vnu.example/x", deps)),
                "evil-vnu.example is not vnu.example: compare the host or its «.vnu.example» suffix")
        deps.question = "Скільки дисциплін на 4 курсі?"
        passes(f"На 4 курсі їх {n} [каталог].", "a number from the question passes")
        passes(f"Їх {n} [каталог], пишіть на dean@vnu.example, ауд. 2.118 [ПЗ-224].",
               "allowlisted domain and numbers from another returned source pass")
        passes(f"Форма: https://vnu.example/apply, а їх {n} [каталог]. Деталі на vnu.example.",
               "an allowlisted link followed by a comma or a period passes: punctuation is not part of the host")
        sol.tools.record(deps, "ПЗ-310 Компілятори", "# ПЗ-310 Компілятори\nКредити ЄКТС: 4\nПререквізити: КН-311, ПЗ-204")
        passes("Пререквізити ПЗ-310 [ПЗ-310 Компілятори]:\n1. КН-311 (4 кредити)\n2. ПЗ-204\n3. ПЗ-224, ауд. 2.118",
               "a list of codes from the sources passes: list numbers are not facts, codes stay apart, "
               "and a number in a label counts (ПЗ-224 is only in a label)")
        sol.tools.record(deps, "Положення про стипендії, с. 2", "Звичайна стипендія — 2 200 грн.")
        sol.tools.record(deps, "Положення про стипендії, с. 2", "Підвищена — 3 100 грн від 95 балів.")
        passes("2 200 або 3 100 грн [Положення про стипендії, с. 2].", "two chunks of one page: numbers from both pass")

    r.guard("validator", fn)


def check_harness(r: Report) -> None:
    q = {"id": "t", "type": "version", "q": "?", "must": [["дві", "2 спроби"]], "forbid": ["74 бали"]}
    r.check("harness", harness.check_text(q, "Дозволено дві спроби.") == (True, []), "must group matched by any variant")
    r.check("harness", harness.check_text(q, "Дві спроби, максимум 74 бали")[1] == ["74 бали"], "forbid string caught")
    r.check("harness", not harness.check_text(q, "Якщо обидві спроби не вдалися…")[0]
            and not harness.check_text({"must": [["8"]]}, "Знайдено 3: ПЗ-448, КН-108")[0]
            and harness.check_text({"must": [["осін"]]}, "У **осінньому** семестрі")[0],
            "must matches from a word boundary: «дві» not in «обидві», «8» not in «ПЗ-448»")
    acl = harness.acl_verdict({"student": "S-1041", "answer": "Вибачте, це конфіденційно.",
                               "tools": [{"tool": "get_student", "args": {"student_id": "s-1002"}}]},
                              {"forbid": ["78"]})
    r.check("harness", acl["leaked"] and acl["fetched_other"] == ["S-1002"],
            "ACL leak counted from the tool log even when the answer refuses")
    lad = {"version": 1, "stages": {}, "acl": {}}
    harness.put_runs(lad, "stages", "3", [{"results": []}] * 2, "m")
    harness.put_runs(lad, "stages", "3", [{"results": []}], "m")
    r.check("harness", len(lad["stages"]["3"]["runs"]) == 1, "re-running a stage replaces its column")
    r.check("harness", "--stage 0" in harness.render_table({"stages": {}, "acl": {}}), "empty ladder -> hint")

    r.check("harness", harness.bootstrap_ci([1.0] * 10) == (1.0, 1.0), "bootstrap interval of a constant is a point")
    c = harness.paired_comparison([0.0] * 12, [1.0] * 12)
    r.check("harness", c["verdict"] == "краще" and c["better"] == 12, "paired comparison detects a clear win")
    c = harness.paired_comparison([0, 1] * 6, [1, 0] * 6)
    r.check("harness", c["verdict"] == "вимірної різниці немає", "traded wins and losses -> no measurable difference")
    res = lambda i, ok, found: {"id": f"q{i}", "type": "semantic", "passed": ok, "cost": 0.0, "llm_calls": 1,
                                "tools": [], "sources_expected": ["d"], "sources_touched": ["d"] if found else []}
    lad = {"version": 1, "stages": {}, "acl": {}}
    harness.put_runs(lad, "stages", "1", [{"results": [res(i, i < 2, True) for i in range(4)]},
                                          {"results": [res(i, i < 1, True) for i in range(4)]}], "m", "j")
    harness.put_runs(lad, "stages", "2", [{"results": [res(i, True, i != 3) for i in range(4)]},
                                          {"results": [res(i, i != 3, i != 3) for i in range(4)]}], "m", "j")
    table = harness.render_table(lad)
    r.check("harness", "| правильно в усіх прогонах | 1/4 | 3/4 |" in table, "pass^k row: right in every run")
    r.check("harness", "| провали: не знайшов / знайшов і помилився | 0 / 5 | 1 / 0 |" in table,
            "failures split into «not found» and «found but wrong»")
    r.check("harness", [(f["id"], f["failed_runs"]) for f in harness.failures(lad, "1")]
            == [("q2", 2), ("q3", 2), ("q1", 1)], "failed questions from every run, most failed runs first")
    r.check("harness", harness.sources_seen({"профіль студента": "Середній бал 91, боргів немає"}, {}, "S-1041")
            == {"api:S-1041"} and harness.sources_seen({"студент S-1002": "{}"}, {}, "S-1041") == {"api:S-1002"},
            "a profile without an id counts as the logged-in student's record; an explicit id wins")
    r.check("harness", "Сх. 2 проти 1:" in table and "Увага" not in table, "paired comparison of adjacent stages")
    out_of_credits = [{"results": [{**res(i, False, False), "reason": "judge error: ModelHTTPError: status_code: 402"}
                                   for i in range(4)]}]
    ok, msg = harness.run_health(out_of_credits)
    r.check("harness", not ok and "кредит" in msg, "a run the judge could not grade (402) is refused, with a hint")
    r.check("harness", harness.run_health(lad["stages"]["2"]["runs"])[0], "a healthy run is accepted")

    async def boom(*_a, **_k):
        raise RuntimeError("provider 502")

    fake = SimpleNamespace(config=SimpleNamespace(DEFAULT_STUDENT="S-1041"),
                           tools=SimpleNamespace(Deps=lambda corpus, student_id: SimpleNamespace(seen={})),
                           stages=SimpleNamespace(run=boom))
    delay, harness.RETRY_DELAY = harness.RETRY_DELAY, 0
    try:
        res = asyncio.run(harness.ask(fake, None, "2", {"id": "t", "type": "semantic", "q": "?"}, None))
        graded = asyncio.run(harness.grade(None, {"id": "t"}, res, {}))
    finally:
        harness.RETRY_DELAY = delay
    r.check("harness", not graded["passed"] and "502" in graded["reason"],
            f"provider error after {harness.ATTEMPTS} attempts -> failed question with a reason, run goes on")

    from pydantic_ai import Agent, ModelRetry
    from pydantic_ai.messages import ModelResponse, TextPart
    from pydantic_ai.models.function import FunctionModel

    stubborn = Agent(FunctionModel(lambda _m, _i: ModelResponse(parts=[TextPart("Їх 7 [каталог].")])),
                     retries={"output": 1})

    @stubborn.output_validator
    def reject(output: str) -> str:
        raise ModelRetry("Числа 7 немає в отриманих джерелах.")

    fake.stages.run = lambda agent, _stage, question, deps: agent.run(question, deps=deps)
    res = asyncio.run(harness.ask(fake, stubborn, "6", {"id": "t", "type": "semantic", "q": "?"}, None))
    graded = asyncio.run(harness.grade(None, {"id": "t"}, res, {}))
    r.check("harness", not graded["passed"] and "Числа 7" in graded["reason"] and res["answer"] == "Їх 7 [каталог]."
            and res["llm_calls"] == 2, "output rejected by the validator: the last answer and its last message are kept")

    r.check("harness", harness.cohen_kappa([True, False] * 5, [True, False] * 5) == 1.0
            and abs(harness.cohen_kappa([True] * 8 + [False] * 2, [True] * 10)) < 1e-9,
            "kappa: 1 for perfect agreement, 0 for an always-pass judge")
    graded_res = lambda i, j: {"id": f"q{i}", "type": harness.TYPES[i % 2], "passed": j, "answer": f"a{i}",
                               "judge": j, "must_ok": True, "error": None}
    lad = {"version": 1, "stages": {"1": {"runs": [{"results": [graded_res(i, i % 3 != 0) for i in range(20)]}]}},
           "acl": {}}
    qs = [{"id": f"q{i}", "type": harness.TYPES[i % 2], "q": "?", "must": [["x"]]} for i in range(20)]
    sample = harness.label_sample(lad, qs, per_type=4)
    r.check("harness", len(sample) == 8 and all(it["label"] is None and "judge" not in it for it in sample)
            and len({it["key"] for it in sample}) == 8,
            "label sample: per type, distinct questions, judge verdict hidden")


# --------------------------------------------------------------------------- stage runs (need a key)


async def run_stage(sol, corpus, stage: str, qs: list[dict], repeats: int, concurrency: int,
                    judge, labels: dict, acl: bool = False) -> list[dict]:
    agent = sol.stages.build_agent(stage)
    sem = asyncio.Semaphore(concurrency)

    async def one(q):
        async with sem:
            res = await harness.ask(sol, agent, stage, q, corpus)
            return harness.acl_verdict(res, q) if acl else await harness.grade(judge, q, res, labels)

    runs = []
    for k in range(1, repeats + 1):
        t0 = time.time()
        results = await asyncio.gather(*(one(q) for q in qs))
        secs = round(time.time() - t0, 1)
        cost = sum(x["cost"] for x in results)
        judge_cost = sum(x.get("judge_cost", 0.0) for x in results)
        runs.append({"secs": secs, "cost": cost, "judge_cost": judge_cost, "results": results})
        ok, msg = harness.run_health(runs[-1:])
        if not ok and harness.FATAL_STATUS.search(msg):
            print(f"    stage {stage} run {k}: stopped — {msg[:160]}")
            break
        if acl:
            print(f"    stage {stage} ACL run {k}: leaks {sum(x['leaked'] for x in results)}/{len(results)}"
                  f" · ${cost:.4f} · {secs:.0f}s")
        else:
            errs = sum(bool(x["error"]) for x in results)
            print(f"    stage {stage} run {k}: passed {sum(x['passed'] for x in results)}/{len(results)}"
                  f" · ${cost:.4f} agent + ${judge_cost:.4f} judge · {secs:.0f}s"
                  + (f" · errors {errs} (see reason in the ladder)" if errs else ""))
    return runs


def run_ladder(sol, r: Report, args, questions: list[dict]) -> None:
    if not os.environ.get("OPENROUTER_API_KEY"):
        r.skip("ladder", "OPENROUTER_API_KEY not set — stage runs skipped")
        return
    wanted = harness.TABLE_STAGES if args.stage == "all" else [args.stage]
    if any(s not in sol.stages.STAGES for s in wanted):
        r.fail("ladder", f"unknown stage {args.stage!r}; known: {sol.stages.STAGES} or all")
        return
    runnable = []
    for s in wanted:
        gaps = r.missing(s)
        if gaps:
            r.skip(f"stage {s}", "needs " + ", ".join(f"{TODO_OF[g]} {g}" for g in gaps) + " — column not written")
        else:
            runnable.append(s)
    if not runnable:
        return

    print(f"\n  Loading the full corpus with dense embeddings (first run downloads {sol.config.ENCODER})…")
    corpus = sol.tools.Corpus.load()
    labels = harness.label_sources(corpus)
    judge = harness.make_judge(sol.config.JUDGE_MODEL)
    path = Path(args.ladder) if args.ladder else sol.config.RESULTS_DIR / "ladder.json"
    ladder = harness.load_ladder(path)
    print(f"  model {sol.config.MODEL} · judge {harness.judge_id(sol.config.JUDGE_MODEL)} · {len(questions)} questions"
          f" × {args.repeats} runs · ladder {path.relative_to(HW_DIR) if path.is_relative_to(HW_DIR) else path}")

    loop = asyncio.new_event_loop()
    try:
        for s in runnable:
            print(f"\n  Stage {s}:")
            runs = loop.run_until_complete(run_stage(sol, corpus, s, questions, args.repeats, args.concurrency,
                                                     judge, labels))
            ok, msg = harness.run_health(runs)
            if not ok:
                r.fail(f"stage {s}", f"column not written: {msg}")
                if harness.FATAL_STATUS.search(msg):
                    break
                continue
            harness.put_runs(ladder, "stages", s, runs, sol.config.MODEL, harness.judge_id(sol.config.JUDGE_MODEL))
            if s == "5":
                attacks = harness.acl_questions(questions, sol.config.DEFAULT_STUDENT)
                for acl_stage in harness.ACL_STAGES:
                    acl_runs = loop.run_until_complete(run_stage(sol, corpus, acl_stage, attacks, args.repeats,
                                                                 args.concurrency, judge, labels, acl=True))
                    ok, msg = harness.run_health(acl_runs)
                    if ok:
                        harness.put_runs(ladder, "acl", acl_stage, acl_runs, sol.config.MODEL)
                    else:
                        r.fail(f"stage {acl_stage}", f"ACL column not written: {msg}")
            harness.save_ladder(path, ladder)
            r.ok(f"stage {s}", f"{args.repeats} runs saved to {path.name}")
    finally:
        loop.close()

    print("\n" + harness.render_table(ladder))
    print_failures(ladder, runnable[-1], path)


def print_failures(ladder: dict, stage: str, path: Path) -> None:
    fails = harness.failures(ladder, stage)
    if not fails:
        return
    print(f"\n  Failed questions of stage {stage}: {len(fails)} (answers and tool calls of these runs in {path.name}; "
          f"re-run live: uv run ask.py --stage {stage} --id <id>):")
    for f in fails:
        print(f"    {f['id']} {f['type']:<11} failed {f['failed_runs']}/{f['runs']} runs · touched "
              f"{len(f['sources_touched'])}/{len(f['sources_expected'])} sources · {f['reason'][:160]}")


# --------------------------------------------------------------------------- judge calibration


def dump_labels(sol, args, questions: list[dict]) -> None:
    ladder_path = Path(args.ladder) if args.ladder else sol.config.RESULTS_DIR / "ladder.json"
    out = Path(args.labels) if args.labels else sol.config.RESULTS_DIR / "judge_labels.json"
    if out.exists() and any(it.get("label") for it in json.loads(out.read_text(encoding="utf-8"))):
        print(f"  {out.name} already has labels; delete it first if you really want a new sample.")
        sys.exit(1)
    items = harness.label_sample(harness.load_ladder(ladder_path), questions)
    if not items:
        print(f"  {ladder_path.name} has no graded runs yet. Run: uv run eval/eval.py --stage <N>")
        sys.exit(1)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(items, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"  Saved {len(items)} answers to {out}.")
    print('  Set "label": "pass" or "fail" for each (is the answer right by the reference?), optional "note",')
    print("  then run: uv run eval/eval.py --kappa")


def run_kappa(sol, args, questions: list[dict]) -> None:
    path = Path(args.labels) if args.labels else sol.config.RESULTS_DIR / "judge_labels.json"
    if not path.exists():
        print(f"  {path.name} not found. Run: uv run eval/eval.py --dump-labels")
        sys.exit(1)
    items = json.loads(path.read_text(encoding="utf-8"))
    bad = [it["key"] for it in items if it.get("label") not in (None, "pass", "fail")]
    if bad:
        print(f"  label must be \"pass\" or \"fail\": {', '.join(bad[:5])}")
        sys.exit(1)
    items = [it for it in items if it.get("label")]
    if not items:
        print(f"  No labels in {path.name} yet.")
        sys.exit(1)
    if not os.environ.get("OPENROUTER_API_KEY"):
        print("  OPENROUTER_API_KEY not set: --kappa runs the current judge on your labeled answers.")
        sys.exit(1)
    qs = {q["id"]: q for q in questions}
    judge = harness.make_judge(sol.config.JUDGE_MODEL)

    async def grade_all():
        sem = asyncio.Semaphore(args.concurrency)

        async def one(it):
            async with sem:
                return await harness.judge_answer(judge, qs[it["key"].split("/")[-1]], it["answer"])
        return await asyncio.gather(*(one(it) for it in items))

    results = asyncio.run(grade_all())
    errors = [it["key"] for it, (v, _) in zip(items, results) if v.reason.startswith("judge error")]
    if errors:
        print(f"  The judge failed on {len(errors)} answers ({', '.join(errors[:3])}…); kappa not computed.")
        sys.exit(1)
    verdicts = [v.correct for v, _ in results]
    rep = harness.kappa_report(items, verdicts)
    print(f"\n  Judge {harness.judge_id(sol.config.JUDGE_MODEL)} vs your labels (n={rep['n']}, "
          f"${sum(c for _, c in results):.4f})")
    print(f"    raw agreement            {rep['agreement']:.2f}")
    print(f"    'always pass' agreement  {rep['always_pass']:.2f}   (what a lazy judge would get)")
    print(f"    Cohen's kappa            {rep['kappa']:.2f}   (>= 0.6 to trust the table)")
    print(f"    bad answers caught       {rep['caught']}")
    print(f"    good answers failed      {rep['wrongly_failed']}")
    diffs = [(it, v) for it, (v, _) in zip(items, results) if (it["label"] == "pass") != v.correct]
    if diffs:
        print("\n  Disagreements:")
        for it, v in diffs:
            print(f"    {it['key']} {it['type']:<11} you: {it['label']} · judge: {'pass' if v.correct else 'fail'}"
                  f" — {v.reason[:110]}")
            if it.get("note"):
                print(f"      your note: {it['note'][:110]}")
    report = {"judge": harness.judge_id(sol.config.JUDGE_MODEL), **rep,
              "disagreements": [{"key": it["key"], "label": it["label"], "judge": v.correct, "reason": v.reason}
                                for it, v in diffs]}
    out = path.with_name("kappa_report.json")
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n  Saved {out.name}.")


# --------------------------------------------------------------------------- main


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", help="0-6, 5-naive or all (needs OPENROUTER_API_KEY)")
    ap.add_argument("--repeats", type=int, default=2, help="runs per stage (default 2)")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--table", action="store_true", help="only print the table from the ladder file")
    ap.add_argument("--ladder", help="ladder file (default results/ladder.json)")
    ap.add_argument("--dump-labels", action="store_true", help="sample 40 answers from the ladder to label by hand")
    ap.add_argument("--kappa", action="store_true", help="current judge vs your labels (needs OPENROUTER_API_KEY)")
    ap.add_argument("--labels", help="labels file (default results/judge_labels.json)")
    args = ap.parse_args()

    print("=" * 60)
    print(f"  RAG Homework 5 Validation   (RAG_PKG={PKG})")
    print("=" * 60)

    try:
        sol = load_solution()
    except Exception as e:  # noqa: BLE001
        print(f"  Could not import package {PKG!r}: {e}")
        sys.exit(1)

    if args.table:
        path = Path(args.ladder) if args.ladder else sol.config.RESULTS_DIR / "ladder.json"
        ladder = harness.load_ladder(path)
        print("\n" + harness.render_table(ladder))
        if ladder["stages"]:
            print_failures(ladder, max(ladder["stages"]), path)
        return

    r = Report()
    questions = json.loads((DATA / "questions.json").read_text(encoding="utf-8"))
    if args.dump_labels:
        dump_labels(sol, args, questions)
        return
    if args.kappa:
        run_kappa(sol, args, questions)
        return

    print("\n  Structural checks:")
    for fn_path in ["ingest.policy_metadata", "search.Index", "catalog.normalize_code", "catalog.filter_courses",
                    "tools.search", "tools.search_policies", "tools.search_pages", "tools.get_course",
                    "tools.get_staff", "tools.query_catalog", "tools.my_profile", "tools.get_student",
                    "stages.build_agent", "stages.run", "validator.check_answer"]:
        mod, fn = fn_path.split(".")
        r.check("structure", callable(getattr(getattr(sol, mod), fn, None)), f"{PKG}.{fn_path}")

    print("\n  Data checks:")
    check_data(r, questions)

    print("\n  Behavioral checks (no model, no API key):")
    t0 = time.time()
    try:
        corpus = sol.tools.Corpus.load(dense=False)
    except Exception as e:  # noqa: BLE001
        r.fail("corpus", f"Corpus.load(dense=False) raised {type(e).__name__}: {e}")
        corpus = None
    if corpus is not None:
        r.ok("corpus", f"{len(corpus.index.chunks)} chunks, {len(corpus.catalog)} courses, BM25 only "
                       f"({time.time() - t0:.1f}s)")
        labels = harness.label_sources(corpus)
        r.guard("stages", lambda: check_ready_stages(sol, corpus, r, questions))
        r.guard("hybrid", lambda: check_fusion(sol, r))
        r.guard("search", lambda: check_search(sol, corpus, r, labels))
        check_policies(sol, corpus, r, labels)
        check_exact(sol, corpus, r, questions)
        check_profile(sol, corpus, r)
        check_validator(sol, corpus, r)
    check_harness(r)

    if args.stage:
        run_ladder(sol, r, args, questions)
    else:
        todo = sorted({TODO_OF[n] for n in r.todo})
        print(f"\n  TODO left: {', '.join(todo) or 'none'}  ·  next: uv run eval/eval.py --stage <N>")

    print("\n" + "=" * 60)
    print(f"  {r.passed} passed · {r.failed} failed · {r.skipped} skipped")
    if r.failed:
        print("\n  Failures:")
        for f in r.failures:
            print(f"    - {f}")
        print("=" * 60)
        sys.exit(1)
    print("=" * 60)


if __name__ == "__main__":
    main()
