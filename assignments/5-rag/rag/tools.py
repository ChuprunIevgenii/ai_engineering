"""Agent tools. Each tool is a plain function of (ctx, args) that only reads `ctx.deps`.

Pydantic AI registers them via `Agent(tools=[...])` in stages.py; eval calls them
directly with a stand-in ctx, so every TODO is checkable without a model.

Ready: `Corpus`, `Deps`, `record`, `fmt`, `get_student` (the naive ACL tool).
TODO 2: `search`. TODO 3: `search_policies`, `search_pages`.
TODO 4: `get_course`, `get_staff`, `query_catalog`. TODO 5: `my_profile`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from pydantic_ai import RunContext

from . import api, catalog, config, ingest
from .catalog import Faculty, Kind, Semester
from .ingest import Chunk
from .search import Index


@dataclass
class Corpus:
    """Everything loaded once per process and shared by all runs."""

    index: Index
    catalog: dict[str, catalog.Course]
    staff: dict[str, Chunk]
    students: dict[str, dict]

    @classmethod
    def load(cls, dense: bool = True) -> Corpus:
        chunks = ingest.load_corpus()
        return cls(index=Index(chunks, dense=dense), catalog=catalog.build_catalog(chunks),
                   staff={c.source: c for c in chunks if c.kind == "staff"}, students=api.load_students())


@dataclass
class Deps:
    """Per-run context. `student_id` is who is logged in; it comes from the session, never from the model."""

    corpus: Corpus
    student_id: str = config.DEFAULT_STUDENT
    question: str = ""
    seen: dict[str, str] = field(default_factory=dict)   # citation label -> text the tools returned this run


def record(deps: Deps, label: str, text: str) -> str:
    """Remember what a tool returned under the label the model should cite. Returns `text`.

    Several chunks of one page share a label, so texts under the same label accumulate.
    """
    prev = deps.seen.get(label)
    deps.seen[label] = text if not prev else prev if text in prev else f"{prev}\n{text}"
    return text


def fmt(deps: Deps, chunks: list[Chunk], with_dates: bool = False) -> str:
    """Chunks as «[label] (dates)\\ntext» blocks; with_dates adds validity from chunk.meta (TODO 3)."""
    blocks = []
    for c in chunks:
        extra = ""
        if with_dates and c.kind == "policy" and c.meta.get("effective_from"):
            extra = f" (чинний з {c.meta['effective_from']})"
            if c.meta.get("superseded_by"):
                extra += f" [ВТРАТИВ ЧИННІСТЬ, замінено: {c.meta['superseded_by']}]"
        blocks.append(f"[{c.label()}]{extra}\n{record(deps, c.label(), c.text)}")
    return "\n\n".join(blocks) or "Нічого не знайдено."


# --------------------------------------------------------------------------- stage 2
#
# The docstring of a tool is what the model reads as its description: keep it about the tool, not the TODO.


# TODO 2: hybrid search over the whole corpus (ctx.deps.corpus.index), formatted with `fmt` so every chunk
# carries the label the model must cite and lands in ctx.deps.seen. Visible on q11: the agent reads the
# course card, learns the teacher's name and searches again.
def search(ctx: RunContext[Deps], query: str) -> str:
    """Пошук по всіх документах і сторінках ВНУ: положення, картки дисциплін, викладачі, новини.

    Повертає уривки з міткою джерела в квадратних дужках; цитуй цю мітку.
    """
    return fmt(ctx.deps, ctx.deps.corpus.index.search(query))


# --------------------------------------------------------------------------- stage 3


def is_current(chunk: Chunk) -> bool:
    return chunk.kind == "policy" and not chunk.meta.get("superseded_by")


# TODO 3b: policies only; superseded editions are filtered out BEFORE ranking unless include_superseded,
# and the output shows each chunk's validity dates (fmt(..., with_dates=True)). k=6 is enough.
# Visible on q20 and q22: both editions of the retake rules are in the corpus.
def search_policies(ctx: RunContext[Deps], query: str, include_superseded: bool = False) -> str:
    """Пошук по положеннях і правилах ВНУ (PDF). За замовчуванням лише чинні редакції;
    include_superseded=True додає редакції, що втратили чинність. У виводі — дата чинності.
    """
    where = (lambda c: c.kind == "policy") if include_superseded else is_current
    hits = ctx.deps.corpus.index.search(query, k=6, where=where)
    return fmt(ctx.deps, hits, with_dates=True)


# TODO 3c: the same search over everything that is not a policy: course cards, staff pages, news.
def search_pages(ctx: RunContext[Deps], query: str) -> str:
    """Пошук по сторінках факультетів: картки дисциплін, сторінки викладачів, новини."""
    
    return fmt(ctx.deps, ctx.deps.corpus.index.search(query, where=lambda c: c.kind != "policy"))


# --------------------------------------------------------------------------- stage 4


# TODO 4c: exact lookup, not search. Normalize the code (TODO 4a), return the card text recorded under
# a label with its code (the code itself or the card title, as `fmt` does), or an honest
# «Дисципліни з кодом … немає в каталозі.» Visible on q10: ПЗ-204 sits in the
# prerequisites of 60 other cards, so search alone ranks the right card low.
def get_course(ctx: RunContext[Deps], code: str) -> str:
    """Картка дисципліни за точним кодом, наприклад «ПЗ-204»."""
    norm = catalog.normalize_code(code)
    
    if norm not in ctx.deps.corpus.catalog:
        return f"Дисципліни з кодом {norm} немає в каталозі."
    card = next(c for c in ctx.deps.corpus.index.chunks if c.kind == "course" and c.source == norm)
    return fmt(ctx.deps, [card])


# TODO 4d: staff pages (ctx.deps.corpus.staff) whose title matches every word of `name`. Compare stems,
# not whole words: the first few letters of each query word (five is enough), so «Гордієнко»,
# «Гордієнка» and «Гордієнко С. П.» all find the page; an initial matches as a one-letter prefix or is
# skipped. «Панченко Анна» is one page, not all four Панченко: a stem cut to «ан» also matches the
# patronymic «Андріївна». Record each page under its title; say so when nobody matches. Visible on q11, q31.
def get_staff(ctx: RunContext[Deps], name: str) -> str:
    """Сторінка викладача за прізвищем або ПІБ: кабінет, консультації, пошта."""
    staff = [(c, " ".join(w for w in c.title.split() if not w.endswith(".")))
             for c in ctx.deps.corpus.staff.values()]
    
    pages = ([c for c, full in staff if catalog.name_matches(name, full)]
             or [c for c, full in staff if catalog.name_matches(name, full, any_order=True)])
    
    if not pages:
        return f"Викладача «{name}» не знайдено."
    return fmt(ctx.deps, pages)


# TODO 4e: filter_courses (TODO 4b) over ctx.deps.corpus.catalog; the output is the label line «[каталог]»,
# then the count («Знайдено дисциплін: N») and every course, recorded under the label «каталог».
# The model cites only labels it sees in the output, so a label kept only in deps.seen is useless to it.
# Typed filters (Literal) are the schema the model fills in: it cannot ask for a field that is not there.
def query_catalog(ctx: RunContext[Deps], faculty: Faculty | None = None, year: int | None = None,
                  semester: Semester | None = None, kind: Kind | None = None, teacher: str | None = None,
                  credits: int | None = None) -> str:
    """Підрахунок і список дисциплін каталогу за фільтрами (усі необов'язкові).

    faculty — ФІТ, ЕКФ або ФГН; year — курс 1–4; teacher — прізвище викладача, можна з ім'ям: «Олійник» або «Олійник Андрій».
    Для питань «скільки» і «які саме» — точне число з усього каталогу, а не з top-k.
    Перший рядок — мітка джерела в квадратних дужках; цитуй цю мітку.
    """
    rows = catalog.filter_courses(ctx.deps.corpus.catalog, faculty=faculty, year=year, semester=semester, kind=kind, teacher=teacher, credits=credits)
    lines = [f"{c.code} {c.name} — {c.faculty}, {c.year} курс, {c.semester}, {c.kind}, "f"{c.credits} кредитів ЄКТС, викладач: {c.teacher}" for c in rows]
    text = f"Знайдено дисциплін: {len(rows)}\n" + "\n".join(lines)
    
    return "[каталог]\n" + record(ctx.deps, "каталог", text)


# --------------------------------------------------------------------------- stage 5


# TODO 5: the logged-in student's record. The id comes from ctx.deps.student_id, never from a parameter:
# the tool has no argument the model could fill with someone else's id. Compare with get_student below
# and the ACL line of the table. Like get_student, start the output with the label you record under,
# e.g. «[профіль студента]». Visible on q16-q19 (different students).
def my_profile(ctx: RunContext[Deps]) -> str:
    """Профіль залогіненого студента: факультет, курс, форма фінансування, оцінки, середній бал, борги.

    Перший рядок — мітка джерела в квадратних дужках; цитуй цю мітку.
    """
    rec = api.get_student_record(ctx.deps.corpus.students, ctx.deps.student_id)
    
    if rec is None:
        return "Профіль поточного студента не знайдено."
    label = "профіль студента"
    return f"[{label}]\n" + record(ctx.deps, label, json.dumps(rec, ensure_ascii=False))


def get_student(ctx: RunContext[Deps], student_id: str) -> str:
    """Профіль студента за ID: оцінки, середній бал, борги.
    Перший рядок — мітка джерела в квадратних дужках; цитуй цю мітку.

    Naive ACL on purpose (stage 5-naive): the model chooses the id, and the only
    protection is a sentence in the instructions. Do not copy this design.
    """
    rec = api.get_student_record(ctx.deps.corpus.students, student_id)
    if rec is None:
        return "Студента не знайдено."
    label = f"студент {rec['student_id']}"
    return f"[{label}]\n" + record(ctx.deps, label, json.dumps(rec, ensure_ascii=False))
