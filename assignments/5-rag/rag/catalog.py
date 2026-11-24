"""Course catalog: course cards parsed into typed rows.

`parse_card` and `build_catalog` are ready; `normalize_code` and `filter_courses` are TODO 4.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from .ingest import Chunk

Faculty = Literal["ФІТ", "ЕКФ", "ФГН"]
Semester = Literal["осінній", "весняний"]
Kind = Literal["вибіркова", "обов'язкова"]


@dataclass(frozen=True)
class Course:
    code: str
    name: str
    faculty: str      # ФІТ | ЕКФ | ФГН
    year: int
    semester: str     # осінній | весняний
    kind: str         # вибіркова | обов'язкова
    credits: int
    teacher: str
    room: str
    prereq: tuple[str, ...]


def parse_card(text: str) -> dict[str, str]:
    return dict(re.findall(r"^([^\n:#]+): (.+)$", text, flags=re.M))


def build_catalog(chunks: list[Chunk]) -> dict[str, Course]:
    out = {}
    for c in chunks:
        if c.kind != "course":
            continue
        f = parse_card(c.text)
        prereq = f.get("Пререквізити", "немає")
        out[c.source] = Course(
            code=c.source,
            name=c.title.split(" ", 1)[1],
            faculty=re.search(r"\((\w+)\)", f["Факультет"]).group(1),
            year=int(f["Курс"]),
            semester=f["Семестр"],
            kind=f["Тип"],
            credits=int(f["Кредити ЄКТС"]),
            teacher=f["Викладач"],
            room=f["Аудиторія"],
            prereq=() if prereq == "немає" else tuple(p.strip() for p in prereq.split(",")),
        )
    return out


def normalize_code(code: str) -> str:
    """TODO 4a: 'пз 204', ' ПЗ-204 ', 'пз-204', 'ПЗ–204' (en dash), 'ПЗ204' -> 'ПЗ-204'.
    Latin look-alikes are not handled.

    Anything that does not look like a code comes back stripped and upper-cased, so get_course can say
    «no such course» instead of crashing.
    """
    m = re.match(r"^\s*([А-ЯҐЄІЇа-яґєії]{2})\s*[-–—]?\s*(\d+)\s*$", code)
    
    if not m:
        return code.strip().upper()
    return f"{m.group(1).upper()}-{m.group(2)}"

def _name_words(text: str) -> list[str]:
    """Words of a name, split on spaces and dots, lower-cased: 'Олійник А. М.' -> ['олійник', 'а', 'м']."""
    return re.findall(r"[^\s.]+", text.lower())

def name_matches(query: str, full_name: str, any_order: bool = False) -> bool:
    """Each query word, cut to a 5-letter stem, starts a word of the full name: 'Гордієнка' and
    'Гордієнко С. П.' match 'Гордієнко Світлана Петрівна'; 'Харченко' does not match 'Захарченко'.
    By default by position (surname, name, patronymic); any_order=True lets the words come in any
    order ('Василь Головко'), each query word taking a different word of the name."""
    
    q = [w[:5] for w in _name_words(query)]
    words = _name_words(full_name)
    
    if not 0 < len(q) <= len(words):
        return False
    if not any_order:
        return all(words[i].startswith(s) for i, s in enumerate(q))
    
    free = list(words)
    
    for s in sorted(q, key=len, reverse=True):
        hit = next((w for w in free if w.startswith(s)), None)
        if hit is None:
            return False
        free.remove(hit)
    
    return True

def filter_courses(catalog: dict[str, Course], faculty: str | None = None, year: int | None = None,
                   semester: str | None = None, kind: str | None = None, teacher: str | None = None,
                   credits: int | None = None) -> list[Course]:
    """TODO 4b: courses matching every given filter (None = no filter), sorted by code.

    `teacher` is a surname, optionally followed by the name and patronymic or initials, case-insensitive;
    each word is a prefix of the matching word of the full name: 'Олійник', 'Олійник Андрій' and
    'Олійник А. М.' match 'Олійник Андрій Миколайович', 'Олійник Марія' does not.
    Visible on q13, q14, q15: an exact count over all 396 courses, which no top-k can give.
    On q15 the model passes the full name, and the surname is shared by five teachers.
    """
    rows = [c for c in catalog.values()
            if (faculty is None or c.faculty == faculty)
            and (year is None or c.year == year)
            and (semester is None or c.semester == semester)
            and (kind is None or c.kind == kind)
            and (credits is None or c.credits == credits)
            and (teacher is None or name_matches(teacher, c.teacher))]
    return sorted(rows, key=lambda c: c.code)
