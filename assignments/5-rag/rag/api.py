"""Mock university API over JSON fixtures (ready).

In a real system this is an HTTP service behind SSO. Here it is a dict lookup,
so the only thing that decides whose data comes back is the id you pass in.
"""

from __future__ import annotations

import json

from . import config


def load_students() -> dict[str, dict]:
    return json.loads(config.STUDENTS_FILE.read_text(encoding="utf-8"))


def get_student_record(students: dict[str, dict], student_id: str) -> dict | None:
    """Profile, grades, average and debts. No access check: callers decide which id is allowed."""
    return students.get(student_id.strip().upper())
