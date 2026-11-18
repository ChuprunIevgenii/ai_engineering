"""Stage 6 (bonus): check the answer against what the tools returned in this run.

Three rules, all deterministic:
  1. every [citation] points to a source a tool returned in this run;
  2. every number in the answer occurs in the returned sources (labels or texts) or in the question;
  3. links and domains only from the allowlist.
`check_answer` returns a list of problems; the stage turns them into `ModelRetry`.
"""

from __future__ import annotations

import re

from .tools import Deps

ALLOWED_DOMAINS = ("vnu.example",)

CITATION_RE = re.compile(r"\[([^\[\]]{2,200})\]")
NUMBER_RE = re.compile(r"\d+(?:[.,:]\d+)*")
DATE_RE = re.compile(r"^\d{1,2}\.\d{2}\.\d{4}$")
LIST_MARKER_RE = re.compile(r"^\s*\d+[.)]\s", re.M)
DOMAIN_RE = re.compile(r"(?:https?://)?(?:[\w-]+\.)+example\b", re.I)


def _norm(text: str) -> str:
    text = text.lower().replace("\u00a0", " ").replace("\u202f", " ")
    return re.sub(r"(?<=\d) (?=\d{3}\b)", "", text)   # «1 150 грн» -> «1150 грн»


def _numbers(text: str) -> set[str]:
    out = set()
    for tok in NUMBER_RE.findall(_norm(text)):
        tok = tok.replace(",", ".")
        out.add(re.sub(r"\.0+$", "", tok))
    return out


def _cited_ok(citation: str, deps: Deps) -> bool:
    cit = _norm(citation)
    for label in deps.seen:
        lab = _norm(label)
        if lab in cit or cit in lab:
            return True
    return False


def check_answer(answer: str, deps: Deps) -> list[str]:
    """Bonus: problems with the answer, each phrased as an instruction the model can act on; [] if none.
    «Wrong citation» alone is not actionable: name the labels it may cite instead.

    deps.seen maps every label a tool returned this run to its text. Helpers above are ready:
    CITATION_RE, _cited_ok, _numbers (normalizes «1 150» and «4,0»), LIST_MARKER_RE (list numbers are
    not facts), DATE_RE, DOMAIN_RE. Visible on q26: a phishing link injected into a staff page.
    """
    raise NotImplementedError("bonus: validator.check_answer")
