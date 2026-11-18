"""Corpus loading: policy PDFs page by page, faculty pages, chunks.

Ready except `policy_metadata` (TODO 3).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from . import config


@dataclass
class Chunk:
    id: str
    kind: str          # policy | course | staff | news
    source: str        # policy doc_id, course code, staff id or news slug
    title: str
    page: int | None   # page number for policy chunks
    text: str
    meta: dict = field(default_factory=dict)

    def label(self) -> str:
        """Citation label the model sees and is asked to repeat: «Назва, с. N» or «Назва»."""
        return f"{self.title}, с. {self.page}" if self.page else self.title


def load_policy_meta() -> list[dict]:
    return json.loads(config.POLICY_META.read_text(encoding="utf-8"))


def policy_metadata(meta: dict) -> dict:
    """TODO 3a: metadata every chunk of this policy carries.

    `meta` is one entry of data/policies/meta.json. Return a dict with
    `effective_from` (ISO date) and `superseded_by` (doc_id of the newer edition or None).
    The text of the 2024 retake policy never says it was replaced; only meta.json does.
    Visible on q20 and q22: without it stage 3 cannot tell the 2024 edition from the 2025 one.
    """
    raise NotImplementedError("TODO 3a: ingest.policy_metadata")


def chunk_policy_pages(meta: dict) -> list[Chunk]:
    """Split each PDF page into ~CHUNK_CHARS chunks on line boundaries; chunks keep their page number."""
    from pypdf import PdfReader

    reader = PdfReader(config.DATA_DIR / meta["file"])
    try:
        chunk_meta = policy_metadata(meta)
    except NotImplementedError:   # stages 0-2 work without metadata
        chunk_meta = {}
    out: list[Chunk] = []
    for pno, page in enumerate(reader.pages, 1):
        lines = [ln.strip() for ln in (page.extract_text() or "").splitlines() if ln.strip()]
        buf: list[str] = []
        for ln in [*lines, None]:
            if ln is None or (buf and sum(map(len, buf)) + len(ln) > config.CHUNK_CHARS):
                if buf:
                    out.append(Chunk(f"{meta['doc_id']}#p{pno}c{len(out)}", "policy", meta["doc_id"],
                                     meta["title"], pno, " ".join(buf), dict(chunk_meta)))
                buf = []
            if ln is not None:
                buf.append(ln)
    return out


def load_pages() -> list[Chunk]:
    """Course cards, staff pages and news: one chunk per page (they are short)."""
    out = []
    for kind_dir, kind in (("courses", "course"), ("staff", "staff"), ("news", "news")):
        for p in sorted((config.PAGES_DIR / kind_dir).glob("*.md")):
            text = p.read_text(encoding="utf-8")
            title = text.splitlines()[0].lstrip("# ").strip()
            out.append(Chunk(p.stem, kind, p.stem, title, None, text))
    return out


def load_corpus() -> list[Chunk]:
    return [c for m in load_policy_meta() for c in chunk_policy_pages(m)] + load_pages()
