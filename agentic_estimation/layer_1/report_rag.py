"""
report_rag.py -- per-company hybrid retrieval over one ESG report.

TWO CHANNELS (see report_structure.py for why)
    tables : selected deterministically by ESG-term / core-metric alias match.
             They never enter similarity ranking, so a data table cannot be
             out-ranked by narrative prose about the same topic.
    prose  : chunked, then ranked per pillar by dense embedding similarity
             UNION sparse keyword score.

WHY HYBRID AND NOT PURE DENSE
    ESG evidence is full of exact tokens where embeddings blur the distinction
    that matters: "Scope 1" vs "Scope 3", "ISO 14001" vs "ISO 45001", "LTIFR".
    Dense retrieval ranks all of those as near-identical. Sparse scoring keeps
    them separable; dense scoring finds the paraphrases sparse misses
    ("we aim to halve our footprint" contains no listed term). Union of both,
    deduped, ranked by max score.

WHY PER-PILLAR AND NOT ONE MERGED QUERY
    A single query vector averaging fifteen ESG concepts is a centroid near
    nothing. Measured on Toyota Industries' 2021 report, merged-concept
    selection ranked the CSR materiality scorecard 19th of 268 blocks -- below
    any sane cutoff -- while it is the densest evidence block in the document.

EMBEDDINGS ARE OPTIONAL
    If sentence-transformers is unavailable the module degrades to sparse-only
    ranking and says so in the result. That keeps the pipeline runnable on a
    machine with no model cached, and makes the embedding contribution
    measurable by A/B rather than assumed.

TEMPORARY BY CONSTRUCTION
    The index is a list of vectors in memory for ONE company and is discarded
    when the call returns. There is no shared store, so cross-company leakage
    is not mitigated -- it is impossible.
"""

from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from agentic_estimation.layer_1.report_structure import (
    METRIC_ALIASES,
    METRIC_PILLAR,
    PILLAR_TERMS,
    ProseBlock,
    TableRecord,
    extract_structure,
    tables_for_pillar,
)
from agentic_estimation.shared.pipeline_logger import get_logger

log = get_logger("report_rag")

# Natural-language query per pillar for the dense half. Written as sentences,
# not keyword bags: sentence embedders are trained on sentences, and a comma
# list embeds to a vaguer point than prose describing the same thing.
PILLAR_QUERIES = {
    "E": ("Greenhouse gas emissions, scope 1 2 and 3 carbon footprint, energy "
          "and electricity consumption, renewable energy share, water "
          "withdrawal, waste generated and recycled, climate targets and "
          "environmental management."),
    "S": ("Employees and workforce, gender diversity and share of women, "
          "training hours, employee turnover, occupational health and safety, "
          "lost time injury rate, human rights, supply chain labour standards "
          "and community engagement."),
    "G": ("Board of directors composition and independence, board committees, "
          "executive remuneration, business ethics and code of conduct, "
          "anti-corruption and bribery, whistleblower mechanism, risk "
          "management, compliance and ESG reporting assurance."),
}

_MODEL_NAME = os.getenv("ESG_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
_CHUNK_CHARS = int(os.getenv("ESG_RAG_CHUNK", "1200"))
_TOP_PROSE = int(os.getenv("ESG_RAG_TOP_PROSE", "12"))
_TOP_TABLES = int(os.getenv("ESG_RAG_TOP_TABLES", "6"))

_model = None
_model_tried = False


def _get_model():
    """Load the embedder once per process. None means sparse-only mode."""
    global _model, _model_tried
    if _model_tried:
        return _model
    _model_tried = True
    try:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(_MODEL_NAME)
        log.info("embedder loaded: %s", _MODEL_NAME)
    except Exception as exc:
        log.warning("no embedder (%s) -- sparse-only retrieval", type(exc).__name__)
        _model = None
    return _model


@dataclass
class Chunk:
    chunk_id: str
    page: Optional[int]
    heading: str
    text: str


def build_chunks(blocks: list[ProseBlock], size: int = _CHUNK_CHARS) -> list[Chunk]:
    """Merge prose blocks into ~size-char chunks, never crossing a heading.

    Heading boundaries are hard: a chunk spanning "Environmental Data" into
    "Board Composition" would be retrieved for both pillars and dilute both.
    """
    out: list[Chunk] = []
    buf: list[str] = []
    cur_head: Optional[str] = None
    cur_page: Optional[int] = None

    def flush() -> None:
        if not buf:
            return
        text = " ".join(buf).strip()
        if len(text) >= 80:
            out.append(Chunk(f"C{len(out) + 1}", cur_page, cur_head or "", text))
        buf.clear()

    for b in blocks:
        if b.heading != cur_head:
            flush()
            cur_head, cur_page = b.heading, b.page
        if sum(len(x) for x in buf) + len(b.text) > size:
            flush()
            cur_page = b.page
        buf.append(b.text)
    flush()
    return out


def _sparse_score(text: str, pillar: str) -> float:
    """Keyword score: pillar topic terms plus core-metric aliases.

    Metric aliases are weighted double for the same reason as in the table
    channel -- a chunk stating a scored metric outranks one merely on-topic.
    """
    low = text.lower()
    score = sum(1.0 for t in PILLAR_TERMS[pillar] if t in low)
    for key, aliases in METRIC_ALIASES.items():
        if METRIC_PILLAR.get(key) != pillar:
            continue
        if any(a in low for a in aliases):
            score += 2.0
    # Length-normalise so a long chunk cannot win on volume alone.
    return score / math.sqrt(max(len(low), 400) / 400.0)


def _dense_scores(chunks: list[Chunk], pillar: str) -> Optional[list[float]]:
    model = _get_model()
    if model is None or not chunks:
        return None
    import numpy as np

    texts = [f"{c.heading}. {c.text}" for c in chunks]
    emb = model.encode(texts, normalize_embeddings=True,
                       show_progress_bar=False, batch_size=32)
    q = model.encode([PILLAR_QUERIES[pillar]], normalize_embeddings=True,
                     show_progress_bar=False)
    return list(np.asarray(emb) @ np.asarray(q)[0])


def _rank_prose(chunks: list[Chunk], pillar: str, top: int) -> list[tuple[Chunk, float, str]]:
    """Hybrid rank. Returns (chunk, score, why) best first.

    Scores are min-max normalised per channel before the union so that a
    cosine in [0,1] and an unbounded keyword count are comparable; without
    that the sparse channel would dominate purely by scale.
    """
    if not chunks:
        return []
    sparse = [_sparse_score(c.text, pillar) for c in chunks]
    dense = _dense_scores(chunks, pillar)

    def norm(xs: list[float]) -> list[float]:
        lo, hi = min(xs), max(xs)
        return [0.0] * len(xs) if hi <= lo else [(x - lo) / (hi - lo) for x in xs]

    ns = norm(sparse)
    nd = norm(dense) if dense is not None else [0.0] * len(chunks)
    rows = []
    for i, c in enumerate(chunks):
        if dense is None:
            score, why = ns[i], "sparse"
        else:
            score = max(ns[i], nd[i])
            why = "dense" if nd[i] > ns[i] else "sparse"
        if sparse[i] <= 0 and (dense is None or nd[i] < 0.35):
            continue                     # no evidence of relevance at all
        rows.append((c, score, why))
    rows.sort(key=lambda r: -r[1])
    return rows[:top]


@dataclass
class PillarContext:
    pillar: str
    tables: list[TableRecord]
    chunks: list[tuple[Chunk, float, str]]
    metrics_present: list[str]

    def as_prompt_text(self, max_chars: int = 14000) -> str:
        """Context block for one extractor. Tables first, and tables never cut.

        Ordering is deliberate: on a quota-limited endpoint the tail is what
        gets dropped, and a truncated data table is worse than absent prose.
        """
        parts: list[str] = []
        if self.tables:
            parts.append("## DATA TABLES (verbatim from the report)")
            for t in self.tables:
                parts.append(t.as_markdown())
        used = sum(len(p) for p in parts)
        if self.chunks:
            parts.append("\n## NARRATIVE EXCERPTS")
            for c, _, _ in self.chunks:
                piece = f"\n[{c.chunk_id} p{c.page}] {c.heading}\n{c.text}"
                if used + len(piece) > max_chars:
                    break
                parts.append(piece)
                used += len(piece)
        return "\n".join(parts)


def retrieve(pdf: Path, pages: str | None = None,
             top_prose: int = _TOP_PROSE,
             top_tables: int = _TOP_TABLES) -> dict[str, PillarContext]:
    """Full per-company retrieval: one PDF in, three pillar contexts out."""
    tables, blocks = extract_structure(pdf, pages=pages)
    chunks = build_chunks(blocks)
    log.info("[%s] %d tables, %d prose blocks -> %d chunks",
             pdf.name[:40], len(tables), len(blocks), len(chunks))

    out: dict[str, PillarContext] = {}
    for pillar in ("E", "S", "G"):
        tabs = tables_for_pillar(tables, pillar, limit=top_tables)
        ranked = _rank_prose(chunks, pillar, top_prose)
        mets = sorted({m for t in tabs for m in t.metrics_for(pillar)})
        out[pillar] = PillarContext(pillar, tabs, ranked, mets)
        log.info("  %s: %d tables, %d chunks, metrics=%s",
                 pillar, len(tabs), len(ranked), mets or "-")
    return out


if __name__ == "__main__":
    import sys

    pdf = Path(sys.argv[1])
    pages = sys.argv[2] if len(sys.argv) > 2 else None
    ctxs = retrieve(pdf, pages=pages)
    for p, c in ctxs.items():
        print(f"\n=== {p} === tables={len(c.tables)} chunks={len(c.chunks)} "
              f"metrics={c.metrics_present}")
        for t in c.tables:
            print(f"   T {t.table_id} p{t.page} {t.n_rows}x{t.n_cols} "
                  f"score={t.score(p)} {t.heading[:44]}")
        for ch, s, why in c.chunks[:5]:
            print(f"   C {ch.chunk_id} p{ch.page} {s:.2f} ({why}) "
                  f"{ch.heading[:40]}")
