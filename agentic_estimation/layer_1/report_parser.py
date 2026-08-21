"""PDF -> text for company sustainability/ESG reports.

The repo had PDF parsing only as CLI probe scripts (calibration/pdf_parser_test.py,
opendataloader_test.py, parser_shootout.py). Nothing importable, so nothing in the
scoring path could ever read a report. This is the reusable half those probes were
written to inform.

TWO BACKENDS, deliberately ordered:

  pdfplumber  -- default. Pure Python, always available, ~1-3s/page.
  opendataloader (ODL) -- opt-in via ESG_PDF_BACKEND=opendataloader. Measured
      decisively better on the ICICI report (recovered the full energy KPI table
      with labels and both years, un-mirrored reversed SDG text, preserved GRI
      index structure) BUT it shells out to `java` -- verified absent from PATH
      on this machine, where it raises FileNotFoundError. So it can never be the
      default; when selected and unavailable it falls back rather than failing
      the company.

WHY EXTRACT_TEXT AND NOT EXTRACT_TABLES: parser_shootout measured that
pdfplumber's extract_tables() drops row labels to None on these layouts while
extract_text() recovers them. A KPI number without its label is worse than
useless to the claim extractor -- it is a hallucination invitation.

Reports are 100-700 pages. Parsing every page of every report is minutes per
company for content that is mostly boilerplate, so parsing is bounded by
_MAX_PAGES and the result is cached to disk next to the PDF.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Optional

from agentic_estimation.shared.pipeline_logger import get_logger

log = get_logger("report_parser")

# pdfminer logs one warning per malformed object; a 700-page report emits
# thousands. They are not actionable here.
logging.getLogger("pdfminer").setLevel(logging.ERROR)

_MAX_PAGES = int(os.getenv("ESG_PDF_MAX_PAGES", "120"))

# OpenDataLoader is the DEFAULT. Measured on Toyota Industries Report 2021
# (40pp), same file, same machine:
#     pdfplumber      298k chars, flat prose, 25s
#     opendataloader  493k chars, 116 markdown table rows, 9s
# The difference is not cosmetic. The CSR scorecard -- target, deadline and
# actual result for every material issue -- survives as table rows under ODL:
#     "Ratio of female managers (non-consolidated) 3.6% (2031) 1.6% P. 53-54"
# while pdfplumber flattens it into prose where "female managers" cannot be
# found at all. Those rows are the densest ESG features in the document.
_BACKEND = os.getenv("ESG_PDF_BACKEND", "opendataloader").lower()
_CACHE_DIR = Path("raw_esg_data/parsed_text")

# A PDF whose text layer is missing (pure scan) or mojibake yields character
# soup. Feeding that to an LLM produces confident nonsense, so it is rejected.
_MIN_CHARS = 400

# Page separator injected by the pdfplumber backend. Consumers split on this to
# recover page boundaries, which are otherwise unrecoverable from concatenated
# text. Chosen to be something no real report contains.
PAGE_MARK = "\n\x0c[[PAGE]] "

# Replacement chars NOT part of a long run. Runs of 4+ are table-of-contents
# dot leaders, which say nothing about whether the prose is readable.
_ISOLATED_BAD_RE = re.compile("(?<!�)�{1,3}(?!�)")


# STACKED-GLYPH REPAIR
#
# Toyota Industries' report renders headline text as 2-3 overlapping copies of
# the same glyphs, so pdfplumber reads every character repeatedly:
#   "vveehhiicclleess cceenntteerreedd aarroouunndd lliifftt ttrruucckkss"
#
# _looks_garbled does NOT catch this: the alphabetic ratio and the space ratio
# are both healthy and the text stays readable to a human. The danger is
# numeric -- "2,118.3" reads as "22,111188..33", so any figure lifted from an
# affected line is silently wrong, which is worse than no figure at all.
#
# Rejecting the document would be the wrong remedy (most pages are clean), so
# affected TOKENS are repaired and the rest of the document is kept.
def _dedupe_run(word: str) -> str:
    """'vveehhiicclleess' -> 'vehicles'. Returns word unchanged if not doubled.

    Only collapses when EVERY character pairs up, so genuine doubles ("committee",
    "1,200") are untouched -- 'committee' fails the all-pairs test on 'com'.
    """
    for k in (3, 2):                       # try triples before doubles
        if len(word) % k:
            continue
        groups = [word[i:i + k] for i in range(0, len(word), k)]
        if all(len(set(g)) == 1 for g in groups) and len(groups) >= 4:
            return "".join(g[0] for g in groups)
    return word


def _repair_duplicated_glyphs(text: str) -> tuple[str, int]:
    """Collapse stacked-glyph runs. Returns (repaired_text, n_repaired)."""
    n = 0

    def fix(m):
        nonlocal n
        w = m.group(0)
        out = _dedupe_run(w)
        if out != w:
            n += 1
        return out

    # Words AND numbers. Numbers matter more: "2,118.3" stacks to
    # "22,,111188..33", which \w+ would not match because of the comma and dot,
    # so the numeric class is handled by its own pattern.
    repaired = re.sub(r"\b\w{8,}\b", fix, text)
    repaired = re.sub(r"[0-9][0-9,.]{7,}", fix, repaired)
    return repaired, n


def _looks_garbled(text: str) -> bool:
    """True when the text layer is unusable.

    Salvaged from calibration/pdf_parser_test.py. Two independent failures:
    scanned pages (no text layer -> almost nothing extracted) and broken CMaps
    (text extracted, but as replacement chars / control bytes).
    """
    if not text or len(text) < _MIN_CHARS:
        return True
    letters = sum(c.isalpha() for c in text)
    if letters / max(1, len(text)) < 0.45:
        return True
    # Real prose has spaces. Broken CMaps often yield one giant token run.
    #
    # Count ALL whitespace, not just " ". OpenDataLoader emits markdown, whose
    # table rows and headings put newlines where prose puts spaces: Toyota's
    # markdown scored a space fraction of exactly 0.080 against a 0.08 floor and
    # was rejected as "garbled" despite a healthy 0.70 alphabetic fraction and
    # zero replacement characters. A whole correctly-parsed 493k-char document
    # was discarded on a rounding margin.
    if sum(c.isspace() for c in text) / max(1, len(text)) < 0.08:
        return True

    # Replacement characters: a REAL broken CMap corrupts words throughout the
    # document. A table of contents corrupts only its DOT LEADERS -- the
    # "Governance & principles ......... 4" filler, which many PDFs encode as a
    # glyph pdfminer cannot map.
    #
    # An earlier version tested `count > 1% of length` and rejected Solvay's
    # 2022 report outright: 638 replacement chars, ALL of them TOC leaders, in
    # a document whose alphabetic fraction was a healthy 0.79. A whole readable
    # report was discarded over page-2 formatting.
    #
    # So: ignore runs of >=4 (leader dots are always long runs) and judge on
    # what is left, which is the corruption that actually touches words.
    scattered = len(_ISOLATED_BAD_RE.findall(text))
    return scattered > len(text) * 0.005


def _cache_path(pdf: Path, page_range: Optional[tuple[int, int]] = None) -> Path:
    # page_range is part of the key: the same PDF parsed over a different page
    # window is different text, and reusing one for the other silently returns
    # the wrong content.
    key = str(pdf.resolve()) + (f"|{page_range[0]}-{page_range[1]}" if page_range else "")
    h = hashlib.sha1(key.encode("utf-8")).hexdigest()[:10]
    return _CACHE_DIR / f"{pdf.stem[:70]}__{h}.json"


def _parse_pdfplumber(pdf: Path, max_pages: int,
                      page_range: Optional[tuple[int, int]] = None) -> tuple[str, int]:
    import pdfplumber

    chunks: list[str] = []
    with pdfplumber.open(pdf) as doc:
        total = len(doc.pages)
        # A combined annual report buries its sustainability statement in the
        # middle (Vestas: pages 60-160 of 195). Reading the first max_pages
        # therefore returns financial front-matter and misses the ESG content
        # entirely. When the caller knows the range -- SRN publishes
        # pdfpage_sust_start/end for 1,886 of its 1,898 reports -- honour it.
        if page_range:
            lo = max(0, page_range[0] - 1)
            hi = min(total, page_range[1])
            pages = list(enumerate(doc.pages))[lo:hi][:max_pages]
        else:
            pages = list(enumerate(doc.pages[:max_pages]))
        for i, page in pages:
            try:
                t = page.extract_text() or ""
            except Exception as exc:      # one bad page must not lose the report
                log.debug("%s p%d: %s", pdf.name, i, type(exc).__name__)
                continue
            if t.strip():
                # PAGE_MARK, not "\n": pdfplumber emits no blank lines between
                # pages, so a plain join produces ONE unsplittable run of text.
                # Downstream selection then saw a single block and silently
                # discarded 99% of the document (measured: a 277k-char report
                # collapsed to one 2,400-char block -- the cover and contents
                # page -- which is exactly the content worth skipping).
                chunks.append(f"{PAGE_MARK}{i + 1}\n{t}")
    return "\n".join(chunks), total


def _ensure_java() -> bool:
    """Put a JRE on PATH. OpenDataLoader shells out to `java -jar`.

    A JRE exists on this machine but is NOT on PATH -- one bundled with DBeaver
    Portable, one under a scratchpad directory. `shutil.which("java")` therefore
    reports nothing, which previously led to the wrong conclusion that Java was
    not installed and to pdfplumber being made the default. Search the known
    locations before concluding it is unavailable.
    """
    import shutil as _sh

    if _sh.which("java"):
        return True

    cands = []
    if os.environ.get("JAVA_HOME"):
        cands.append(Path(os.environ["JAVA_HOME"]) / "bin")
    cands.append(Path(r"C:\portapps\dbeaver-portable\app\jre\bin"))
    tmp = Path(os.environ.get("TEMP", "")) / "claude"
    if tmp.exists():
        cands += [p.parent for p in tmp.glob("*/*/scratchpad/jre/*/bin/java.exe")]

    for d in cands:
        if d and (d / "java.exe").exists():
            os.environ["PATH"] = f"{d}{os.pathsep}" + os.environ.get("PATH", "")
            os.environ.setdefault("JAVA_HOME", str(d.parent))
            log.info("using java at %s", d)
            return True
    return False


def _parse_opendataloader(pdf: Path, max_pages: int) -> tuple[str, int]:
    """ODL markdown. Raises if java is unavailable -- caller falls back."""
    import tempfile

    if not _ensure_java():
        raise RuntimeError("no java runtime found for opendataloader")

    import opendataloader_pdf

    with tempfile.TemporaryDirectory() as tmp:
        # The two APIs take DIFFERENT keyword names, so they cannot share a call:
        #   convert(input_path, output_dir=..., format="markdown")   -- current
        #   run(input_path, output_folder=..., generate_markdown=True) -- deprecated
        # Passing run()'s kwargs to convert() raises TypeError, which the caller
        # would have swallowed as "opendataloader unavailable" and silently
        # fallen back to pdfplumber.
        if hasattr(opendataloader_pdf, "convert"):
            opendataloader_pdf.convert(input_path=[str(pdf)], output_dir=tmp,
                                       format="markdown", quiet=True)
        else:
            opendataloader_pdf.run(input_path=str(pdf), output_folder=tmp,
                                   generate_markdown=True)
        mds = list(Path(tmp).rglob("*.md"))
        if not mds:
            raise RuntimeError("opendataloader produced no markdown")
        text = max(mds, key=lambda p: p.stat().st_size).read_text(
            encoding="utf-8", errors="replace")
    return _strip_image_refs(text), 0


# ODL writes one markdown image reference per embedded image, and the path it
# writes is derived from the INPUT FILENAME:
#
#     ![](<SonoSuite__Sonoco_2024CSR_FullReport_images/imageFile1.png>)
#
# Our downloads are named "{company}__{title}.pdf", so every such reference
# repeats the company name we ASSIGNED to the file. On a 300-image report that
# is the assigned name several hundred times, in text that is supposed to be
# independent evidence of whose report this is.
#
# That silently inverted the entity check: it counts name occurrences, so every
# document contained its assigned name hundreds of times and matched itself.
# Sonoco's report was accepted as SonoSuite's on 143 such placeholders. The
# check was written against pdfplumber, which emits no image markup, so this
# only became reachable when ODL became the default backend.
#
# Stripped at the source rather than in the entity check: this text also feeds
# claim extraction and the scorer, and "sonosuite" repeated 143 times is
# corrupting evidence there too, just less visibly.
_IMG_REF_RE = re.compile(r"!\[[^\]]*\]\(<[^>]*>\)|!\[[^\]]*\]\([^)]*\)")


def _strip_image_refs(text: str) -> str:
    """Drop markdown image references, which carry the input filename."""
    return _IMG_REF_RE.sub(" ", text)


def parse_pdf(pdf_path: str | Path, max_pages: int = _MAX_PAGES,
              use_cache: bool = True,
              page_range: Optional[tuple[int, int]] = None) -> dict:
    """Extract text from one report PDF.

    Returns {ok, text, chars, pages_total, pages_parsed, backend, elapsed_s,
    error}. Never raises -- a broken PDF is a missing signal, not a dead run.
    """
    pdf = Path(pdf_path)
    out = {"ok": False, "text": "", "chars": 0, "pages_total": 0,
           "pages_parsed": 0, "backend": None, "elapsed_s": 0.0,
           "error": None, "path": str(pdf)}
    if not pdf.exists():
        out["error"] = "file not found"
        return out

    cp = _cache_path(pdf, page_range)
    if use_cache and cp.exists():
        try:
            cached = json.loads(cp.read_text(encoding="utf-8"))
            cached["cached"] = True
            return cached
        except Exception:
            pass          # a corrupt cache entry just means re-parsing

    t0 = time.time()
    order = ([_parse_opendataloader, _parse_pdfplumber]
             if _BACKEND.startswith("open") else [_parse_pdfplumber])
    text, total, backend = "", 0, None
    for fn in order:
        try:
            # Only the pdfplumber backend can honour a page window; ODL
            # converts whole documents. Passing page_range to it would be a
            # TypeError, so it is filtered here rather than silently ignored.
            if fn is _parse_pdfplumber:
                text, total = fn(pdf, max_pages, page_range)
            else:
                text, total = fn(pdf, max_pages)
            backend = fn.__name__.replace("_parse_", "")
            if text.strip():
                break
        except Exception as exc:
            log.warning("%s via %s: %s: %s", pdf.name, fn.__name__,
                        type(exc).__name__, str(exc)[:120])
            continue

    out["elapsed_s"] = round(time.time() - t0, 1)
    out["backend"] = backend
    out["pages_total"] = total
    if not text.strip():
        out["error"] = "no text extracted"
        return out
    # Repair BEFORE the garbled check and before whitespace normalisation: a
    # stacked-glyph document passes _looks_garbled but yields corrupted numbers,
    # and repairing after normalisation would leave the damaged runs in place.
    text, n_fixed = _repair_duplicated_glyphs(text)
    if n_fixed:
        out["glyph_repairs"] = n_fixed
        log.info("%s: repaired %d stacked-glyph tokens", pdf.name, n_fixed)

    if _looks_garbled(text):
        out["error"] = f"garbled text layer ({len(text)} chars)"
        return out

    # Whitespace normalisation must not destroy PAGE_MARK -- it is the only
    # record of page boundaries, and consumers split on it. PAGE_MARK contains
    # a single "\n" plus \x0c, so neither rule below can touch it; keep it that
    # way if these patterns are ever changed.
    text = re.sub(r"[ \t]{3,}", "  ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    assert PAGE_MARK.strip("\n") in text or PAGE_MARK not in text
    out.update(ok=True, text=text, chars=len(text),
               pages_parsed=min(total or max_pages, max_pages))

    if use_cache:
        try:
            _CACHE_DIR.mkdir(parents=True, exist_ok=True)
            cp.write_text(json.dumps(out), encoding="utf-8")
        except Exception as exc:
            log.debug("cache write failed for %s: %s", pdf.name, exc)
    return out
