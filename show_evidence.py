"""
Show the evidence trail for a company: what was fetched, what was extracted
from it, and how that fed the score.

Usage:
    python show_evidence.py "Coursera"              # summary
    python show_evidence.py "Coursera" --full       # + full signal text
    python show_evidence.py --list                  # all available companies

Reads the frozen benchmark corpus (calibration/abl_final_*.json) -- the exact
evidence the 12 scoring approaches in BENCHMARK.md were run against. Nothing is
re-fetched, so what prints here is provably the same evidence the numbers came
from.
"""

import argparse
import json
import os
import re
import sys

# The corpus holds real company names with accents/CJK ("Fazenda da Toca
# Organicos", "ISIGNY SAINTE-MERE"). The Windows console defaults to cp1252 and
# raises UnicodeEncodeError on those mid-print, killing the run partway through
# a demo. Reconfigure to UTF-8 where supported, and fall back to replacing
# unencodable chars so output degrades to "?" instead of a traceback.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, OSError):
    pass

CORPORA = [
    ("tune", os.path.join("calibration", "abl_final_tune400.json")),
    ("holdout", os.path.join("calibration", "abl_final_holdout68.json")),
]

_HERE = os.path.dirname(os.path.abspath(__file__))


def _load_all() -> list[tuple[str, dict]]:
    out = []
    for tag, rel in CORPORA:
        path = os.path.join(_HERE, rel)
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            for rec in json.load(f):
                out.append((tag, rec))
    return out


def _find(records, query: str):
    q = query.strip().lower()
    exact = [(t, r) for t, r in records if r.get("name", "").lower() == q]
    if exact:
        return exact[0]
    partial = [(t, r) for t, r in records if q in r.get("name", "").lower()]
    if len(partial) == 1:
        return partial[0]
    if len(partial) > 1:
        print(f"'{query}' matches {len(partial)} companies:")
        for _, r in partial[:20]:
            print("   ", r.get("name"))
        sys.exit(1)
    return None, None


def _rule(char="=", width=78):
    print(char * width)


# Collectors append source links inline as "<https://...>" markers after each
# snippet/headline (see signal_agent._ddg_fallback / _google_news_rss_query).
_URL_RE = re.compile(r"<(https?://[^>\s]+)>")


def _extract_urls(text: str) -> list[str]:
    """Unique source URLs embedded in a signal's text, in first-seen order."""
    seen, out = set(), []
    for u in _URL_RE.findall(text or ""):
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def _strip_urls(text: str) -> str:
    """Signal text with the inline <url> markers removed, for clean previews."""
    return _URL_RE.sub("", text or "")


# How many individual items (headlines / snippets) to show per source before
# collapsing the rest behind --full.
_MAX_ITEMS_SHOWN = 6

# A source's text is a run of "<chunk> <url>" pairs -- news feeds join them
# with newlines, DDG snippets with spaces. Capturing the text preceding each
# <url> marker recovers the pairing either way.
_ITEM_RE = re.compile(r"([^<>\n]+?)\s*<(https?://[^>\s]+)>")


def _split_items(text: str) -> list[tuple[str, str]]:
    """Split a signal into (label, url) pairs, one per cited item.

    Returns [] for text with no inline links (pre-URL-capture records), letting
    the caller fall back to a plain preview.
    """
    out = []
    for label, url in _ITEM_RE.findall(text or ""):
        label = " ".join(label.split()).strip(" -:;,")
        # Strip the "Reuters (via Google News RSS):" style prefix the collector
        # puts on the first chunk only -- the source column already says it.
        label = re.sub(r"^[A-Z][\w /&-]{0,40}(?:\([^)]*\))?\s*:\s*", "", label, count=1)
        if label:
            out.append((label, url))
    return out


def _compute_scores(pillar: str, pillar_rec: dict, holistic_val):
    """Recompute this pillar's formula score and final blended score.

    The corpus stores formula INPUTS (baseline, contributions, peer anchor) but
    not the resulting score -- the replay harness recomputes it per variant, so
    nothing final is persisted. Rather than re-derive the arithmetic here (which
    would drift from production the moment the formula changes), this calls the
    same saturation + reconcile functions the pipeline and the benchmark use.

    Returns (formula_score, final_score); either may be None if the record is
    missing the inputs that step needs.
    """
    try:
        sys.path.insert(0, _HERE)
        from calibration.ablation_replay import _saturate, _reconcile_score
    except Exception:
        return None, None

    contribs = pillar_rec.get("contributions") or []
    formula_score = None
    final_score = None
    try:
        formula_score = _saturate(pillar_rec, pillar, contribs).score
    except Exception:
        pass
    try:
        final_score = _reconcile_score(pillar, pillar_rec, contribs, holistic_val)
    except Exception:
        pass
    return formula_score, final_score


def _pretty_url(url: str) -> str:
    """Google News wraps every article in an opaque /rss/articles/CBMi... redirect.
    It resolves fine in a browser but is unreadable on screen, so it's labelled
    rather than printed in full. Direct publisher links print as-is."""
    if "news.google.com/rss/articles/" in url:
        return f"[Google News link] {url[:60]}..."
    return url


def show(tag: str, rec: dict, full: bool = False) -> None:
    name = rec.get("name", "?")
    meta = rec.get("metadata") or {}
    _rule()
    print(f"  {name}")
    print(f"  corpus: {tag} set   |   country: {rec.get('country', '?')}"
          f"   |   identity matched via: {meta.get('source', '?')}")
    _rule()

    # ---- 1. What was fetched -------------------------------------------
    signals = rec.get("signals") or {}
    print(f"\n[1] EVIDENCE FETCHED  --  {len(signals)} sources returned content\n")
    total_links = 0
    for src, text in signals.items():
        text = (text or "").strip()
        links = _extract_urls(text)
        total_links += len(links)
        if full:
            print(f"  --- {src} ({len(text)} chars) ---")
            print("  " + text.replace("\n", "\n  "))
            print()
        else:
            print(f"  * {src:<28} {len(text):>6} chars"
                  f"{f'   [{len(links)} link(s)]' if links else ''}")
            items = _split_items(text)
            if items:
                # Headline/snippet paired with its own link, so each line is a
                # readable citation rather than a wall of opaque redirect URLs.
                for label, url in items[:_MAX_ITEMS_SHOWN]:
                    print(f"      - {label[:150]}")
                    if url:
                        print(f"        {_pretty_url(url)}")
                if len(items) > _MAX_ITEMS_SHOWN:
                    print(f"      ... {len(items) - _MAX_ITEMS_SHOWN} more"
                          f" (use --full to see all)")
            else:
                preview = " ".join(_strip_urls(text).split())[:220]
                if preview:
                    print(f"      {preview}...")
    if not signals:
        print("  (no signals recorded)")
    if signals and total_links == 0:
        print("\n  NOTE: this record predates source-URL capture, so it stores the")
        print("        outlet name and headline but not the article link. Companies")
        print("        gathered from now on include the URL for every source.")

    # ---- 2. What was extracted from it ---------------------------------
    kept = rec.get("kept_claims") or []
    raw = rec.get("raw_claims") or []
    dropped = max(0, len(raw) - len(kept))
    print(f"\n[2] CLAIMS EXTRACTED  --  {len(kept)} kept"
          f"{f', {dropped} dropped by validation' if dropped else ''}\n")
    for c in kept:
        pillar = c.get("pillar", "?")
        factor = c.get("factor", "?")
        pol = c.get("polarity", 0)
        conf = c.get("confidence", 0)
        src = c.get("source_tag", "?")
        sign = "+" if pol > 0 else ("-" if pol < 0 else "0")
        print(f"  [{pillar}] {factor:<28} {sign}  conf={conf:.2f}   from: {src}")
        reasoning = (c.get("reasoning") or "").strip()
        if reasoning:
            print(f"      \"{' '.join(reasoning.split())}\"")
        # Link(s) for the signal this claim cites -- the audit trail from a
        # score back to a page someone can actually open.
        for u in _extract_urls(signals.get(src, "")):
            print(f"      -> {u}")
    if not kept:
        print("  (no claims survived extraction/validation)")

    # ---- 3. Flags (why things were dropped or capped) ------------------
    flags = rec.get("flags") or []
    if flags:
        print(f"\n[3] VALIDATION FLAGS  --  {len(flags)}\n")
        for f in flags:
            print(f"  ! {f}")

    # ---- 4. How the score was built ------------------------------------
    print("\n[4] HOW THE SCORE WAS BUILT\n")
    truth = rec.get("truth") or {}
    holistic = rec.get("holistic") or {}
    pillars = rec.get("pillars") or {}

    def fmt(v):
        return f"{v:.1f}" if isinstance(v, (int, float)) else "  -"

    for pillar in ("E", "S", "G"):
        p = pillars.get(pillar) or {}
        base = p.get("baseline")
        print(f"  --- {pillar} ---")
        print(f"    country baseline ......... {fmt(base)}"
              f"   (source: {p.get('baseline_source', '?')})")

        contribs = p.get("contributions") or []
        if contribs:
            print(f"    evidence adjustments ..... {len(contribs)}")
            wsum = p.get("registry_weight_sum") or 0
            for c in contribs:
                w = c.get("weight", 0)
                conf = c.get("confidence", 0)
                delta = c.get("delta", 0)
                # points = weight * confidence * delta, normalised by weight sum
                pts = (w * conf * delta / wsum * 100) if wsum else 0
                arrow = "up" if pts > 0 else ("down" if pts < 0 else "flat")
                print(f"      {c.get('factor', '?'):<22} w={w:<5} conf={conf:.2f}"
                      f"  delta={delta:+.2f}  -> {pts:+.2f} pts ({arrow})"
                      f"  [{c.get('method', '?')}]")
        else:
            print("    evidence adjustments ..... none (no usable claims)")

        anchor = p.get("peer_anchor")
        if anchor:
            print(f"    peer comparison .......... {anchor.get('basis', '?')}")
            print(f"                               tier={anchor.get('tier', '?')}"
                  f"  n_peers={anchor.get('n_peers', '?')}"
                  f"  conf={anchor.get('confidence', 0):.2f}")

        h = holistic.get(pillar)
        t = truth.get(pillar)
        formula_score, final_score = _compute_scores(pillar, p, h)

        print()
        print(f"    = FORMULA SCORE .......... {fmt(formula_score)}"
              f"   (baseline + adjustments, after saturation)")
        print(f"      AI holistic judgment ... {fmt(h)}"
              f"   {'(unavailable -- formula used alone)' if h is None else ''}")
        print(f"    = ESTIMATED {pillar} SCORE ..... {fmt(final_score)}"
              f"   <-- what the pipeline outputs")
        print(f"      real published rating .. {fmt(t)}"
              f"   (rating agency's own scale -- see note below)")
        print()

    _rule("-")
    print("Every claim above cites the source it was extracted from. Claims citing a")
    print("source that was not actually fetched are discarded before scoring, so no")
    print("part of the score traces back to something the pipeline did not read.")
    print("Note: 'real published rating' is on the rating agency's own scale, not 0-100 --")
    print("      it is used to check ranking order, not compared point-for-point.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("company", nargs="?", help="company name (partial match ok)")
    ap.add_argument("--full", action="store_true",
                    help="print full fetched text for each source, not a preview")
    ap.add_argument("--list", action="store_true", help="list all companies in the corpus")
    args = ap.parse_args()

    records = _load_all()
    if not records:
        print("No corpus files found under calibration/. Expected:")
        for _, rel in CORPORA:
            print("   ", rel)
        sys.exit(1)

    if args.list:
        for tag, r in sorted(records, key=lambda x: x[1].get("name", "")):
            print(f"[{tag:<7}] {r.get('name')}")
        print(f"\n{len(records)} companies total.")
        return

    if not args.company:
        ap.print_help()
        sys.exit(1)

    tag, rec = _find(records, args.company)
    if rec is None:
        print(f"No company matching '{args.company}'. Try --list.")
        sys.exit(1)
    show(tag, rec, full=args.full)


if __name__ == "__main__":
    main()
