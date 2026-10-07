"""
evidence_clusters.py — cluster-severity assignment for evidence claims.

WHY: the extractor's per-claim `strength` is an LLM judgment with no empirical
basis (see research/evidence_quantification.md — "how negative? -0.15 or
-0.91?"). This module replaces that per-claim guess with a per-CLUSTER label:
claims are grouped by text similarity (TF-IDF + k-means, fitted offline on the
tune-corpus claims), each cluster is labeled ONCE with an impact-magnitude
level, and at runtime every claim inherits its nearest cluster's label.

Two numbers, two slots — deliberately NOT the same thing:
  cluster label      -> claim.strength   (impact magnitude of this KIND of evidence)
  cosine membership  -> confidence mult. (how sure we are it IS that kind)
Membership is typicality, not severity — a textbook-ordinary event has high
membership; using it as severity would invert exactly the unusual/catastrophic
cases. Hence the split.

This is the interim quantification path: the enforcement-data yardstick
(research/ENFORCEMENT_DATA_SOURCES.md) later replaces cluster labels with
measured percentiles for penalty-bearing claims without touching this plumbing
— both produce (strength, confidence) on the same claims.

Offline (run once, artifacts are committed):
    python -m agentic_estimation.layer_2.evidence_clusters build
    python -m agentic_estimation.layer_2.evidence_clusters label
    python -m agentic_estimation.layer_2.evidence_clusters show

Runtime: apply_cluster_severity(claims) — called from claim_validators.
Degrades to a no-op (original strengths kept) if artifacts are absent.
"""

import json
import pickle
import sys
from pathlib import Path
from typing import Optional

from agentic_estimation.shared.pipeline_logger import get_logger

log = get_logger("evidence_clusters")

_DIR = Path(__file__).resolve().parent
MODEL_PATH = _DIR / "evidence_clusters.pkl"
LABELS_PATH = _DIR / "evidence_clusters.json"

# Claims come from the frozen tune corpus ONLY (never holdout — the holdout
# stays untouched by every fitting step, same rule as the weight tuning).
_TUNE_CORPUS = _DIR.parent.parent / "calibration" / "abl_final_tune400.json"

# k-means k per pillar. Fixed at 3 clusters/pillar (9 total): a tune-vs-holdout
# sweep (calibration/sweep_cluster_k.py) compared this against silhouette-picked
# k (which came out E=4/S=5/G=7=16 total) on both corpora -- k=3 matched or
# beat the finer split on every pillar (notably E: holdout Spearman +0.187 vs
# +0.169), so the extra granularity wasn't earning its complexity.
#
# At k=3 with k-means++'s default (random-among-claims) init, G specifically
# regressed: its fraud/litigation cluster got absorbed into a mixed bucket and
# all 3 G clusters ended up labeled the same severity=0.5. Fixed via
# _SEED_KEYWORDS below (seeded centroids instead of random init) -- see that
# constant's comment for the full story.
_K_RANGE = {"E": (3, 3), "G": (3, 3), "S": (3, 3)}

# Below this cosine similarity to the nearest centroid, a claim is considered
# out-of-vocabulary for the fitted clusters and is left completely untouched.
MIN_MEMBERSHIP = 0.12
# Similarity at (or above) which the confidence multiplier saturates at 1.0.
_SIM_SATURATION = 0.45
# Floor of the confidence multiplier at MIN_MEMBERSHIP.
_CONF_FLOOR = 0.7

_VALID_SEVERITIES = (0.25, 0.5, 0.75, 1.0)

# Keyword-seeded starting centroids, one group per pillar per cluster (k=3
# each, matching _K_RANGE). Replaces k-means++'s random-among-claims init,
# which caused a real regression at k=3: G's fraud/litigation cluster got
# absorbed into a mixed board-independence bucket and ALL 3 G clusters ended
# up labeled severity=0.5 (see conversation -- a routine "published an ESG
# report" claim and a "securities fraud lawsuit" claim inherited the SAME
# severity). Seeding centroids at severity-distinct concepts up front (severe
# legal exposure / routine certification+policy / routine disclosure
# boilerplate) gives k-means a starting point already spread across the
# severity range we care about, instead of hoping random initialization finds
# it. Each group's seed text is vectorized through the SAME fitted TF-IDF
# vectorizer as the claims, then L2-normalized, exactly like a real claim --
# no special-casing at inference time.
_SEED_KEYWORDS: dict[str, list[str]] = {
    "E": [
        # severe: verified/enforced climate events and violations
        "environmental violation pollution spill fine penalty enforcement controversy",
        # routine: voluntary disclosure/reporting practices
        "cdp disclosure carbon climate signal reporting transparency",
        # moderate: concrete commitments and targets
        "net zero pledge sbti science based target emissions renewable energy commitment",
    ],
    "S": [
        # severe: rights violations, deaths, forced/child labor
        "human rights incident forced labor child labor death injury violation abuse",
        # moderate: labor disputes and strikes
        "labor dispute strike union wage complaint workplace safety",
        # routine: governance-adjacent disclosure
        "female board diversity composition directors representation disclosure",
    ],
    "G": [
        # severe: verified legal/regulatory exposure with a material outcome
        # (settlement/fine amount, class action, death) -- deliberately
        # excludes generic legal-process words ("10-K", "item", "proceedings")
        # that also appear in routine board/litigation-disclosure boilerplate.
        "million settlement class action fraud fine penalty deceptive alleging death",
        # routine: certification, policy programs, AND board-composition
        # disclosure. Board-independence vocabulary ("independent director
        # nominee committee") was found to have ZERO TF-IDF similarity to any
        # of these 3 seeds (it shares no words with fraud/compliance/report
        # language) -- with nothing to attract it, ~59 board claims landed on
        # the SEVERE seed by floating-point tiebreak alone and diluted that
        # cluster's label to "moderate". Folding board-independence terms into
        # this ROUTINE seed instead (not the severe one) gives k-means a
        # correctly-valenced home for them, so the severe cluster stays pure.
        "compliance certification anti corruption policy whistleblower program iso "
        "independent director nominee committee board composition",
        # routine: voluntary disclosure boilerplate
        "sustainability report published esg report annual disclosure",
    ],
}


def _claim_text(c: dict) -> str:
    """The text a claim is clustered/matched on: extractor reasoning. Factor
    key is deliberately NOT included, so clusters are driven by evidence
    content, not by tag identity.

    Was "reasoning + source_note" -- found 2026-09-18 that source_note has
    never existed anywhere: not on ExtractedClaim (shared/claim_types.py),
    not in the tune corpus this was fit on (calibration/abl_final_tune400.json,
    checked all 610 claims, key absent on every one). Both build() (below,
    reading from the corpus dict) and apply_cluster_severity() (reading
    ExtractedClaim attributes) always silently fell back to "" for it via
    .get()/getattr() defaults -- NOT a train/production mismatch (both sides
    only ever saw reasoning), just a reference to a field that was never
    wired up on either end. Removed rather than added, since inventing a
    source_note value now would change what the 9 already-fit-and-labeled
    clusters were actually trained on."""
    return str(c.get("reasoning") or "").strip()


# ── Offline: build ───────────────────────────────────────────────────────────

def build() -> None:
    from sklearn.cluster import KMeans
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import normalize

    rows = json.loads(_TUNE_CORPUS.read_text(encoding="utf-8"))
    by_pillar: dict[str, list[dict]] = {"E": [], "S": [], "G": []}
    for row in rows:
        for c in row.get("kept_claims") or []:
            if c.get("pillar") in by_pillar and len(_claim_text(c)) >= 30:
                by_pillar[c["pillar"]].append(c)

    model: dict = {}
    meta: dict = {}
    for pillar, claims in by_pillar.items():
        texts = [_claim_text(c) for c in claims]
        if len(texts) < 20:
            log.warning("[%s] only %d claims -- skipping pillar", pillar, len(texts))
            continue
        vec = TfidfVectorizer(
            stop_words="english", ngram_range=(1, 2), min_df=2,
            sublinear_tf=True, max_features=5000,
        )
        X = normalize(vec.fit_transform(texts))  # unit rows -> dot = cosine

        lo, hi = _K_RANGE[pillar]
        k = lo  # fixed at 3 (lo == hi == 3); see _K_RANGE comment
        seeds = _SEED_KEYWORDS.get(pillar)
        if seeds and len(seeds) == k:
            # Seed vectors go through the SAME fitted vectorizer as the
            # claims (transform, not fit_transform) -- terms not seen in the
            # claim corpus are simply absent, exactly like an out-of-vocabulary
            # word in a real claim.
            seed_X = normalize(vec.transform(seeds)).toarray()
            km = KMeans(n_clusters=k, init=seed_X, n_init=1, random_state=42)
        else:
            log.warning("[%s] no seed keywords for k=%d -- falling back to k-means++", pillar, k)
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        score = silhouette_score(X, labels, metric="cosine") if k > 1 else -1
        log.info("[%s] k=%d (seeded) silhouette=%.3f", pillar, k, score)
        centroids = normalize(km.cluster_centers_)
        model[pillar] = {"vectorizer": vec, "centroids": centroids}

        clusters_meta = {}
        import numpy as np
        sims = X @ centroids.T
        for cid in range(k):
            idx = [i for i, l in enumerate(labels) if l == cid]
            # representatives: members closest to the centroid
            reps = sorted(idx, key=lambda i: -sims[i, cid])[:4]
            terms = np.asarray(vec.get_feature_names_out())
            top_terms = terms[np.argsort(-km.cluster_centers_[cid])[:8]].tolist()
            from collections import Counter
            clusters_meta[str(cid)] = {
                "size": len(idx),
                "top_terms": top_terms,
                "factors": dict(Counter(claims[i]["factor"] for i in idx).most_common(5)),
                "polarity_mix": dict(Counter(int(claims[i].get("polarity", 0)) for i in idx)),
                "representatives": [_claim_text(claims[i])[:200] for i in reps],
                "severity": None,       # filled by `label`
                "label_rationale": None,
            }
        meta[pillar] = {"k": k, "silhouette": round(float(score), 3),
                        "n_claims": len(texts), "clusters": clusters_meta}
        log.info("[%s] fitted k=%d over %d claims (silhouette %.3f)", pillar, k, len(texts), score)

    MODEL_PATH.write_bytes(pickle.dumps(model))
    LABELS_PATH.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {MODEL_PATH.name} + {LABELS_PATH.name}")


# ── Offline: label ───────────────────────────────────────────────────────────

_LABEL_PROMPT = """You are calibrating an ESG evidence-scoring system. Below are clusters of similar evidence claims for the {pillar} pillar. For EACH cluster, assign an IMPACT MAGNITUDE — how much this KIND of evidence should move a company's 0-100 pillar score when present, regardless of direction (polarity is handled separately).

Allowed values ONLY: 0.25 (routine/weak — boilerplate disclosures, unverified mentions), 0.5 (moderate — concrete but common events), 0.75 (strong — regulator involvement, material commitments/violations), 1.0 (severe — major incidents, litigation with damages, deaths/injuries, large fines).

Clusters:
{cluster_block}

Respond with ONLY a JSON object mapping cluster id to {{"severity": <0.25|0.5|0.75|1.0>, "rationale": "<one short sentence>"}}:
{{"0": {{"severity": 0.5, "rationale": "..."}}, ...}}"""


def label() -> None:
    from zen_client import call_with_prompt
    from agentic_estimation.shared.llm_json import extract_json_object

    meta = json.loads(LABELS_PATH.read_text(encoding="utf-8"))
    for pillar, pdata in meta.items():
        block_lines = []
        for cid, c in pdata["clusters"].items():
            block_lines.append(
                f"Cluster {cid} (n={c['size']}, factors={list(c['factors'])[:3]}, "
                f"top terms: {', '.join(c['top_terms'][:6])}):\n"
                + "\n".join(f"  - {r}" for r in c["representatives"][:3])
            )
        prompt = _LABEL_PROMPT.format(pillar=pillar, cluster_block="\n\n".join(block_lines))
        resp = call_with_prompt(prompt, max_tokens=2000, timeout=180)
        if not resp.get("ok"):
            log.error("[%s] label call failed: %s", pillar, resp.get("error"))
            continue
        parsed = extract_json_object(resp.get("raw", "")) or extract_json_object(resp.get("reasoning", ""))
        if not parsed:
            log.error("[%s] could not parse label JSON", pillar)
            continue
        for cid, entry in parsed.items():
            if cid in pdata["clusters"] and isinstance(entry, dict):
                sev = entry.get("severity")
                try:
                    sev = float(sev)
                except (TypeError, ValueError):
                    continue
                # snap to nearest allowed level
                sev = min(_VALID_SEVERITIES, key=lambda v: abs(v - sev))
                pdata["clusters"][cid]["severity"] = sev
                pdata["clusters"][cid]["label_rationale"] = str(entry.get("rationale", ""))[:200]
        n_labeled = sum(1 for c in pdata["clusters"].values() if c["severity"] is not None)
        log.info("[%s] labeled %d/%d clusters", pillar, n_labeled, len(pdata["clusters"]))

    LABELS_PATH.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"updated {LABELS_PATH.name}")


# ── Runtime inference ────────────────────────────────────────────────────────

_runtime_cache: Optional[dict] = None


def _load_runtime() -> Optional[dict]:
    """{pillar: {vectorizer, centroids, severities: {cid: float}}} or None."""
    global _runtime_cache
    if _runtime_cache is not None:
        return _runtime_cache or None
    if not (MODEL_PATH.exists() and LABELS_PATH.exists()):
        log.warning("cluster artifacts missing -- cluster severity disabled (no-op)")
        _runtime_cache = {}
        return None
    try:
        model = pickle.loads(MODEL_PATH.read_bytes())
        meta = json.loads(LABELS_PATH.read_text(encoding="utf-8"))
        runtime = {}
        for pillar, m in model.items():
            sev = {
                int(cid): c["severity"]
                for cid, c in meta.get(pillar, {}).get("clusters", {}).items()
                if c.get("severity") is not None
            }
            if sev:
                runtime[pillar] = {**m, "severities": sev}
        _runtime_cache = runtime
        return runtime or None
    except Exception as exc:
        log.error("failed to load cluster artifacts (%s) -- disabled", exc)
        _runtime_cache = {}
        return None


def apply_cluster_severity(claims: list) -> list[dict]:
    """Adjust claims IN PLACE: strength <- nearest labeled cluster's severity,
    confidence *= membership-scaled multiplier in [_CONF_FLOOR, 1.0].
    Claims below MIN_MEMBERSHIP (or in pillars without artifacts) are left
    untouched. Returns a list of adjustment dicts for logging/explainability.
    Accepts ExtractedClaim objects (attribute access) — the claim's text is
    rebuilt exactly like training's _claim_text (reasoning only -- see that
    function's docstring for why source_note was removed 2026-09-18)."""
    runtime = _load_runtime()
    if not runtime:
        return []
    from sklearn.preprocessing import normalize

    adjustments = []
    for claim in claims:
        pr = runtime.get(getattr(claim, "pillar", None))
        if pr is None:
            continue
        text = (getattr(claim, "reasoning", "") or "").strip()
        if len(text) < 30:
            continue
        X = normalize(pr["vectorizer"].transform([text]))
        sims = (X @ pr["centroids"].T).toarray()[0] if hasattr(X @ pr["centroids"].T, "toarray") \
            else (X @ pr["centroids"].T)[0]
        cid = int(sims.argmax())
        sim = float(sims[cid])
        if sim < MIN_MEMBERSHIP or cid not in pr["severities"]:
            continue
        severity = pr["severities"][cid]
        membership = min(1.0, sim / _SIM_SATURATION)
        conf_mult = _CONF_FLOOR + (1.0 - _CONF_FLOOR) * membership
        old_strength, old_conf = claim.strength, claim.confidence
        claim.strength = severity
        claim.confidence = round(claim.confidence * conf_mult, 3)
        adjustments.append({
            "factor": claim.factor, "pillar": claim.pillar, "cluster": cid,
            "similarity": round(sim, 3),
            "strength": (round(old_strength, 3), severity),
            "confidence": (round(old_conf, 3), claim.confidence),
        })
    if adjustments:
        log.info("cluster severity adjusted %d/%d claims", len(adjustments), len(claims))
    return adjustments


# ── CLI ──────────────────────────────────────────────────────────────────────

def _show() -> None:
    meta = json.loads(LABELS_PATH.read_text(encoding="utf-8"))
    for pillar, pdata in meta.items():
        print(f"\n=== {pillar}: k={pdata['k']} silhouette={pdata['silhouette']} n={pdata['n_claims']} ===")
        for cid, c in pdata["clusters"].items():
            sev = c["severity"] if c["severity"] is not None else "?"
            print(f"  [{cid}] n={c['size']:<4} sev={sev:<5} factors={list(c['factors'])[:3]}")
            print(f"       terms: {', '.join(c['top_terms'][:6])}")
            if c.get("label_rationale"):
                print(f"       why: {c['label_rationale']}")


def _replay(dump_path: str) -> None:
    """Run every frozen claim in an ablation dump through the clusterer and
    print what it matched, so cluster assignment can be eyeballed. Holdout
    dumps are the honest check (clusters were fitted on tune only)."""
    from collections import Counter
    from sklearn.preprocessing import normalize

    runtime = _load_runtime()
    if not runtime:
        print("no artifacts -- run build + label first")
        sys.exit(1)
    meta = json.loads(LABELS_PATH.read_text(encoding="utf-8"))

    rows = json.loads(Path(dump_path).read_text(encoding="utf-8"))
    matched, unmatched, per_cluster = 0, 0, Counter()
    strength_moves = Counter()
    print(f"replaying frozen claims from {dump_path} ({len(rows)} companies)\n")
    for row in rows:
        for c in row.get("kept_claims") or []:
            pillar = c.get("pillar")
            pr = runtime.get(pillar)
            text = _claim_text(c)
            if pr is None or len(text) < 30:
                continue
            X = normalize(pr["vectorizer"].transform([text]))
            sims_m = X @ pr["centroids"].T
            sims = sims_m.toarray()[0] if hasattr(sims_m, "toarray") else sims_m[0]
            cid = int(sims.argmax())
            sim = float(sims[cid])
            if sim < MIN_MEMBERSHIP or cid not in pr["severities"]:
                unmatched += 1
                continue
            matched += 1
            sev = pr["severities"][cid]
            key = f"{pillar}[{cid}]"
            per_cluster[key] += 1
            old = c.get("strength", 0.5)
            direction = "down" if sev < old else ("up" if sev > old else "same")
            strength_moves[direction] += 1
            if per_cluster[key] <= 2:   # print first 2 examples per cluster
                terms = ", ".join(meta[pillar]["clusters"][str(cid)]["top_terms"][:4])
                print(f"  [{row['name'][:24]:24s}] {c.get('factor','?'):26s} -> {key} "
                      f"(sim={sim:.2f}, sev={sev}, was strength={old})")
                print(f"      cluster terms: {terms}")
                print(f"      claim text:    {text[:110]}")
    total = matched + unmatched
    print(f"\nmatched {matched}/{total} claims "
          f"({unmatched} out-of-vocabulary, left untouched)")
    print(f"strength moves: {dict(strength_moves)}")
    print("\nper-cluster hit counts:")
    for key, n in per_cluster.most_common():
        pillar, cid = key[0], key[2:-1]
        cmeta = meta[pillar]["clusters"][cid]
        print(f"  {key:8s} n={n:<4} sev={cmeta['severity']}  {', '.join(cmeta['top_terms'][:5])}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "build":
        build()
    elif cmd == "label":
        label()
    elif cmd == "show":
        _show()
    elif cmd == "replay":
        _replay(sys.argv[2] if len(sys.argv) > 2 else "calibration/abl_final_holdout68.json")
    else:
        print("usage: python -m agentic_estimation.layer_2.evidence_clusters build|label|show|replay [dump.json]")
        sys.exit(1)
