"""
evidence_classifier.py -- deterministic evidence tagger (factor / polarity / strength).

WHY THIS EXISTS
    The extraction LLM is not reproducible. Measured 2026-08-08 against the
    free-tier endpoint with temperature=0 in the payload: the same prompt
    returned three different answers with three different md5s, and one company
    with byte-identical frozen evidence produced 17 / 10 / 3 claims on three
    consecutive runs. The endpoint accepts the temperature parameter and
    ignores it. A scoring pipeline whose evidence layer changes run-to-run
    cannot be audited, and every A/B we ran on claim counts was measuring
    sampling noise.

    This module replaces the deterministic part of that job. Same input always
    yields the same output, because the weights are numbers in a file.

WHAT IT DOES NOT REPLACE
    Genuine natural-language judgement over novel phrasing. This model
    recognises patterns resembling its training data; an LLM generalises
    better. The trade is reliability for flexibility, taken deliberately
    because the failures we measured were reliability failures.

THREE HEADS, NOT ONE
    factor   : 30 classes -- 28 registry factors + wrong_entity + no_evidence
    polarity : is the factor asserted or DENIED
    strength : how decisive the wording is, 0-1

    Separate heads because the production defect was correct-factor +
    wrong-polarity: the LLM emitted net_zero_pledge polarity=+1 while its own
    reasoning read "does not explicitly mention a net-zero pledge". A single
    joint model blurs that distinction; a dedicated polarity head trains on
    "has committed" vs "has not announced" as the whole task.

REJECTION IS A CLASS, NOT A SEPARATE GATE
    57% of real evidence snippets are about a DIFFERENT company (hand-measured,
    calibration/labeled_evidence.py). A gate-then-classify design needs two
    thresholds and lets junk reach a classifier that will confidently label it.
    wrong_entity/no_evidence as first-class labels make "not about this
    company" a prediction with a probability attached.

FEATURES -- THREE BLOCKS, EACH COVERING THE OTHERS' BLIND SPOT
    embedding  : MiniLM, 384 dims. Handles paraphrase and cross-language
                 ("klimaneutral bis 2030" ~ "carbon neutral by 2030"). Blurs
                 exact tokens: "Scope 1" and "Scope 3" land almost on top of
                 each other despite being different 8-weight factors.
    tf-idf     : char n-grams. Keeps exactly what the embedding blurs --
                 "Scope 1" vs "Scope 3", "ISO 14001" vs "ISO 45001". No concept
                 of meaning.
    entity     : does the company name actually appear in the text. Neither
                 representation above reliably encodes this, and it is the
                 single most predictive signal for the 57% failure mode.

    More features than rows (2400 vs 632) risks memorisation, so: L2
    regularisation, TF-IDF capped by document frequency, and blocks scaled so
    one cannot numerically dominate.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

MODEL_PATH = Path("calibration/evidence_clf.joblib")
TRAINSET = Path("calibration/trainset_evidence.jsonl")
_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Below this the model says nothing rather than guessing. With 57% of real
# input being junk, abstaining is frequently the correct output -- a confident
# wrong label is worse than no label, because it enters scoring as evidence.
DEFAULT_MIN_PROB = 0.45
DEFAULT_MIN_MARGIN = 0.10          # gap to the runner-up

# Relaxed bars applied ONLY when the entity head is confident the snippet really
# belongs to the company it is filed under. See predict() for the reasoning:
# the expensive failure is another company's data entering a score, and that is
# already ruled out by the time these apply.
# MEASURED AND REJECTED: relaxing these below the strict bar changed nothing on
# the held-out sets (the misses were confident no_evidence predictions at
# 0.68-0.86, not threshold rejections), and once combined with other changes it
# let through leaks at 0.33-0.40 that the strict bar had caught. Left equal to
# the strict values: the split costs nothing and stays available if a future
# model is genuinely threshold-limited.
DEFAULT_MIN_PROB_ATTRIBUTED = DEFAULT_MIN_PROB
DEFAULT_MIN_MARGIN_ATTRIBUTED = DEFAULT_MIN_MARGIN
DEFAULT_ENTITY_CONF_FLOOR = 0.60   # P(right company) needed to relax

_embedder = None


def _get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer(_EMBED_MODEL)
    return _embedder


_STOP = {"ltd", "limited", "inc", "plc", "sa", "srl", "gmbh", "ag", "nv", "bv",
         "co", "corp", "corporation", "group", "holdings", "the", "and", "sarl",
         "spa", "as", "asa", "kk", "pty", "sb", "llc"}


def _name_tokens(company: str) -> list[str]:
    toks = re.findall(r"[a-z0-9]+", (company or "").lower())
    return [t for t in toks if t not in _STOP and len(t) > 2]


def entity_features(text: str, company: Optional[str]) -> list[float]:
    """Does this text actually mention the company it is filed under?

    Two numbers rather than one boolean: full-name presence is decisive when
    true, but many legitimate snippets use a trade name, so token coverage
    carries the partial-credit signal.
    """
    if not company:
        return [0.0, 0.0]
    low = (text or "").lower()
    toks = _name_tokens(company)
    if not toks:
        return [0.0, 0.0]
    full = 1.0 if " ".join(toks) in low else 0.0
    cover = sum(1 for t in toks if t in low) / len(toks)
    return [full, cover]


# The collector prefixes every snippet with the SOURCE SLOT it was filed
# under -- "ESG Controversies: ", "Board Composition: ", "Facility Footprint: "
# and so on. That is metadata about which query produced the text, not part of
# the evidence, and it is frequently WRONG about the content: measured on the
# held-out 100, a company's own disclosure report arrived prefixed "Board
# Composition:", and a Spanish emissions-offsetting claim arrived prefixed
# "Facility Footprint:".
#
# Left in place the prefix dominates classification -- the model reads
# "ESG Controversies:" and predicts a controversy for an impact-report
# announcement. Stripped, the model sees the evidence itself.
_SOURCE_PREFIX = re.compile(
    r"^\s*(?:ESG Controversies|Board Composition|Facility Footprint|"
    r"Compliance Certifications?|Litigation Records|Regulatory Fines/Violations|"
    r"Net Zero Commitment|CDP Climate Disclosure|Sustainability Report|"
    r"GRI Database|SBTi|B&HR Resource Centre|Country Governance Context|"
    r"SEC 10-K Item \d+[^:]*|SEC DEF 14A Proxy Statement[^:]*)\s*:\s*",
    re.I)


# Collector snippets interleave prose with the URL each fragment came from:
#   "Silk Grass Farms Ltd Disclosure Report 2025. <https://www.bcorporation.
#    net/en-us/find-a-b-corp/company/silk-grass-farms> ..."
# URLs are long, and char n-grams over them swamp the few n-grams carrying the
# actual claim. Measured on the held-out 100: three snippets scored
# `no_evidence` at 0.68-0.86 with URLs present and were classified correctly at
# 0.61-1.00 with the same URLs removed. The URL is provenance, kept elsewhere;
# it is not evidence about the company.
#
# KEEP THE DOMAIN, DROP THE PATH. Removing URLs entirely raised factor recall
# (3/6 -> 4/6) but cost junk rejection (86/86 -> 81/86): five GRI/news pages
# were suddenly read as company disclosures, because the host was the only
# thing marking them third-party. globalreporting.org and esgtoday.com are
# strong "not this company's own disclosure" signals; the 90-character path
# after them is the noise.
def clean_snippet(text: str) -> str:
    """Strip the collector's source-slot prefix. URLs are left ALONE.

    Three URL treatments were measured on the held-out 100 (junk rejected /
    factors found, out of 86 and 6):

        prefix only, URLs kept    86 / 3   <- best total, shipped
        URLs removed entirely     81 / 4
        URLs collapsed to host    83 / 3

    Removing URLs lifts factor recall -- long paths swamp the char n-grams --
    but costs more in junk rejection than it gains: five third-party pages
    (Rio Tinto news on esgtoday.com, GRI framework documents) were read as
    company disclosures once the host was gone. Collapsing to the bare host was
    the intuitive compromise and tested WORSE than leaving URLs untouched;
    domain tokens appear to confuse the factor head more than they help the
    entity head.

    Kept as a single documented function so the next person re-testing this
    starts from the measurements rather than the intuition.
    """
    return _SOURCE_PREFIX.sub("", text or "", count=1).strip()


# Back-compat name; callers outside this module use clean_snippet().
def strip_source_prefix(text: str) -> str:
    return clean_snippet(text)


@dataclass
class Prediction:
    factor: Optional[str]          # None when abstaining
    label: str                     # raw argmax, even when abstaining
    probability: float
    margin: float
    polarity: Optional[int]
    strength: Optional[float]
    abstained: bool
    runner_up: Optional[str] = None


class EvidenceClassifier:
    def __init__(self, min_prob: float = DEFAULT_MIN_PROB,
                 min_margin: float = DEFAULT_MIN_MARGIN,
                 min_prob_attributed: float = DEFAULT_MIN_PROB_ATTRIBUTED,
                 min_margin_attributed: float = DEFAULT_MIN_MARGIN_ATTRIBUTED,
                 entity_conf_floor: float = DEFAULT_ENTITY_CONF_FLOOR):
        self.min_prob = min_prob
        self.min_margin = min_margin
        self.min_prob_attributed = min_prob_attributed
        self.min_margin_attributed = min_margin_attributed
        self.entity_conf_floor = entity_conf_floor
        self.factor_clf = None
        self.entity_clf = None
        self.polarity_clf = None
        self.strength_reg = None
        self.tfidf = None
        self.scaler_txt = None      # text-only block (factor/polarity/strength)
        self.scaler_ent = None      # text + entity block (entity head)

    # ── feature assembly ────────────────────────────────────────────────────
    def _features(self, texts: list[str], companies: list[Optional[str]],
                  fit: bool = False, with_entity: bool = True) -> np.ndarray:
        """Feature matrix. `with_entity` controls the entity block.

        THE FACTOR HEAD MUST NOT SEE ENTITY FEATURES. "What is this text
        about?" is a property of the text alone; whether the filed company is
        named in it is a separate question. Measured on the held-out 100:
        DJS Research's own impact report was classified `regulatory_fines`
        with its real name attached and `esg_report_published` -- correct --
        with the name swapped out. The entity block was leaking into topic
        prediction and corrupting it.

        Entity information is not discarded; it moves to its own head (see
        predict()), which is the only place it is decision-relevant.
        """
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.preprocessing import StandardScaler

        texts = [clean_snippet(t) for t in texts]

        emb = _get_embedder().encode(texts, normalize_embeddings=True,
                                     show_progress_bar=False, batch_size=64)
        emb = np.asarray(emb)

        # fit() calls this twice (text-only, then with entity). The vectoriser
        # must be fitted ONCE or the second call silently rebuilds a different
        # vocabulary and the two scalers end up describing different feature
        # spaces.
        if fit and self.tfidf is None:
            # char_wb n-grams: robust to the tokenisation noise in PDF/HTML
            # text, and they keep "scope 1"/"scope 3" separable.
            self.tfidf = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                                         min_df=2, max_features=2000,
                                         sublinear_tf=True)
            tf = self.tfidf.fit_transform(texts).toarray()
        else:
            tf = self.tfidf.transform(texts).toarray()

        blocks = [emb, tf]
        if with_entity:
            blocks.append(np.array([entity_features(t, c)
                                    for t, c in zip(texts, companies)]))
        X = np.hstack(blocks)

        key = "scaler_ent" if with_entity else "scaler_txt"
        if fit:
            sc = StandardScaler(with_mean=True)
            X = sc.fit_transform(X)
            setattr(self, key, sc)
        else:
            X = getattr(self, key).transform(X)
        return X

    # ── training ────────────────────────────────────────────────────────────
    def fit(self, rows: list[dict]) -> dict:
        from sklearn.linear_model import LogisticRegression, Ridge

        texts = [r["text"] for r in rows]
        comps = [r.get("company") for r in rows]
        y_factor = [r["label"] for r in rows]

        # TEXT-ONLY features for the factor head -- see _features().
        Xt = self._features(texts, comps, fit=True, with_entity=False)
        # WITH entity features for the entity head.
        Xe = self._features(texts, comps, fit=True, with_entity=True)

        # C=1.0 with L2: the feature count exceeds the row count, so the
        # penalty is doing real work here, not ceremony.
        self.factor_clf = LogisticRegression(max_iter=3000, C=1.0,
                                             class_weight="balanced")
        self.factor_clf.fit(Xt, y_factor)

        # Entity head: "is this text about the company it is filed under?"
        # Binary, and the only place entity features are decision-relevant.
        # wrong_entity is the positive class; everything else -- including
        # no_evidence -- is "right company, whatever else is true".
        self.entity_clf = LogisticRegression(max_iter=2000, C=1.0,
                                             class_weight="balanced")
        self.entity_clf.fit(Xe, [int(l == "wrong_entity") for l in y_factor])

        X = Xt   # polarity/strength read topic, not entity

        # Polarity and strength train ONLY on rows that carry them -- the
        # rejection classes have no polarity, and including them as a third
        # value would teach the head that junk has a sign.
        pol_idx = [i for i, r in enumerate(rows) if r.get("polarity") is not None]
        if pol_idx:
            self.polarity_clf = LogisticRegression(max_iter=2000, C=1.0,
                                                   class_weight="balanced")
            self.polarity_clf.fit(X[pol_idx],
                                  [rows[i]["polarity"] for i in pol_idx])

        st_idx = [i for i, r in enumerate(rows) if r.get("strength") is not None]
        if st_idx:
            self.strength_reg = Ridge(alpha=1.0)
            self.strength_reg.fit(X[st_idx],
                                  [rows[i]["strength"] for i in st_idx])

        return {"rows": len(rows), "classes": len(set(y_factor)),
                "features": X.shape[1], "polarity_rows": len(pol_idx),
                "strength_rows": len(st_idx)}

    # ── inference ───────────────────────────────────────────────────────────
    def predict(self, text: str, company: Optional[str] = None) -> Prediction:
        """Two independent questions, answered separately then combined.

            factor head : what is this text about?      (text only)
            entity head : is it about THIS company?     (text + entity)

        A claim survives only if the topic is a real factor AND the entity
        check passes. Keeping them apart means a perfectly-written Rio Tinto
        disclosure filed under a company called "Mine" is rejected for the
        right reason -- wrong company -- rather than by corrupting the topic
        prediction, which is what a single joint model did.
        """
        Xt = self._features([text], [company], fit=False, with_entity=False)
        proba = self.factor_clf.predict_proba(Xt)[0]
        order = np.argsort(proba)[::-1]
        classes = self.factor_clf.classes_
        top, second = classes[order[0]], classes[order[1]]
        p_top, p_second = proba[order[0]], proba[order[1]]
        margin = float(p_top - p_second)

        wrong_entity = False
        entity_conf = 0.0
        if self.entity_clf is not None and company:
            Xe = self._features([text], [company], fit=False, with_entity=True)
            # P(not wrong_entity) -- how sure we are this IS the right company.
            entity_conf = float(self.entity_clf.predict_proba(Xe)[0][0])
            wrong_entity = bool(self.entity_clf.predict(Xe)[0])

        # TWO THRESHOLDS, NOT ONE. Rejection and detection are different
        # decisions and deserve different bars.
        #
        # The single 0.45 bar was costing real evidence: "Silk Grass Farms Ltd
        # Disclosure Report 2025" scored esg_report_published at 0.38 and was
        # discarded -- despite the entity head being confident it really was
        # that company's own page.
        #
        # Once the entity head has cleared a snippet, most of the risk is
        # already gone: the failure mode this pipeline actually suffers is
        # ANOTHER COMPANY'S data entering a score (57% of collected evidence),
        # not a mild topic confusion within the right company's own text. So a
        # confidently-attributed snippet gets the relaxed bar, and anything the
        # entity head is unsure about keeps the strict one.
        confident_entity = entity_conf >= self.entity_conf_floor
        min_prob = (self.min_prob_attributed if confident_entity
                    else self.min_prob)
        min_margin = (self.min_margin_attributed if confident_entity
                      else self.min_margin)

        abstain = (p_top < min_prob) or (margin < min_margin)
        is_reject = top in ("wrong_entity", "no_evidence") or wrong_entity

        pol = stg = None
        if not abstain and not is_reject:
            if self.polarity_clf is not None:
                pol = int(self.polarity_clf.predict(Xt)[0])
            if self.strength_reg is not None:
                stg = float(np.clip(self.strength_reg.predict(Xt)[0], 0.05, 0.95))

        label = "wrong_entity" if (wrong_entity and top not in
                                   ("wrong_entity", "no_evidence")) else str(top)
        return Prediction(
            factor=(None if (abstain or is_reject) else str(top)),
            label=label, probability=float(p_top), margin=margin,
            polarity=pol, strength=(round(stg, 3) if stg is not None else None),
            abstained=bool(abstain), runner_up=str(second))

    # ── persistence ─────────────────────────────────────────────────────────
    def save(self, path: Path = MODEL_PATH) -> None:
        import joblib
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"factor": self.factor_clf, "entity": self.entity_clf,
                     "polarity": self.polarity_clf,
                     "strength": self.strength_reg, "tfidf": self.tfidf,
                     "scaler_txt": self.scaler_txt,
                     "scaler_ent": self.scaler_ent,
                     "min_prob": self.min_prob,
                     "min_margin": self.min_margin,
                     "min_prob_attributed": self.min_prob_attributed,
                     "min_margin_attributed": self.min_margin_attributed,
                     "entity_conf_floor": self.entity_conf_floor}, path)

    @classmethod
    def load(cls, path: Path = MODEL_PATH) -> "EvidenceClassifier":
        import joblib
        d = joblib.load(path)
        c = cls(min_prob=d.get("min_prob", DEFAULT_MIN_PROB),
                min_margin=d.get("min_margin", DEFAULT_MIN_MARGIN),
                min_prob_attributed=d.get("min_prob_attributed",
                                          DEFAULT_MIN_PROB_ATTRIBUTED),
                min_margin_attributed=d.get("min_margin_attributed",
                                            DEFAULT_MIN_MARGIN_ATTRIBUTED),
                entity_conf_floor=d.get("entity_conf_floor",
                                        DEFAULT_ENTITY_CONF_FLOOR))
        c.factor_clf, c.polarity_clf = d["factor"], d["polarity"]
        c.entity_clf = d.get("entity")
        c.strength_reg, c.tfidf = d["strength"], d["tfidf"]
        c.scaler_txt, c.scaler_ent = d["scaler_txt"], d["scaler_ent"]
        return c


def load_rows(path: Path = TRAINSET) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
