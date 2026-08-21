"""
report_structure.py -- structured extraction of an ESG report into two channels.

WHY TWO CHANNELS
    Tables and prose have opposite retrieval characteristics. A table row is
    mostly numbers, and its embedding sits nowhere near a concept query like
    "greenhouse gas emissions". Measured on Toyota Industries' 2021 report: the
    CSR materiality scorecard -- the single densest evidence block in the
    document -- ranked 19th of 268 blocks for E under similarity selection and
    was never retrieved, while 174k chars of E-relevant content existed and the
    extractor received 3,422 (2%).

    So tables never enter similarity ranking at all:

        table channel : ODL JSON -> inventory -> ESG-term filter (deterministic)
        prose channel : text -> chunks -> embeddings (Phase 4/5, separate module)

    A scorecard cannot be buried by a ranking it never joins.

WHAT ODL'S JSON ACTUALLY GIVES US (verified 2026-08-07, opendataloader_pdf)
    Top level : {"file name", "number of pages", "kids": [...]}
    Nodes     : type in {heading, paragraph, table, list, list item, image}
    Every node carries "page number".
    Tables carry "number of rows"/"number of columns" and a "rows" list --
    NOT "children". Each row: {"row number", "cells":[...]}, each cell:
    {"row number","column number","row span","column span","kids":[...]} where
    kids are paragraph nodes holding the actual "content" string.

    Measured: Toyota page 7 yields 5 table nodes (7x9, 4x9, 4x9, 4x9, 7x9) --
    that is the CSR scorecard, split by ODL into sub-tables.

PAGE PROVENANCE
    This is the module that makes claim["page"] honest. The markdown backend in
    report_parser.py flattens pages away; every node here keeps its page.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

from agentic_estimation.shared.pipeline_logger import get_logger

log = get_logger("report_structure")

# ── ESG vocabulary, per pillar ───────────────────────────────────────────────
# Used for the DETERMINISTIC table filter and for the sparse half of hybrid
# prose retrieval. Deliberately concrete: framework names, metric names and
# certification codes are exact tokens where embeddings blur precisely the
# distinctions that matter ("Scope 1" vs "Scope 3").
# Expanded 2026-08-07 from vocabulary observed in the corpus itself rather than
# invented: table headings and row labels harvested across the accepted report
# set (see calibration/_table_vocab.json). Misses that drove specific additions:
#   * Toyota p37 prints "CO2 emissions" and "Greenhouse gases other than CO2"
#     under a heading "Into the Air" -- no "scope" wording anywhere.
#   * Fanuc's G tables use "Frequency of Board of Directors" / "Outside
#     Directors", matching no alias in the first draft (measured: 0 G metrics).
#   * Japanese/Nordic reports favour "occupational accident", "frequency rate",
#     "return-to-work ratio", "paid leave-taking rate".
# Terms are substrings, so stems ("recycl", "certif") cover their inflections.
PILLAR_TERMS: dict[str, tuple[str, ...]] = {
    "E": (
        # emissions & climate
        "emission", "emissions", "ghg", "greenhouse", "carbon", "co2", "co2e",
        "t-co2", "tco2", "scope 1", "scope 2", "scope 3", "climate", "tcfd",
        "net zero", "net-zero", "carbon neutral", "sbti", "science based",
        "decarbon", "offset", "sequestration", "into the air", "air emission",
        "nox", "sox", "sulfur oxide", "nitrogen oxide", "particulate", "vocs",
        "ozone", "refrigerant", "methane", "flaring",
        # energy
        "energy", "electricity", "renewable", "solar", "wind power", "biomass",
        "geothermal", "hydro", "fuel", "petroleum", "diesel", "gasoline",
        "city gas", "lpg", "lng", "coal", "kwh", "mwh", "gwh", "gj", "tj",
        "energy intensity", "power consumption",
        # water & effluent
        "water", "wastewater", "effluent", "water withdrawal", "water discharge",
        "water intensity", "water stress", "potable", "groundwater", "cooling water",
        # waste & materials
        "waste", "recycl", "landfill", "incinerat", "hazardous", "non-hazardous",
        "circular", "reuse", "raw material", "packaging", "plastic", "sludge",
        "by-product", "scrap", "prtr", "chemical substance",
        # nature & systems
        "biodiversity", "deforestation", "ecosystem", "habitat", "land use",
        "environmental", "iso 14001", "environmental management",
        "environmental compliance", "spill", "pollution", "life cycle",
        "intensity", "footprint", "cdp",
        # REMOVED after substring testing (2026-08-07):
        #   "ems" -> matches systems/problems/items, fires on almost any table
        #   "lca" -> matches "local content"; "life cycle" already covers it
        #   "gj"  -> matches Norwegian "regjeringen"; TJ/kWh/MWh cover energy
        #   "hydro" -> matches hydrocarbon/hydrogen, neither of which is
        #              renewable-energy evidence. "hydropower" is explicit.
        "hydropower",
    ),
    "S": (
        # workforce
        # "staff" alone matches "staff costs" (an income-statement line); the
        # workforce senses are kept explicitly. "associate" also matches the
        # accounting sense ("investment in associates") -- Toyota uses it for
        # employees, so it stays, but only via the metric aliases, not here.
        "employee", "workforce", "headcount", "staff number", "staff turnover",
        "personnel", "new hire", "recruit", "turnover", "attrition", "retention",
        "absentee", "temporary", "contractor", "apprentice",
        # diversity
        "diversity", "inclusion", "gender", "women", "female", "male ratio",
        "disabilit", "ethnic", "age group", "nationality",
        "equal opportunity", "pay gap", "equal pay", "inclusive",
        # "minorit" removed: "minority interest" is a balance-sheet line and
        # was the dominant match. The workforce sense is covered by
        # "ethnic"/"diversity"/"underrepresented".
        "underrepresented",
        # development
        "training", "education", "learning", "development hours", "upskill",
        "career", "performance review", "engagement survey", "satisfaction",
        # health & safety
        "safety", "accident", "injury", "ltifr", "ltir", "trir", "fatalit",
        "occupational", "iso 45001", "ohsas", "near miss", "lost time",
        "frequency rate", "severity rate", "health and safety", "wellbeing",
        "well-being", "mental health", "health check", "medical",
        "zero fatalities", "incident rate", "work-related",
        # rights & labour
        "human rights", "labour", "labor", "child labour", "forced labour",
        # "union" alone matches "European Union" (a jurisdiction, in every
        # EU-domiciled report) and "communion". The labour sense needs the
        # qualifier.
        "modern slavery", "collective bargaining", "trade union",
        "labour union", "union membership", "sa8000",
        "freedom of association", "grievance", "wage", "living wage",
        "minimum wage", "overtime", "working hours", "parental leave",
        "maternity", "paternity", "return-to-work", "return to work",
        "paid leave", "leave-taking",
        # value chain & community
        "supplier", "supply chain", "procurement", "vendor", "sourcing",
        "community", "donation", "philanthrop", "volunteer", "csr activity",
        "social contribution", "customer satisfaction", "product safety",
        "recall", "data privacy", "personal information",
    ),
    "G": (
        # board
        # Harvested from governance tables across 12 corpus reports
        # (calibration/_gov_vocab.json). What those tables ACTUALLY contain:
        # meeting-attendance grids by director name, management-tier
        # breakdowns, and GRI/SASB disclosure indexes -- not stated
        # independence percentages. Terms below reflect that reality.
        "board", "director", "independent", "outside director",
        "non-executive", "chairman", "board composition", "board meeting",
        "attendance", "tenure", "board evaluation", "succession",
        "frequency of board", "board of directors", "board effectiveness",
        "supervisory board", "audit & supervisory", "audit and supervisory",
        "executive officer", "senior management", "executive management",
        "upper management", "middle management", "junior management",
        "management level", "management tier", "leadership team",
        # Disclosure indexes are the most common governance table in the
        # corpus (GRI content index: 7 of 12 reports). They evidence
        # esg_report_published and map where every other metric is disclosed.
        "gri content index", "sasb content index", "content index",
        "gri 2-", "gri 3-", "gri 205", "gri 405", "gri 403",
        "disclosure number", "disclosure title", "omission",
        # committees
        "committee", "audit", "nomination", "remuneration committee",
        "compensation committee", "risk committee", "sustainability committee",
        "esg committee", "advisory",
        # pay & ownership
        "remuneration", "compensation", "executive pay", "shareholder",
        "voting", "share capital", "ownership", "dividend", "stock option",
        "related party", "capital structure",
        # ethics & compliance
        "ethics", "ethical", "code of conduct", "compliance", "anti-corruption",
        "anti corruption", "bribery", "bribe", "corruption", "whistleblow",
        "speak up", "hotline", "conflict of interest", "antitrust",
        "money laundering", "sanctions screening", "penalty", "violation",
        "legal proceeding", "litigation",
        # REMOVED after substring testing (2026-08-07):
        #   "tax"  -> 37/37 hits in the corpus were financial ("pretax income",
        #             "deferred tax assets"). Tax TRANSPARENCY is the governance
        #             signal, so the specific phrase is kept below instead.
        #   "fine" -> matches "defined benefit obligation", "refined products".
        #             "penalty"/"violation" carry the same meaning unambiguously.
        #   "sanction" -> matches "sanctioned"/"sanctuary"; narrowed to the
        #             compliance phrase.
        "tax transparency", "tax strategy", "effective tax rate",
        # risk & reporting
        "risk management", "internal control", "internal audit", "assurance",
        "transparency", "disclosure", "gri", "sasb", "tcfd",
        "materiality", "sdg", "stakeholder", "cyber", "information security",
        "business continuity", "governance",
    ),
}

# Terms that mark a table as ESG-relevant regardless of pillar. A materiality
# or SDG scorecard is the highest-value table in most reports and its headers
# often name no single pillar.
_CROSS_PILLAR = ("materiality", "sdg", "sustainable development goal", "gri",
                 "esg", "sustainability", "non-financial", "scorecard")

# ── Core-metric aliases ──────────────────────────────────────────────────────
# The 12 SCORED entries of CORE_METRICS are what the formula estimator actually
# consumes, and 10 of them are numeric benchmark_band factors. Numeric ESG
# disclosure lives in data tables, not prose -- measured on Toyota's report,
# page 37: "Energy consumption 15,938 TJ", "Water withdrawal 4,548 km3", both
# direct reads of core metrics.
#
# Pillar topic words alone do not reliably surface those tables: an appendix
# headed "Energy [consolidated]" carries no word from the G list even though
# it holds board data two rows down, and LTIFR/board-independence tables are
# routinely labelled only by their acronym. So every scored metric gets its own
# alias set, and a table matching ANY alias is kept for that metric's pillar.
#
# This is also what lets the extractor be a MAPPER rather than an interpreter:
# it is handed a labelled row and a closed list of metric keys.
METRIC_ALIASES: dict[str, tuple[str, ...]] = {
    # ── E ────────────────────────────────────────────────────────────────
    # "co2 emissions" belongs to scope 1 because that is what an unqualified
    # CO2 figure in a company's own environmental data table means -- measured
    # on Toyota p37, which prints "CO2 emissions 824,841 t-CO2" and never uses
    # the word "scope" anywhere in the table. Extraction still records the
    # report's own label, so a mis-assignment stays auditable downstream.
    "scope_1_emissions":        ("scope 1", "scope1", "scope i",
                                 "direct emission", "direct ghg",
                                 "direct co2", "co2 emissions", "co2 emission",
                                 "greenhouse gas emissions", "ghg emissions",
                                 "carbon emissions", "t-co2", "tco2e",
                                 "greenhouse gases", "emissions from fuel"),
    "scope_2_emissions":        ("scope 2", "scope2", "scope ii",
                                 "indirect emission", "purchased electricity",
                                 "energy indirect", "electricity emissions",
                                 "market-based", "location-based"),
    "scope_3_emissions":        ("scope 3", "scope3", "scope iii",
                                 "value chain emission", "other indirect",
                                 "upstream emission", "downstream emission",
                                 "emissions from logistics", "business travel",
                                 "purchased goods"),
    "renewable_energy_pct":     ("renewable", "green electricity",
                                 "green power", "solar", "wind power",
                                 "biomass", "geothermal", "hydroelectric",
                                 "clean energy", "renewable ratio",
                                 "renewable share"),
    "total_energy_consumption": ("energy consumption", "energy use",
                                 "total energy", "energy usage", "electricity",
                                 "power consumption", "city gas", "lpg", "lng",
                                 "fuel consumption", "petroleum products",
                                 "coal products", "energy intensity",
                                 "kwh", "mwh", "gwh"),
    "water_withdrawal":         ("water withdrawal", "water consumption",
                                 "water use", "water intake", "water usage",
                                 "water resource", "water input",
                                 "groundwater", "municipal water",
                                 "water discharge", "water withdrawn"),
    "total_waste_generated":    ("waste", "landfill", "recycled", "recycling",
                                 "hazardous waste", "non-hazardous waste",
                                 "waste generated", "waste volume",
                                 "industrial waste", "waste disposal",
                                 "sludge", "incinerated", "by-product"),
    # ── S ────────────────────────────────────────────────────────────────
    "female_employees_pct":     ("female employee", "women employee",
                                 "gender ratio", "female ratio",
                                 "women in workforce", "proportion of women",
                                 "percentage of women", "share of women",
                                 "female staff", "female workforce",
                                 "gender distribution", "women in management",
                                 "female manager", "diversity"),
    # Corpus reality (measured): gender is disclosed by MANAGEMENT TIER --
    # "senior management / total female", "upper management" -- far more often
    # than as an explicit board percentage. Those rows are the closest printed
    # evidence for board-level gender balance, so they are matched here; the
    # extractor records the report's own label, so a tier figure is never
    # silently passed off as a board figure.
    "female_board_pct":         ("female director", "women on board",
                                 "female board", "women director",
                                 "board diversity", "women on the board",
                                 "female representation on the board",
                                 "gender balance of the board",
                                 "senior management", "executive management",
                                 "upper management", "women in leadership"),
    "employee_turnover_rate":   ("turnover", "attrition", "retention rate",
                                 "employee turnover", "staff turnover",
                                 "voluntary turnover", "leavers",
                                 "separation rate", "retention ratio"),
    "lost_time_injury_rate":    ("lost time", "ltifr", "ltir", "trir",
                                 "injury rate", "accident rate",
                                 "occupational accident", "frequency rate",
                                 "severity rate", "recordable incident",
                                 "work-related injur", "industrial accident",
                                 "fatalities", "zero fatalities",
                                 "number of accidents", "safety record"),
    # ── G ────────────────────────────────────────────────────────────────
    "board_independence_pct":   ("independent director", "board independence",
                                 "outside director", "non-executive director",
                                 "board composition", "independent member",
                                 "outside audit", "board of directors",
                                 "frequency of board", "board meeting",
                                 "directors and audit", "independent officer"),
    "anti_corruption_policy":   ("anti-corruption", "anti corruption",
                                 "bribery", "bribe", "code of conduct",
                                 "business ethic", "ethical guideline",
                                 "zero tolerance", "corruption",
                                 "compliance training", "code of ethics"),
    "whistleblower_mechanism":  ("whistleblow", "speak up", "hotline",
                                 "grievance mechanism", "reporting channel",
                                 "compliance hotline", "internal reporting",
                                 "consultation desk", "helpline"),
    "esg_report_published":     ("gri content index", "sasb content index",
                                 "content index", "reporting scope",
                                 "assurance", "reporting period",
                                 "report boundary", "editorial policy",
                                 "sustainability report", "esg data",
                                 "gri 2-", "gri 3-", "disclosure number",
                                 "disclosure title"),
    "third_party_esg_audit":    ("third party assurance", "independent assurance",
                                 "external assurance", "verification statement",
                                 "independent practitioner", "limited assurance",
                                 "assurance statement", "third-party verification"),
}

# Which pillar each metric scores under -- drives table routing.
METRIC_PILLAR: dict[str, str] = {
    "scope_1_emissions": "E", "scope_2_emissions": "E", "scope_3_emissions": "E",
    "renewable_energy_pct": "E", "total_energy_consumption": "E",
    "water_withdrawal": "E", "total_waste_generated": "E",
    "female_employees_pct": "S", "female_board_pct": "G",
    "employee_turnover_rate": "S", "lost_time_injury_rate": "S",
    "board_independence_pct": "G", "anti_corruption_policy": "G",
    "whistleblower_mechanism": "G", "esg_report_published": "G",
    "third_party_esg_audit": "G",
}


def _match_metrics(haystack: str) -> dict[str, list[str]]:
    """Core metrics whose aliases appear in `haystack`, alias hits per metric."""
    low = haystack.lower()
    out: dict[str, list[str]] = {}
    for key, aliases in METRIC_ALIASES.items():
        hits = [a for a in aliases if a in low]
        if hits:
            out[key] = hits
    return out

# ODL renders some cover/design text as 1xN or Nx1 "tables" (measured on Ebro
# Foods: a disclaimer paragraph became a 1-cell table). Real data tables have
# at least two rows and two columns.
_MIN_ROWS, _MIN_COLS = 2, 2


@dataclass
class TableRecord:
    """One table, kept whole, with everything needed to audit its selection."""
    table_id: str
    page: Optional[int]
    n_rows: int
    n_cols: int
    heading: str                      # nearest preceding heading
    header_row: list[str]
    cells: list[list[str]]            # row-major, header row included
    term_hits: dict[str, list[str]] = field(default_factory=dict)
    metric_hits: dict[str, list[str]] = field(default_factory=dict)

    def score(self, pillar: str) -> int:
        """Relevance to one pillar.

        A core-metric hit counts double a topic-word hit: a table whose row
        label is literally "Water withdrawal" is a direct read of a scored
        benchmark_band factor, whereas a table merely mentioning "climate" is
        context. Ranking must prefer the former when context is tight.
        """
        topical = len(self.term_hits.get(pillar, ()))
        metric = sum(1 for m in self.metric_hits
                     if METRIC_PILLAR.get(m) == pillar)
        return topical + 2 * metric

    def metrics_for(self, pillar: str) -> list[str]:
        return [m for m in self.metric_hits if METRIC_PILLAR.get(m) == pillar]

    def as_markdown(self) -> str:
        """Pipe table -- how the extractor LLM sees it."""
        if not self.cells:
            return ""
        out = []
        if self.heading:
            out.append(f"### {self.heading}  (page {self.page})")
        for i, row in enumerate(self.cells):
            out.append("| " + " | ".join(c.replace("|", "/") for c in row) + " |")
            if i == 0:
                out.append("|" + "|".join(["---"] * len(row)) + "|")
        return "\n".join(out)


@dataclass
class ProseBlock:
    """One heading-delimited prose block, for the embedding channel."""
    block_id: str
    page: Optional[int]
    heading: str
    text: str


def _node_text(node: dict) -> str:
    """Concatenated 'content' of a node and everything under it."""
    out: list[str] = []

    def walk(n) -> None:
        if isinstance(n, dict):
            c = n.get("content")
            if isinstance(c, str) and c.strip():
                out.append(c.strip())
            for v in n.values():
                walk(v)
        elif isinstance(n, list):
            for v in n:
                walk(v)

    walk(node)
    return " ".join(out)


def _cell_text(cell: dict) -> str:
    return " ".join(
        k.get("content", "").strip()
        for k in cell.get("kids", [])
        if isinstance(k, dict) and k.get("content")
    ).strip()


def _table_cells(tbl: dict) -> list[list[str]]:
    grid: list[list[str]] = []
    for row in tbl.get("rows", []):
        grid.append([_cell_text(c) for c in row.get("cells", [])])
    return grid


def _match_terms(haystack: str, pillar: str) -> list[str]:
    low = haystack.lower()
    return sorted({t for t in PILLAR_TERMS[pillar] if t in low})


def run_odl_json(pdf: Path, pages: str | None = None) -> dict:
    """ODL structured output for one PDF. Raises on failure -- caller decides."""
    import opendataloader_pdf

    from agentic_estimation.layer_1.report_parser import _ensure_java

    _ensure_java()
    with tempfile.TemporaryDirectory() as tmp:
        kw = dict(input_path=[str(pdf)], output_dir=tmp, format=["json"],
                  quiet=True)
        if pages:
            kw["pages"] = pages
        opendataloader_pdf.convert(**kw)
        js = list(Path(tmp).rglob("*.json"))
        if not js:
            raise RuntimeError("opendataloader produced no json")
        return json.loads(js[0].read_text(encoding="utf-8", errors="replace"))


def _walk_in_order(doc: dict):
    """Yield content nodes in reading order, carrying the current heading.

    ODL nests nodes under "kids"; a depth-first pass preserves reading order,
    which is what makes "nearest preceding heading" meaningful.
    """
    heading = ""

    def walk(n):
        nonlocal heading
        if isinstance(n, dict):
            t = n.get("type")
            if t == "heading":
                heading = (n.get("content") or "").strip() or heading
                yield ("heading", n, heading)
            elif t in ("table", "paragraph", "list"):
                yield (t, n, heading)
            for k in n.get("kids", []) or []:
                yield from walk(k)
        elif isinstance(n, list):
            for v in n:
                yield from walk(v)

    yield from walk(doc.get("kids", []))


def extract_structure(pdf: Path, pages: str | None = None
                      ) -> tuple[list[TableRecord], list[ProseBlock]]:
    """Split one report into (ESG-relevant tables, prose blocks).

    Tables are filtered here and only here: everything that survives goes whole
    to an extractor. Prose is returned unfiltered -- ranking it is the
    embedding channel's job.
    """
    doc = run_odl_json(pdf, pages=pages)
    tables: list[TableRecord] = []
    prose: list[ProseBlock] = []
    ti = pi = 0

    for kind, node, heading in _walk_in_order(doc):
        page = node.get("page number")
        if kind == "table":
            n_r = int(node.get("number of rows") or 0)
            n_c = int(node.get("number of columns") or 0)
            if n_r < _MIN_ROWS or n_c < _MIN_COLS:
                continue                      # ODL layout artefact, not data
            cells = _table_cells(node)
            if not cells:
                continue
            header = cells[0]
            ti += 1
            # TWO probes, deliberately different scopes.
            #
            # Topical relevance reads heading + header row only: a table full
            # of years and figures would match noise, and topicality is a
            # property of what the table is ABOUT.
            #
            # Metric aliases read the LABEL COLUMN too. ESG data appendices put
            # the metric name in the row label, not the header -- measured on
            # Toyota p37, where the header is "Energy [consolidated]" and the
            # scored values sit in rows "Energy consumption 15,938 TJ" and
            # "Water withdrawal 4,548 km3". Probing headers alone would find
            # the energy table and miss the water metric inside it.
            probe = f"{heading} {' '.join(header)}"
            labels = " ".join(r[0] for r in cells if r)
            hits = {p: _match_terms(probe, p) for p in ("E", "S", "G")}
            metrics = _match_metrics(f"{probe} {labels}")
            # Cross-pillar tables (materiality/SDG/GRI index) are relevant to
            # every extractor -- that is exactly the Toyota scorecard case.
            if any(c in probe.lower() for c in _CROSS_PILLAR):
                for p in ("E", "S", "G"):
                    hits[p] = sorted(set(hits[p]) | {"<cross-pillar>"})
            if not any(hits.values()) and not metrics:
                continue
            tables.append(TableRecord(
                table_id=f"T{ti}", page=page, n_rows=n_r, n_cols=n_c,
                heading=heading, header_row=header, cells=cells,
                term_hits={k: v for k, v in hits.items() if v},
                metric_hits=metrics))
        else:
            txt = _node_text(node)
            if len(txt) < 40:
                continue
            pi += 1
            prose.append(ProseBlock(block_id=f"P{pi}", page=page,
                                    heading=heading, text=txt))

    log.info("[%s] structure: %d ESG tables, %d prose blocks",
             pdf.name[:40], len(tables), len(prose))
    return tables, prose


def tables_for_pillar(tables: list[TableRecord], pillar: str,
                      limit: int = 6) -> list[TableRecord]:
    """Highest term-hit tables for one pillar, whole, best first."""
    hit = [t for t in tables if t.score(pillar) > 0]
    hit.sort(key=lambda t: (-t.score(pillar), t.page or 0))
    return hit[:limit]


if __name__ == "__main__":
    import sys

    p = Path(sys.argv[1])
    pg = sys.argv[2] if len(sys.argv) > 2 else None
    tabs, blocks = extract_structure(p, pages=pg)
    print(f"{len(tabs)} ESG tables, {len(blocks)} prose blocks")
    for t in tabs[:20]:
        print(f"  {t.table_id} p{t.page} {t.n_rows}x{t.n_cols} "
              f"E/S/G={t.score('E')}/{t.score('S')}/{t.score('G')} "
              f"| {t.heading[:38]}")
        if t.metric_hits:
            print(f"      metrics: {', '.join(sorted(t.metric_hits))}")
