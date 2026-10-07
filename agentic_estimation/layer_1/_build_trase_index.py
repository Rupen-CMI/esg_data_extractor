"""
_build_trase_index.py -- ONE-TIME build script, NOT part of the live
pipeline (not imported by public_records_collector.py or anything else
at runtime). Run manually if external_data/trase/*.csv is ever
refreshed (Trase re-publishes datasets periodically, see the
trase.earth/api/data/supply-chains/contexts endpoint for current
model_name/version strings).

Builds external_data/trase/trase_index.db from the 38 unzipped Trase
CSVs (see public_records_collector.py's module docstring for how those
CSVs were obtained and why this needs its own index rather than the
in-memory list[dict] cache pattern every other source in that module
uses).

WHY AN INDEX, NOT IN-MEMORY: the 38 datasets total ~7.5M rows. A naive
first attempt stored one DB row per (source row x company-field)
mention -- 40.6M rows, 12.8GB on disk, because a company appearing
1.4M times (e.g. JBS in the Brazil beef dataset) got 1.4M near-
duplicate copies of its exposure/commitment text. This version
AGGREGATES BY COMPANY NAME FIRST: one row per distinct (entity_name,
role, dataset), with summary stats (record_count, year range, total
volume, an exposure_flag_count, one sample exposure detail string, the
union of any zero-deforestation commitment flags) instead of repeating
raw per-transaction text. Result: ~113k distinct entity-role rows,
24.7MB -- same information content for a per-company lookup, 500x
less disk.

WHY NOT A FIXED COLUMN SCHEMA: Trase's 38 commodity datasets have
genuinely different schemas (confirmed live -- soy/corn/beef use
exporter/exporter_group/importer/importer_group; Cote d'Ivoire cocoa
uses trader_group instead; Indonesia palm oil adds mill/mill_group/
refinery/refinery_group; Indonesia wood pulp uses wood_supplier/
wood_supplier_corporate_group/pulp_mill/pulp_mill_corporate_group
instead of exporter/importer at all). Company-identifying columns are
discovered per-file by regex (COMPANY_FIELD_RE) rather than hardcoded,
so a new Trase dataset with yet another naming convention is still
indexed correctly without a code change here.
"""

import csv
import re
import sqlite3
import time
from collections import defaultdict
from pathlib import Path

TRASE_DIR = Path(__file__).parent.parent.parent / "external_data" / "trase"
DB_PATH = TRASE_DIR / "trase_index.db"

COMPANY_FIELD_RE = re.compile(r"exporter|importer|trader|mill|refinery|supplier", re.IGNORECASE)
# "_id"/"_trase_id" fields (e.g. exporter_id, mill_trase_id) are internal
# numeric/code identifiers, not names -- matched by COMPANY_FIELD_RE's
# substring search but must be excluded.
EXCLUDE_SUFFIXES = ("_id", "_trase_id")
EXPOSURE_RE = re.compile(r"deforestation|exposure|emission", re.IGNORECASE)
# zero_deforestation_* fields are a POSITIVE commitment flag (did this
# exporter sign a zero-deforestation pledge), categorically different
# from the exposure/risk fields EXPOSURE_RE matches -- kept in a
# separate commitment_flags column, not lumped into exposure_summary.
ZERO_DEFORESTATION_RE = re.compile(r"^zero_deforestation", re.IGNORECASE)
_NULLISH = {"", "NA", "NOT ASSESSED", "NONE", None}


def build():
    if DB_PATH.exists():
        DB_PATH.unlink()

    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("""
        CREATE TABLE entities (
            entity_name TEXT NOT NULL,
            role TEXT NOT NULL,
            dataset TEXT NOT NULL,
            record_count INTEGER,
            year_min TEXT,
            year_max TEXT,
            countries TEXT,
            total_volume REAL,
            exposure_flag_count INTEGER,
            sample_exposure_detail TEXT,
            commitment_flags TEXT
        )
    """)
    conn.execute("CREATE INDEX idx_entity_name ON entities(entity_name COLLATE NOCASE)")

    t0 = time.time()
    total_rows_in = 0

    for csv_path in sorted(TRASE_DIR.glob("*.csv")):
        dataset = csv_path.stem
        agg: dict[tuple[str, str], dict] = defaultdict(lambda: {
            "count": 0, "year_min": None, "year_max": None, "countries": set(),
            "volume_sum": 0.0, "exposure_count": 0, "sample_exposure": None, "commitments": set(),
        })

        with open(csv_path, encoding="utf-8", errors="replace", newline="") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames or []
            company_fields = [h for h in fieldnames
                               if COMPANY_FIELD_RE.search(h) and not any(h.lower().endswith(s) for s in EXCLUDE_SUFFIXES)]
            exposure_fields = [h for h in fieldnames if EXPOSURE_RE.search(h) and not ZERO_DEFORESTATION_RE.match(h)]
            commitment_fields = [h for h in fieldnames if ZERO_DEFORESTATION_RE.match(h)]
            year_field = "year" if "year" in fieldnames else None
            country_field = "country_of_production" if "country_of_production" in fieldnames else None
            volume_field = "volume" if "volume" in fieldnames else None

            if not company_fields:
                print(f"{dataset}: SKIPPED, no company-identifying field found")
                continue

            for row in reader:
                total_rows_in += 1
                year = (row.get(year_field) or "").strip() if year_field else ""
                country = (row.get(country_field) or "").strip() if country_field else ""
                try:
                    volume = float(row.get(volume_field) or 0) if volume_field else 0.0
                except ValueError:
                    volume = 0.0

                has_exposure = False
                exposure_sample = None
                for ef in exposure_fields:
                    v = (row.get(ef) or "").strip()
                    if v and v not in _NULLISH:
                        has_exposure = True
                        if exposure_sample is None:
                            exposure_sample = f"{ef}={v}"

                commitments_here = set()
                for cf in commitment_fields:
                    v = (row.get(cf) or "").strip()
                    if v and v not in _NULLISH:
                        commitments_here.add(f"{cf}={v}")

                for field in company_fields:
                    name = (row.get(field) or "").strip()
                    if not name:
                        continue
                    a = agg[(name, field)]
                    a["count"] += 1
                    if year:
                        if a["year_min"] is None or year < a["year_min"]:
                            a["year_min"] = year
                        if a["year_max"] is None or year > a["year_max"]:
                            a["year_max"] = year
                    if country:
                        a["countries"].add(country)
                    a["volume_sum"] += volume
                    if has_exposure:
                        a["exposure_count"] += 1
                        if a["sample_exposure"] is None:
                            a["sample_exposure"] = exposure_sample
                    a["commitments"] |= commitments_here

        batch = []
        for (name, role), a in agg.items():
            batch.append((
                name, role, dataset, a["count"], a["year_min"], a["year_max"],
                ", ".join(sorted(a["countries"]))[:200], round(a["volume_sum"], 2),
                a["exposure_count"], a["sample_exposure"], ", ".join(sorted(a["commitments"]))[:300],
            ))
        conn.executemany(
            """INSERT INTO entities
               (entity_name, role, dataset, record_count, year_min, year_max, countries,
                total_volume, exposure_flag_count, sample_exposure_detail, commitment_flags)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            batch)
        conn.commit()
        print(f"{dataset}: {len(batch)} distinct entities indexed")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s. Total source rows scanned: {total_rows_in}")
    conn.close()
    print(f"DB size: {DB_PATH.stat().st_size / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    build()
