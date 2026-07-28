"""
peer_anchor_collector.py's ground-truth-leakage guard (_drop_self / find_peers).
Confirmed live during the DEFECT_FIX_PLAN.md 0.1 audit: bcorp_lookup/upright_lookup
company names carry legal suffixes and punctuation ("Fish Tales Holding BV",
"Acces Personnel SA") that a plain SQL `!= %s` exact-string filter does not
catch when the caller's exclude_name is a bare/differently-punctuated form of
the same company -- letting a labeled company appear in its own peer set
(literal ground-truth leakage in the backtest). _find_real_metric_peers had
NO exclusion at all before this fix.

No DB access -- psycopg2.connect is monkeypatched with a fake cursor so these
run fast and don't need ASYNC_DB_URL.
"""
import pytest

from agentic_estimation.layer_1.peer_anchor_collector import _drop_self, find_peers, PeerRecord


# ── _drop_self ──────────────────────────────────────────────────────────────

def test_drop_self_exact_match():
    peers = [PeerRecord(source="bcorp", name="Alpkit", sector=None, country=None,
                         size_bucket=None, fields={})]
    assert _drop_self(peers, "Alpkit") == []


def test_drop_self_casing_variant():
    peers = [PeerRecord(source="bcorp", name="ALPKIT LTD", sector=None, country=None,
                         size_bucket=None, fields={})]
    assert _drop_self(peers, "alpkit ltd") == []


def test_drop_self_legal_suffix_variant():
    """The exact leakage shape found live: a labeled company's DB row carries
    a legal suffix the caller's plain name doesn't."""
    peers = [PeerRecord(source="upright", name="Fish Tales Holding BV", sector=None,
                         country=None, size_bucket=None, fields={})]
    assert _drop_self(peers, "Fish Tales") == []


def test_drop_self_stacked_suffix_variant():
    """normalize_company_name strips suffixes from the tail REPEATEDLY, not
    just once -- "Foo Bar Ltd Co" must still match "Foo Bar"."""
    peers = [PeerRecord(source="bcorp", name="Foo Bar Ltd Co", sector=None,
                         country=None, size_bucket=None, fields={})]
    assert _drop_self(peers, "Foo Bar") == []


def test_drop_self_keeps_genuine_peers():
    peers = [
        PeerRecord(source="bcorp", name="Alpkit", sector=None, country=None, size_bucket=None, fields={}),
        PeerRecord(source="bcorp", name="Patagonia Inc.", sector=None, country=None, size_bucket=None, fields={}),
    ]
    kept = _drop_self(peers, "Alpkit")
    assert [p.name for p in kept] == ["Patagonia Inc."]


def test_drop_self_empty_normal_form_falls_back_to_plain_compare():
    """An all-suffix exclude_name ("Ltd" alone) normalizes to "" -- must NOT
    treat "" == "" as a match and wipe out every peer with an unparseable
    name. Falls back to a plain strip/lower compare instead."""
    peers = [
        PeerRecord(source="bcorp", name="Ltd", sector=None, country=None, size_bucket=None, fields={}),
        PeerRecord(source="bcorp", name="Real Company Inc.", sector=None, country=None, size_bucket=None, fields={}),
    ]
    kept = _drop_self(peers, "Ltd")
    assert [p.name for p in kept] == ["Real Company Inc."]


def test_drop_self_no_exclude_name_is_noop():
    peers = [PeerRecord(source="bcorp", name="Alpkit", sector=None, country=None, size_bucket=None, fields={})]
    assert _drop_self(peers, None) == peers
    assert _drop_self(peers, "") == peers


# ── find_peers smoke test (mocked cursor, no DB) ─────────────────────────────

class _FakeCursor:
    def __init__(self, rows):
        self._rows = rows

    def execute(self, query, params):
        self.last_query = query
        self.last_params = params

    def fetchall(self):
        return self._rows

    def close(self):
        pass


class _FakeConn:
    def __init__(self, rows):
        self._rows = rows

    def cursor(self):
        return _FakeCursor(self._rows)

    def close(self):
        pass


def test_find_peers_metric_key_excludes_self(monkeypatch):
    """Regression for the confirmed gap: find_peers(metric_key=...) used to
    silently drop exclude_name before it ever reached _find_real_metric_peers,
    so the target company's OWN real metric row could return as its own peer."""
    import agentic_estimation.layer_1.peer_anchor_collector as pac

    # bcorp/upright disabled for this test -- isolate the metric_key path.
    fake_rows = [("Target Co", "US", 1000.0, 2025), ("Peer Co", "US", 2000.0, 2025)]

    def fake_conn():
        return _FakeConn(fake_rows)

    monkeypatch.setattr(pac, "_db_conn", fake_conn)
    peers = pac.find_peers(
        metric_key="employee_count", country="US", exclude_name="Target Co",
        include_bcorp=False, include_upright=False,
    )
    names = [p.name for p in peers]
    assert "Target Co" not in names
    assert "Peer Co" in names
