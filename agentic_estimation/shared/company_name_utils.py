"""
company_name_utils.py — single source of truth for legal-entity-suffix tokens
used when matching/normalising company names.

Both agentic_estimation/layer_1/company_metadata.py (fuzzy bidirectional
token-overlap matching against Wikidata/GLEIF results) and
agentic_estimation/layer_2/climate_trace_anchor.py (strict exact/prefix
matching against Climate TRACE owner names) strip legal suffixes before
comparing names, using different matching ALGORITHMS for different purposes
-- but they must agree on what counts as a legal suffix, or the same company
pair (e.g. "Smith Jones LLP" vs "Smith Jones") can match in one module and
not the other. Found live during code review: the two suffix lists had
already drifted apart (LLP, Group, Holdings, SAS, Sdn Bhd were only in one
of the two).
"""

# Lowercase legal-entity suffix tokens, no punctuation. Consumers strip
# punctuation before splitting into tokens, so "S.A." and "SA" both reduce to "sa".
LEGAL_SUFFIXES: frozenset[str] = frozenset({
    "inc", "incorporated", "corp", "corporation", "ltd", "limited", "llc",
    "llp", "plc", "co", "company", "group", "holding", "holdings",
    "sa", "sau", "se", "ag", "gmbh", "bv", "nv", "srl", "spa", "sarl", "sas",
    "ab", "as", "oy", "kk", "pte", "pty", "pvt", "sdn", "bhd",
})


def normalize_company_name(name: str) -> str:
    """Canonical company-name normal form for self-exclusion / dedup matching:
    lowercase, punctuation -> space, tokenize, strip LEGAL_SUFFIXES tokens
    from the TAIL only (repeatedly -- "Foo Bar Ltd Co" -> "foo bar", not just
    one pass), keep single-char tokens. Returns "" for an all-suffix or empty
    name (e.g. "Ltd" alone) -- callers must handle that case explicitly (an
    empty normal form can never be trusted to positively identify a company).

    This is agentic_estimation/layer_2/climate_trace_anchor.py's
    _normalise + _strip_suffixes semantics, promoted here as the one
    implementation both that module and peer_anchor_collector.py's
    self-exclusion should share -- company_metadata.py's own _norm() drops
    single-char tokens for its fuzzy-overlap scoring and is intentionally
    left separate (see that module's docstring).

    KNOWN BLIND SPOT: punctuation-to-space runs BEFORE suffix-matching, so a
    period-separated suffix ("S.A.", "Co.") splits into single-char tokens
    ("s", "a") that don't match LEGAL_SUFFIXES' unpunctuated "sa" entry --
    "Foo S.A." normalizes to "foo s a", not "foo" (contrast "Foo SA" -> "foo").
    Inherited unchanged from climate_trace_anchor's existing behavior; not
    fixed here to avoid changing established matching behavior without a
    dedicated pass across both consumers."""
    import re
    cleaned = re.sub(r"[^\w\s]", " ", name.lower())
    tokens = [t for t in cleaned.split() if t]
    while tokens and tokens[-1] in LEGAL_SUFFIXES:
        tokens = tokens[:-1]
    return " ".join(tokens)
