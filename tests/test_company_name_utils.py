"""normalize_company_name -- the canonical name-normal-form function used by
peer_anchor_collector.py's _drop_self ground-truth-leakage guard."""
from agentic_estimation.shared.company_name_utils import normalize_company_name


def test_strips_single_tail_suffix():
    assert normalize_company_name("Alpkit Ltd") == "alpkit"


def test_strips_stacked_tail_suffixes():
    assert normalize_company_name("Foo Bar Ltd Co") == "foo bar"


def test_keeps_single_char_tokens():
    assert normalize_company_name("A B C Inc") == "a b c"


def test_all_suffix_name_returns_empty():
    assert normalize_company_name("Ltd") == ""
    assert normalize_company_name("Ltd Co") == ""


def test_empty_name_returns_empty():
    assert normalize_company_name("") == ""


def test_punctuation_and_casing_normalized():
    assert normalize_company_name("Accès Personnel SA") == normalize_company_name("ACCÈS PERSONNEL sa")


def test_periods_in_suffix_split_into_separate_tokens():
    """"S.A." with periods tokenizes to two single-char tokens ("s", "a"), not
    the single "sa" LEGAL_SUFFIXES entry -- documents actual behavior so a
    caller passing punctuated suffixes isn't surprised they don't collapse
    the way the punctuation-free form does."""
    assert normalize_company_name("Foo S.A.") == "foo s a"
    assert normalize_company_name("Foo SA") == "foo"


def test_suffix_only_stripped_from_tail_not_mid_name():
    """A suffix-looking token in the MIDDLE of a name is not a legal suffix
    there -- only trailing tokens strip."""
    assert normalize_company_name("Group Therapy Solutions") == "group therapy solutions"
