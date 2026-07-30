"""Regression tests for source-printing-aware card resolution."""

from scraper.card_index import CardIndex


def test_unknown_set_does_not_fall_back_to_same_name_in_another_set():
    index = CardIndex()

    result = index.match("Eevee", "NOT_A_REAL_SET", "999")

    assert result.card_id is None
    assert result.method == "unresolved"


def test_known_set_does_not_fall_back_to_a_printing_from_another_set():
    index = CardIndex()

    result = index.match("Eevee", "DRI", "999")

    assert result.card_id is None
    assert result.method == "unresolved"


def test_exact_name_respects_the_supplied_set():
    index = CardIndex()

    result = index.match("Eevee", "SFA", "50")

    assert result.card_id == 145
    assert index.by_id[result.card_id].set_code == "SFA"


def test_fuzzy_name_respects_the_supplied_set():
    index = CardIndex()

    result = index.match("Eeveee", "SFA")

    assert result.method == "fuzzy"
    assert result.card_id == 145
    assert index.by_id[result.card_id].set_code == "SFA"
