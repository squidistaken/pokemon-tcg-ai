"""Regression tests for source-printing-aware card resolution."""

from scraper.card_index import CardIndex


def test_unknown_set_falls_back_to_the_same_name_in_another_set():
    index = CardIndex()

    result = index.match("Eevee", "NOT_A_REAL_SET", "999")

    assert index.by_id[result.card_id].name == "Eevee"
    assert result.method == "exact"


def test_known_set_falls_back_to_a_printing_from_another_set():
    index = CardIndex()

    result = index.match("Eevee", "DRI", "999")

    assert index.by_id[result.card_id].name == "Eevee"
    assert result.method == "exact"


def test_number_picks_the_printing_when_the_set_is_a_reprint():
    index = CardIndex()

    result = index.match("Eevee", "DRI", "50")

    assert result.card_id == 145


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
