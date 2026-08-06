"""Regression tests for source-printing-aware card resolution."""

from scraper.card_index import CardIndex


def test_unknown_set_exposes_ambiguous_same_name_candidates():
    index = CardIndex()

    result = index.match("Eevee", "NOT_A_REAL_SET", "999")

    assert result.card_id is None
    assert result.method == "ambiguous"
    assert len(result.candidate_ids) > 1


def test_known_set_does_not_silently_choose_a_same_name_printing():
    index = CardIndex()

    result = index.match("Eevee", "DRI", "999")

    assert result.card_id is None
    assert result.method == "ambiguous"


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


def test_gameplay_profile_aggregates_multirow_cards():
    profile = CardIndex().profiles[788]

    assert profile.name == "Charmander"
    assert profile.hp == 80
    assert len(profile.moves) == 2
