"""Tests for the hardcoded staple-substitute table."""

import pytest

from scraper.card_index import CardIndex
from scraper.card_swapper import CardSwapper


def test_default_mapping_targets_all_resolve_against_the_real_pool():
    index = CardIndex()

    CardSwapper(index)  # must not raise: every DEFAULT_SWAP_MAP target is real


def test_lookup_hits_a_seeded_missing_staple():
    index = CardIndex()
    swapper = CardSwapper(index)

    result = swapper.lookup("Iono")

    assert result is not None
    sub_name, sub_id = result
    assert sub_name == "Judge"
    assert index.by_id[sub_id].name == "Judge"


def test_lookup_is_name_normalized():
    index = CardIndex()
    swapper = CardSwapper(index)

    assert swapper.lookup("iono") == swapper.lookup("Iono")


def test_lookup_misses_an_unmapped_name():
    index = CardIndex()
    swapper = CardSwapper(index)

    assert swapper.lookup("Some Totally Unmapped Card") is None


def test_constructor_rejects_a_substitute_missing_from_the_pool():
    index = CardIndex()

    with pytest.raises(ValueError):
        CardSwapper(index, mapping={"Some Card": "Not A Real Card Name At All"})
