"""Card-swap command-line controls."""

from __future__ import annotations

import json

from scraper.__main__ import build_parser, main
from scraper.card_index import CardIndex
from scraper.card_swapper import CardSwapper, MappingCardSwapper
from scraper.models import RawCard, RawDeck
from scraper.sources import NETWORK_SOURCES, SOURCES


def test_card_swapping_can_be_disabled():
    args = build_parser().parse_args(["--disable-card-swap"])

    assert args.disable_card_swap is True


def test_all_means_every_network_source():
    assert list(NETWORK_SOURCES) == ["limitless", "bulbapedia"]
    assert "text" in SOURCES
    assert "text" not in NETWORK_SOURCES


def test_card_swapper_requires_resolve_implementation():
    class IncompleteSwapper(CardSwapper):
        pass

    assert IncompleteSwapper.__abstractmethods__ == frozenset({"resolve"})

    assert MappingCardSwapper(CardIndex()).resolve(RawCard(1, "Missing")) == ()


def test_all_strategies_fetch_once_and_create_separate_manifests(
    tmp_path, monkeypatch
):
    class FakeSource:
        calls = 0

        def iter_decks(self, **_kwargs):
            type(self).calls += 1
            yield RawDeck(
                source="fake",
                archetype="missing",
                cards=[RawCard(1, "Entirely Missing")],
            )

    monkeypatch.setitem(SOURCES, "fake", FakeSource)

    assert (
        main(
            [
                "--source",
                "fake",
                "--card-swap-strategy",
                "all",
                "--out",
                str(tmp_path),
            ]
        )
        == 0
    )

    assert FakeSource.calls == 1
    for folder in ("mapping-resolved", "heuristic-resolved"):
        manifest = json.loads((tmp_path / folder / "manifest.json").read_text())
        assert manifest["schema_version"] == 3
        assert manifest["decks"] == {}
