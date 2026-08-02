"""Integration tests for the fetch-only card-discovery command."""

from __future__ import annotations

from scraper import discovery
from scraper.inventory import read_inventory
from scraper.models import RawCard, RawDeck


class FakeProfileLoader:
    def __init__(self, client=None):  # noqa: ARG002
        return None

    @staticmethod
    def canonical_set_code(value):
        return {"Obsidian Flames": "OBF"}.get(value, value)

    def __call__(self, _card):
        return None


class FakeSource:
    @staticmethod
    def iter_decks(**kwargs):
        assert kwargs["max_decks"] == 5000
        assert kwargs["bulbapedia_max_pages"] == 200
        yield RawDeck(
            "fake",
            "Example",
            [RawCard(2, "Charmeleon", "Obsidian Flames", "027", "pokemon")],
            external_ids={"deck": "1"},
        )


def test_discovery_writes_inventory_only_and_resumes_idempotently(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(discovery, "NETWORK_SOURCES", {"fake": FakeSource})
    monkeypatch.setattr(discovery, "LimitlessProfileLoader", FakeProfileLoader)
    output = tmp_path / "seen_cards.jsonl.gz"
    args = ["--source", "all", "--out", str(output), "--checkpoint-every", "1"]

    assert discovery.main(args) == 0
    assert discovery.main(args) == 0

    (record,) = read_inventory(output)
    assert record.identity.name == "charmeleon"
    assert record.identity.set_code == "OBF"
    assert record.identity.number == "27"
    assert record.counts_by_source["fake"].decks == 1
    assert not list(tmp_path.rglob("*.csv"))
    assert not list(tmp_path.rglob("manifest.json"))


def test_negative_checkpoint_interval_fails_before_fetching(monkeypatch):
    monkeypatch.setattr(discovery, "NETWORK_SOURCES", {"fake": FakeSource})

    assert discovery.main(["--source", "all", "--checkpoint-every", "-1"]) == 2


def test_set_catalogue_failure_skips_observation_and_retries_canonically(
    tmp_path, monkeypatch
):
    class TwoDeckSource:
        @staticmethod
        def iter_decks(**_kwargs):
            for deck_id in ("first", "second"):
                yield RawDeck(
                    "fake",
                    deck_id,
                    [RawCard(1, "Charmeleon", "Obsidian Flames", "027")],
                    external_ids={"deck": deck_id},
                )

    class TransientCanonicalProfileLoader(FakeProfileLoader):
        fail_once = True

        @classmethod
        def canonical_set_code(cls, value):
            if cls.fail_once:
                cls.fail_once = False
                raise RuntimeError("set catalogue unavailable")
            return super().canonical_set_code(value)

    monkeypatch.setattr(discovery, "NETWORK_SOURCES", {"fake": TwoDeckSource})
    monkeypatch.setattr(
        discovery, "LimitlessProfileLoader", TransientCanonicalProfileLoader
    )
    output = tmp_path / "seen_cards.jsonl.gz"
    args = ["--source", "all", "--out", str(output)]

    assert discovery.main(args) == 1
    (record,) = read_inventory(output)
    assert record.identity.set_code == "OBF"
    assert record.counts_by_source["fake"].decks == 1

    assert discovery.main(args) == 0
    (record,) = read_inventory(output)
    assert record.identity.set_code == "OBF"
    assert record.counts_by_source["fake"].decks == 2


def test_source_failure_checkpoints_partial_inventory_and_returns_nonzero(
    tmp_path, monkeypatch
):
    class PartialSource:
        @staticmethod
        def iter_decks(**_kwargs):
            yield RawDeck(
                "partial",
                "Kept",
                [RawCard(1, "Missing")],
                external_ids={"deck": "kept"},
            )
            raise RuntimeError("remote failed")

    monkeypatch.setattr(discovery, "NETWORK_SOURCES", {"partial": PartialSource})
    monkeypatch.setattr(discovery, "LimitlessProfileLoader", FakeProfileLoader)
    output = tmp_path / "seen_cards.jsonl.gz"

    assert discovery.main(["--source", "all", "--out", str(output)]) == 1
    assert len(read_inventory(output)) == 1
