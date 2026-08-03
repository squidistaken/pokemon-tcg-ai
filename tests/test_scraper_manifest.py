"""Tests for the deck corpus manifest and the hash contract it rests on.

The contract under test: a deck's ``id_hash`` is **card-order agnostic** but
**card-copy-count sensitive**. Reorderings of the same multiset are the same deck
and must collapse onto one entry; a different number of copies of any card is a
different deck. Everything about dedup and ``observation_count`` follows from that,
so the round trip (save -> reload from CSV -> re-hash -> compare to the manifest) is
pinned down here rather than assumed.
"""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

import pytest

from scraper import manifest as manifest_mod
from scraper.analysis.loading import count_unique_decks
from scraper.analysis.prune import prune_near_duplicates
from scraper.card_index import CardIndex
from scraper.manifest import DeckEntry, Manifest, ManifestError, Observation
from scraper.models import CardSwap, RawCard, RawDeck, ResolvedDeck
from scraper.pipeline import RunSummary, process_deck
from scraper.writer import DeckWriter, deck_hash, read_deck_ids
from src.env.deck import load_deck

EXAMPLE_DECK = Path(__file__).parents[1] / "decks" / "example.csv"


def make_deck(ids: list[int], **raw_kwargs) -> ResolvedDeck:
    """
    Build a ResolvedDeck around explicit card IDs, bypassing name resolution.

    :param ids: Card IDs, one entry per copy, in the order a source listed them.
    :param raw_kwargs: Provenance overrides for the underlying RawDeck.
    :return: A :class:`~scraper.models.ResolvedDeck` ready to hand to a writer.
    """
    fields = {"source": "test", "archetype": "Test Deck"} | raw_kwargs
    return ResolvedDeck(source_deck=RawDeck(**fields), ids=list(ids))


def sixty(seed: int = 0) -> list[int]:
    """
    A plausible 60-card deck: 15 distinct cards, 4 copies each.

    :param seed: Offsets the card IDs so different seeds are different decks.
    :return: 60 Card IDs, one entry per copy.
    """
    return [100 + seed * 100 + i for i in range(15) for _ in range(4)]


# --------------------------------------------------------------------------- #
# The hash contract
# --------------------------------------------------------------------------- #


def test_hash_is_order_agnostic():
    """Any permutation of a deck hashes identically."""
    ids = sixty()
    baseline = deck_hash(ids)
    rng = random.Random(1234)
    for _ in range(50):
        shuffled = ids[:]
        rng.shuffle(shuffled)
        assert deck_hash(shuffled) == baseline


def test_hash_is_order_agnostic_for_reordered_copies():
    """Interleaving copies of the same cards does not change the hash.

    This is the case the corpus actually hits: sources group cards differently
    (pokemon/trainer/energy, or alphabetically), so the same list arrives in
    different orders with copies clumped or spread out.
    """
    grouped = [1, 1, 1, 2, 2, 3]
    interleaved = [3, 1, 2, 1, 2, 1]
    assert sorted(grouped) == sorted(interleaved)
    assert deck_hash(grouped) == deck_hash(interleaved)


def test_hash_is_copy_count_sensitive():
    """Three copies and four copies of the same card are different decks."""
    three = [1, 1, 1, 2, 2, 2]
    four = [1, 1, 1, 1, 2, 2]
    assert deck_hash(three) != deck_hash(four)


def test_hash_distinguishes_redistributed_copies():
    """Same distinct cards, same total, different copy split -> different hash.

    Guards against a regression to set/``frozenset`` semantics, which would call
    these two decks identical.
    """
    a = [1] * 4 + [2] * 2 + [3] * 1  # 4/2/1
    b = [1] * 3 + [2] * 3 + [3] * 1  # 3/3/1
    assert set(a) == set(b)
    assert len(a) == len(b)
    assert deck_hash(a) != deck_hash(b)


def test_hash_separator_prevents_id_run_together():
    """Adjacent IDs cannot smear into each other: [11, 2] is not [1, 12]."""
    assert deck_hash([11, 2]) != deck_hash([1, 12])
    assert deck_hash([1, 112]) != deck_hash([11, 12])


def test_hash_rejects_empty_deck():
    """An empty deck gets no hash — it must not mint a stable key for nothing."""
    with pytest.raises(ValueError, match="empty deck"):
        deck_hash([])


def test_hash_rejects_string_ids():
    """String IDs are refused rather than silently hashed under a lexicographic sort.

    ``sorted(["9", "10"]) == ["10", "9"]`` but ``sorted([9, 10]) == [9, 10]``, so
    accepting strings would give the same deck two different hashes depending on
    how its IDs happened to be typed.
    """
    with pytest.raises(TypeError, match="must be ints"):
        deck_hash(["9", "10", "100"])  # type: ignore[list-item]


def test_hash_is_stable_across_processes():
    """The digest is pinned, so hashes in an existing manifest stay valid."""
    assert deck_hash([1, 2, 2]) == "9d8d5ed5834f"
    assert len(deck_hash(sixty())) == 12


# --------------------------------------------------------------------------- #
# Save -> reload -> re-hash: the manifest agrees with the corpus on disk
# --------------------------------------------------------------------------- #


def test_saved_decks_reload_to_their_manifest_hashes(tmp_path):
    """Save several decks, reload them from disk, and re-hash: all must match.

    This is the manifest's core promise — ``id_hash`` describes the file next to
    it — and it is what makes a damaged manifest rebuildable from the corpus.
    """
    writer = DeckWriter(str(tmp_path))
    for seed in range(5):
        result = writer.write(make_deck(sixty(seed), archetype=f"Deck {seed}"))
        assert result.new_deck

    manifest = manifest_mod.load(tmp_path)
    assert len(manifest.decks) == 5

    for slug, entry in manifest.decks.items():
        reloaded = load_deck(str(tmp_path / entry.file))  # the engine's own loader
        assert len(reloaded) == 60
        assert deck_hash(reloaded) == entry.id_hash, f"{slug} does not match its file"


def test_written_csv_keeps_source_order_yet_still_matches(tmp_path):
    """The CSV preserves the order the source listed cards; the hash sorts anyway."""
    ids = sixty()
    rng = random.Random(7)
    scrambled = ids[:]
    rng.shuffle(scrambled)

    writer = DeckWriter(str(tmp_path))
    result = writer.write(make_deck(scrambled))
    entry = writer.manifest.decks[result.slug]

    on_disk = read_deck_ids(str(tmp_path / entry.file))
    assert on_disk == scrambled  # not sorted on the way out
    assert deck_hash(on_disk) == entry.id_hash
    assert entry.id_hash == deck_hash(ids)  # ... and equals the canonical order


def test_reloaded_example_deck_matches_its_hash(tmp_path):
    """A real 60-card corpus deck round-trips through the writer unchanged."""
    ids = load_deck(str(EXAMPLE_DECK))
    writer = DeckWriter(str(tmp_path))
    result = writer.write(make_deck(ids, archetype="Example"))

    entry = writer.manifest.decks[result.slug]
    assert deck_hash(load_deck(str(tmp_path / entry.file))) == entry.id_hash
    assert sorted(read_deck_ids(str(tmp_path / entry.file))) == sorted(ids)


def test_whole_corpus_agrees_after_reopening(tmp_path):
    """Reopening the corpus in a fresh writer reproduces every hash."""
    writer = DeckWriter(str(tmp_path))
    for seed in range(4):
        writer.write(make_deck(sixty(seed), archetype=f"Deck {seed}"))

    reopened = DeckWriter(str(tmp_path))
    assert len(reopened.manifest.decks) == 4
    for entry in reopened.manifest.decks.values():
        assert deck_hash(read_deck_ids(str(tmp_path / entry.file))) == entry.id_hash


# --------------------------------------------------------------------------- #
# Dedup: reordered copies collapse, copy-count changes do not
# --------------------------------------------------------------------------- #


def test_reordered_duplicate_is_deduplicated(tmp_path):
    """A reordering of a saved deck writes no second file — it is the same deck."""
    ids = sixty()
    rng = random.Random(99)
    reordered = ids[:]
    rng.shuffle(reordered)

    writer = DeckWriter(str(tmp_path))
    first = writer.write(make_deck(ids, archetype="Charizard ex", event="Event A"))
    second = writer.write(make_deck(reordered, archetype="Charizard ex", event="Event B"))

    assert first.new_deck
    assert not second.new_deck
    assert second.slug == first.slug
    assert len(writer.manifest.decks) == 1
    assert len(list(tmp_path.rglob("*.csv"))) == 1


def test_copy_count_difference_writes_a_second_deck(tmp_path):
    """One card swapped from 4 copies to 3+1 is a different deck, not a duplicate."""
    base = sixty()
    variant = base[:-1] + [9999]  # drop a 4th copy, add a different card

    writer = DeckWriter(str(tmp_path))
    first = writer.write(make_deck(base, archetype="Deck"))
    second = writer.write(make_deck(variant, archetype="Deck"))

    assert first.new_deck
    assert second.new_deck
    assert first.slug != second.slug
    assert len(writer.manifest.decks) == 2
    assert len(list(tmp_path.rglob("*.csv"))) == 2


def test_dedup_survives_a_writer_restart(tmp_path):
    """Dedup is driven by the manifest, so it holds across separate runs."""
    ids = sixty()
    DeckWriter(str(tmp_path)).write(make_deck(ids))

    later = DeckWriter(str(tmp_path))
    assert later.is_duplicate(ids)
    result = later.write(make_deck(list(reversed(ids)), event="A Later Event"))
    assert not result.new_deck
    assert len(later.manifest.decks) == 1


# --------------------------------------------------------------------------- #
# Observations: duplicates are counted, not discarded
# --------------------------------------------------------------------------- #


def occurrence(**kwargs) -> dict:
    """
    Provenance for one occurrence of a deck at a tournament.

    :param kwargs: Overrides for any of the provenance fields.
    :return: RawDeck keyword arguments.
    """
    return {
        "archetype": "Gardevoir ex",
        "event": "Chicago Regional",
        "event_date": "2026-05-10",
        "record": "8-1-0",
        "placing": 3,
        "url": "https://play.limitlesstcg.com/tournament/abc/standings",
        "external_ids": {"tournament_id": "abc", "placing": "3"},
    } | kwargs


def test_duplicate_increments_observation_count(tmp_path):
    """A second occurrence of the same 60 cards bumps the count instead of vanishing."""
    ids = sixty()
    writer = DeckWriter(str(tmp_path))
    writer.write(make_deck(ids, **occurrence()), date="2026-05-11")

    second = writer.write(
        make_deck(
            list(reversed(ids)),
            **occurrence(
                event="Toronto Regional",
                event_date="2026-06-14",
                record="7-2-0",
                placing=11,
                url="https://play.limitlesstcg.com/tournament/xyz/standings",
                external_ids={"tournament_id": "xyz", "placing": "11"},
            ),
        ),
        date="2026-06-15",
    )

    assert not second.new_deck
    assert second.new_observation
    entry = writer.manifest.decks[second.slug]
    assert entry.observation_count == 2


def test_each_occurrence_retains_its_own_provenance(tmp_path):
    """Every occurrence keeps its event, dates, record, placing, URL and IDs."""
    ids = sixty()
    writer = DeckWriter(str(tmp_path))
    writer.write(make_deck(ids, **occurrence()), date="2026-05-11")
    writer.write(
        make_deck(
            list(reversed(ids)),
            **occurrence(
                event="Toronto Regional",
                event_date="2026-06-14",
                record="7-2-0",
                placing=11,
                url="https://play.limitlesstcg.com/tournament/xyz/standings",
                external_ids={"tournament_id": "xyz", "placing": "11"},
            ),
        ),
        date="2026-06-15",
    )

    entry = manifest_mod.load(tmp_path).decks["gardevoir-ex"]
    first, second = entry.observations

    assert first.event == "Chicago Regional"
    assert first.event_date == "2026-05-10"
    assert first.scraped_date == "2026-05-11"
    assert first.record == "8-1-0"
    assert first.placing == 3
    assert first.url == "https://play.limitlesstcg.com/tournament/abc/standings"
    assert first.external_ids == {"tournament_id": "abc", "placing": "3"}

    assert second.event == "Toronto Regional"
    assert second.record == "7-2-0"
    assert second.placing == 11
    assert second.external_ids == {"tournament_id": "xyz", "placing": "11"}

    # The deck-level span covers both occurrences, by event date.
    assert entry.first_seen == "2026-05-10"
    assert entry.last_seen == "2026-06-14"


def test_rescraping_the_same_occurrence_changes_nothing(tmp_path):
    """Re-running a scrape must not inflate the count; the occurrence is identified.

    Without this the popularity signal would just track how often the scraper ran.
    """
    ids = sixty()
    writer = DeckWriter(str(tmp_path))
    writer.write(make_deck(ids, **occurrence()), date="2026-05-11")

    # Same standing, fetched again on a later day (so only scraped_date differs).
    again = DeckWriter(str(tmp_path))
    result = again.write(make_deck(ids, **occurrence()), date="2026-07-29")

    assert result.reobserved
    assert not result.new_deck
    assert not result.new_observation
    assert again.manifest.decks[result.slug].observation_count == 1


def test_warnings_from_a_later_occurrence_are_persisted(tmp_path):
    """A warning first seen on a repeat occurrence reaches the file, not just memory."""
    ids = sixty()
    writer = DeckWriter(str(tmp_path))
    writer.write(make_deck(ids, **occurrence()))
    writer.write(
        make_deck(ids, **occurrence(external_ids={"tournament_id": "xyz"})),
        warnings=["'Vaporeon' has no 'Eevee' in the deck"],
    )

    reloaded = manifest_mod.load(tmp_path).decks["gardevoir-ex"]
    assert reloaded.warnings == ["'Vaporeon' has no 'Eevee' in the deck"]
    assert reloaded.observation_count == 2


def test_observation_count_is_derived_from_the_observations(tmp_path):
    """The stored count cannot drift from the list it summarises."""
    ids = sixty()
    writer = DeckWriter(str(tmp_path))
    for i in range(3):
        writer.write(
            make_deck(ids, **occurrence(placing=i, external_ids={"tournament_id": f"t{i}"}))
        )

    raw = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    entry = raw["decks"]["gardevoir-ex"]
    assert entry["observation_count"] == 3
    assert entry["observation_count"] == len(entry["observations"])


def test_bulbapedia_style_occurrences_stay_distinct(tmp_path):
    """Two decklists with no event, record, placing or date are still told apart.

    Bulbapedia gives none of the descriptive fields, so without source-native IDs
    (page + table index) two tables on one page would look like one occurrence.
    """
    writer = DeckWriter(str(tmp_path))
    ids = sixty()
    common = {
        "source": "bulbapedia",
        "archetype": "Abyss (TCG)",
        "url": "https://bulbapedia.bulbagarden.net/wiki/Abyss_(TCG)",
    }
    writer.write(make_deck(ids, **common, external_ids={"page": "Abyss (TCG)", "table_index": "0"}))
    result = writer.write(
        make_deck(ids, **common, external_ids={"page": "Abyss (TCG)", "table_index": "1"})
    )

    assert result.new_observation
    assert writer.manifest.decks[result.slug].observation_count == 2


def test_occurrence_key_ignores_the_scrape_date():
    """Scrape date is provenance, never identity — else every day looks new."""
    a = Observation(source="s", url="u", event="e", placing=1, record="5-0-0", scraped_date="2026-01-01")
    b = Observation(source="s", url="u", event="e", placing=1, record="5-0-0", scraped_date="2026-09-09")
    assert a.key() == b.key()


def test_occurrence_key_distinguishes_placings():
    """Two players at one event are two occurrences, not one."""
    a = Observation(source="limitless", external_ids={"tournament_id": "t", "placing": "1"})
    b = Observation(source="limitless", external_ids={"tournament_id": "t", "placing": "2"})
    assert a.key() != b.key()


# --------------------------------------------------------------------------- #
# Manifest file: schema, upgrade, durability
# --------------------------------------------------------------------------- #


def test_manifest_file_carries_its_schema_header(tmp_path):
    """The file says which schema and hash algorithm produced it."""
    DeckWriter(str(tmp_path)).write(make_deck(sixty()))
    raw = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert raw["schema_version"] == manifest_mod.SCHEMA_VERSION
    assert raw["hash_algo"] == manifest_mod.HASH_ALGO
    assert set(raw) == {"schema_version", "hash_algo", "decks"}


def test_v1_manifest_upgrades_its_single_occurrence(tmp_path):
    """A pre-observations manifest is read, not discarded: its entry becomes one
    observation, and the deck still dedupes against incoming scrapes."""
    ids = sixty()
    (tmp_path / "charizard-ex").mkdir()
    (tmp_path / "charizard-ex" / "charizard-ex.csv").write_text(
        "\n".join(str(i) for i in ids) + "\n", encoding="utf-8"
    )
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "charizard-ex": {
                    "file": "charizard-ex/charizard-ex.csv",
                    "source": "limitless",
                    "archetype": "Charizard ex",
                    "event": "Old Event",
                    "placing": 1,
                    "url": "https://example.test/old",
                    "format": "standard",
                    "record": "9-0-0",
                    "date": "2026-01-05",
                    "id_hash": deck_hash(ids),
                }
            }
        ),
        encoding="utf-8",
    )

    writer = DeckWriter(str(tmp_path))
    entry = writer.manifest.decks["charizard-ex"]
    assert writer.manifest.read_version == 1
    assert entry.observation_count == 1
    assert entry.observations[0].event == "Old Event"
    # v1 never recorded the event's own date, only the scrape date.
    assert entry.observations[0].scraped_date == "2026-01-05"
    assert entry.observations[0].event_date is None

    result = writer.write(make_deck(list(reversed(ids)), archetype="Charizard ex", event="New Event"))
    assert not result.new_deck
    assert result.slug == "charizard-ex"
    assert writer.manifest.decks["charizard-ex"].observation_count == 2

    raw = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert raw["schema_version"] == manifest_mod.SCHEMA_VERSION


def test_corrupt_manifest_is_fatal_not_silently_replaced(tmp_path):
    """A damaged manifest stops the run rather than being overwritten.

    It holds accumulated provenance a fresh scrape cannot reconstruct, so starting
    from empty would destroy the corpus's history.
    """
    (tmp_path / "manifest.json").write_text("{ this is not json", encoding="utf-8")
    with pytest.raises(ManifestError, match="cannot read manifest"):
        DeckWriter(str(tmp_path))


def test_newer_schema_is_not_downgraded(tmp_path):
    """A manifest from a future scraper is refused rather than rewritten lossily."""
    (tmp_path / "manifest.json").write_text(
        json.dumps({"schema_version": manifest_mod.SCHEMA_VERSION + 1, "decks": {}}),
        encoding="utf-8",
    )
    with pytest.raises(ManifestError, match="newer than this scraper"):
        DeckWriter(str(tmp_path))


def test_interrupted_save_leaves_the_previous_manifest_intact(tmp_path, monkeypatch):
    """Saves are atomic: a crash mid-write cannot truncate the manifest."""
    writer = DeckWriter(str(tmp_path))
    writer.write(make_deck(sixty(0), archetype="Deck 0"))
    good = (tmp_path / "manifest.json").read_text(encoding="utf-8")

    def explode(*_args, **_kwargs):
        """Fail partway through serialising, as a full disk or a kill would."""
        raise OSError("disk full")

    monkeypatch.setattr(manifest_mod.json, "dump", explode)
    with pytest.raises(OSError, match="disk full"):
        writer.write(make_deck(sixty(1), archetype="Deck 1"))

    assert (tmp_path / "manifest.json").read_text(encoding="utf-8") == good
    monkeypatch.undo()
    assert manifest_mod.load(tmp_path).decks  # still parseable


def test_save_leaves_no_temporary_file_behind(tmp_path):
    """The atomic rename cleans up after itself."""
    DeckWriter(str(tmp_path)).write(make_deck(sixty()))
    assert not list(tmp_path.glob("manifest.json.tmp"))


def test_hash_collision_does_not_merge_two_decks(tmp_path, monkeypatch):
    """A truncated-hash collision must not fuse two decks' provenance.

    48 bits makes this vanishingly unlikely in practice, but the consequence would
    be silent corruption, so the hash match is confirmed against the CSV on disk.
    """
    monkeypatch.setattr("scraper.writer.deck_hash", lambda _ids: "collide00cafe")

    writer = DeckWriter(str(tmp_path))
    first = writer.write(make_deck(sixty(0), archetype="Deck A"))
    second = writer.write(make_deck(sixty(1), archetype="Deck B"))

    assert first.new_deck
    assert second.new_deck, "different decks sharing a hash must not be merged"
    assert first.slug != second.slug
    assert len(writer.manifest.decks) == 2

    # ... and each still finds *itself*, not the other, on a later occurrence.
    assert writer.find_slug(sixty(0)) == first.slug
    assert writer.find_slug(sixty(1)) == second.slug


def test_manifest_round_trips_through_json():
    """Every field survives a save/load cycle."""
    entry = DeckEntry(file="a/a.csv", id_hash="abc123", archetype="A", fmt="standard")
    entry.add_observation(
        Observation(
            source="limitless",
            archetype="A",
            fmt="standard",
            event="E",
            event_date="2026-02-02",
            scraped_date="2026-02-03",
            record="6-1-1",
            placing=4,
            url="https://example.test/e",
            external_ids={"tournament_id": "t"},
            substitutions=[
                {
                    "source_name": "Charmeleon",
                    "source_set": "OBF",
                    "source_number": "27",
                    "count": 2,
                    "target_id": 927,
                    "target_name": "Charmeleon",
                    "kind": "variant",
                    "confidence": 0.9,
                    "rationale": "same-name gameplay profile",
                }
            ],
            merged_from="a-2",
        )
    )
    restored = Manifest.from_json(Manifest(decks={"a": entry}).to_json())
    assert restored.decks["a"] == entry


def test_v2_observations_load_without_substitution_provenance():
    restored = Manifest.from_json(
        {
            "schema_version": 2,
            "decks": {
                "a": {
                    "file": "a.csv",
                    "id_hash": "abc",
                    "archetype": "A",
                    "observations": [{"source": "limitless"}],
                }
            },
        }
    )

    assert restored.read_version == 2
    assert restored.decks["a"].observations[0].substitutions == []


def test_rescrape_backfills_missing_substitution_provenance(tmp_path):
    resolved = make_deck(sixty(), external_ids={"event": "1"})
    writer = DeckWriter(str(tmp_path))
    writer.write(resolved)
    resolved.swaps.append(
        CardSwap(
            source_name="Charmeleon",
            source_set="OBF",
            source_number="27",
            count=2,
            target_id=927,
            target_name="Charmeleon",
            kind="mapping",
            confidence=None,
            rationale="reviewed mapping fixture",
            mapping_confidence=4,
            rule_id="reviewed-rule-1",
            family_id="family-1",
            source_stage="Stage 1 Pokémon",
            source_previous_stage="Charmander",
        )
    )

    result = writer.write(resolved)

    assert result.new_observation is False
    observation = next(iter(writer.manifest.decks.values())).observations[0]
    assert observation.substitutions[0]["target_id"] == 927
    assert observation.substitutions[0]["rule_id"] == "reviewed-rule-1"
    assert observation.substitutions[0]["family_id"] == "family-1"
    assert observation.substitutions[0]["source_stage"] == "Stage 1 Pokémon"
    assert observation.substitutions[0]["source_previous_stage"] == "Charmander"
    assert observation.substitutions[0]["mapping_confidence"] == 4
    assert "confidence" not in observation.substitutions[0]


def test_missing_manifest_is_an_empty_one(tmp_path):
    """A corpus with no manifest yet is not an error."""
    assert manifest_mod.load(tmp_path).decks == {}


# --------------------------------------------------------------------------- #
# End to end: the scrape pipeline, and pruning
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def card_index() -> CardIndex:
    """
    :return: The real card index, loaded once for the module.
    """
    return CardIndex()


def raw_from_ids(ids: list[int], index: CardIndex, **kwargs) -> RawDeck:
    """
    Turn card IDs back into a scraped-looking decklist of names and counts.

    Going through names means the deck travels the same resolve -> validate ->
    write path a real scrape does.

    :param ids: Card IDs, one entry per copy.
    :param index: Card index used to look each ID's printed name up.
    :param kwargs: Provenance overrides for the RawDeck.
    :return: The :class:`~scraper.models.RawDeck`.
    """
    counts = Counter(ids)
    cards = [
        RawCard(
            count=n,
            name=index.by_id[cid].name,
            set_code=index.by_id[cid].set_code,
            number=index.by_id[cid].number,
        )
        for cid, n in counts.items()
    ]
    fields = {"source": "test", "archetype": "Example Deck", "cards": cards} | kwargs
    return RawDeck(**fields)


def test_pipeline_records_a_second_occurrence_instead_of_dropping_it(tmp_path, card_index):
    """The full pipeline counts a repeat list as an occurrence, not a discard.

    The dedup check used to short-circuit ahead of the writer, so a duplicate deck
    never reached the code that would record it.
    """
    ids = load_deck(str(EXAMPLE_DECK))
    writer = DeckWriter(str(tmp_path))
    summary = RunSummary()

    first = raw_from_ids(ids, card_index, event="Event A", external_ids={"t": "1"})
    process_deck(first, card_index, writer, summary, date="2026-07-01")
    assert summary.written == 1, summary.drops

    # The same 60 cards, listed in a different order, from a different event.
    shuffled = ids[:]
    random.Random(3).shuffle(shuffled)
    second = raw_from_ids(shuffled, card_index, event="Event B", external_ids={"t": "2"})
    process_deck(second, card_index, writer, summary, date="2026-07-02")

    assert summary.written == 1  # no second file
    assert summary.reobserved == 1
    assert len(list(tmp_path.rglob("*.csv"))) == 1

    entry = next(iter(manifest_mod.load(tmp_path).decks.values()))
    assert entry.observation_count == 2
    assert [o.event for o in entry.observations] == ["Event A", "Event B"]


def test_pipeline_is_idempotent_across_runs(tmp_path, card_index):
    """Scraping the same source twice leaves the counts where they were."""
    ids = load_deck(str(EXAMPLE_DECK))
    raw = raw_from_ids(ids, card_index, event="Event A", external_ids={"t": "1"})

    writer = DeckWriter(str(tmp_path))
    process_deck(raw, card_index, writer, RunSummary(), date="2026-07-01")

    rerun = RunSummary()
    process_deck(raw, card_index, DeckWriter(str(tmp_path)), rerun, date="2026-07-29")
    assert rerun.already_recorded == 1
    assert rerun.written == 0
    assert rerun.reobserved == 0

    entry = next(iter(manifest_mod.load(tmp_path).decks.values()))
    assert entry.observation_count == 1


def test_pruning_folds_observations_into_the_survivor(tmp_path):
    """Collapsing near-duplicates must not delete the occurrences they carried.

    Pruning exists to stop near-identical lists oversampling an archetype; it
    should not also erase the record of how many players brought them.
    """
    base = sixty()
    variant = base[:-1] + [base[0]]  # one copy swapped: a near-duplicate, not equal

    writer = DeckWriter(str(tmp_path))
    kept = writer.write(
        make_deck(base, archetype="Gardevoir ex", **{k: v for k, v in occurrence().items() if k != "archetype"})
    )
    dropped = writer.write(
        make_deck(
            variant,
            **occurrence(
                archetype="Gardevoir ex",
                event="Toronto Regional",
                external_ids={"tournament_id": "xyz", "placing": "2"},
            ),
        )
    )
    assert kept.new_deck and dropped.new_deck
    before = sum(e.observation_count for e in writer.manifest.decks.values())
    assert before == 2

    prune_near_duplicates(tmp_path, threshold=0.9)

    after = manifest_mod.load(tmp_path)
    assert len(after.decks) == 1, "the near-duplicate cluster should collapse to one deck"
    survivor = next(iter(after.decks.values()))
    assert survivor.observation_count == 2, "the pruned deck's occurrence must survive"
    merged = [o for o in survivor.observations if o.merged_from]
    assert len(merged) == 1
    assert merged[0].merged_from in {kept.slug, dropped.slug}


def test_unique_deck_count_uses_the_same_identity_as_dedup(tmp_path):
    """The corpus's "unique decks" tally must agree with the dedup contract.

    Counting distinct *sets* of card IDs would be copy-count blind, reporting two
    decks that differ only in copy counts — two separate manifest entries — as one.
    """
    base = sixty()
    reordered = list(reversed(base))
    redistributed = base[:-1] + [base[0]]  # same cards, different copy split

    assert count_unique_decks([base, reordered]) == 1
    assert count_unique_decks([base, redistributed]) == 2
    assert count_unique_decks([base, reordered, redistributed]) == 2

    # And it matches what the writer actually does with the same three decks.
    writer = DeckWriter(str(tmp_path))
    for ids in (base, reordered, redistributed):
        writer.write(make_deck(ids, archetype="Deck"))
    assert len(writer.manifest.decks) == count_unique_decks([base, reordered, redistributed])


def test_pruning_keeps_the_schema_envelope(tmp_path):
    """Pruning rewrites the manifest through the schema, not as a bare mapping."""
    writer = DeckWriter(str(tmp_path))
    base = sixty()
    writer.write(make_deck(base, archetype="Deck"))
    writer.write(make_deck(base[:-1] + [base[0]], archetype="Deck"))

    prune_near_duplicates(tmp_path, threshold=0.9)

    raw = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert raw["schema_version"] == manifest_mod.SCHEMA_VERSION
    assert raw["hash_algo"] == manifest_mod.HASH_ALGO
    assert "decks" in raw
