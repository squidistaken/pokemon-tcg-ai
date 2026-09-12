"""
Build a synthetic, type-based deck corpus for curriculum experiments.

The tournament corpus is a pulled artifact (``scripts/fetch_decks.sh``) and is
not always available -- on a machine without ``gh`` credentials, or in CI.
This script fabricates a stand-in from the bundled card data so the curriculum
can be exercised end to end.

The decks are **not** meant to resemble the competitive meta. They exist to
give the level space real structure: each archetype is built around one energy
type, and the engine's weakness rule roughly doubles damage from the type a
Pokemon is weak to. That makes matchups genuinely asymmetric and partly
non-transitive, which is the property a matchup curriculum is supposed to
exploit. A corpus of near-identical decks would leave it nothing to curate and
would make any A/B comparison meaningless.

Each deck is 20 Pokemon (four copies each of five different Basic attackers of
one type) plus 40 basic Energy of the matching type, which satisfies every rule
the engine enforces at ``BattleStart``: 60 cards, at least one Basic Pokemon,
at most four of any non-Energy name, and at most one ACE SPEC.

Usage::

    python -m scripts.make_synthetic_corpus --out /tmp/corpus --variants 3
"""

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path

CARD_DATA = Path(__file__).parents[1] / "scraper" / "EN_Card_Data.csv"
STAGE_COLUMN = "Stage (Pokémon)/Type (Energy and Trainer)"

#: Energy type -> (readable archetype name, basic Energy card ID).
TYPES: dict[str, tuple[str, int]] = {
    "{G}": ("grass", 1),
    "{R}": ("fire", 2),
    "{W}": ("water", 3),
    "{L}": ("lightning", 4),
    "{P}": ("psychic", 5),
    "{F}": ("fighting", 6),
    "{D}": ("darkness", 7),
    "{M}": ("metal", 8),
}

POKEMON_PER_DECK = 5
COPIES_PER_POKEMON = 4
ENERGY_PER_DECK = 40


def load_attackers() -> dict[str, list[tuple[int, str]]]:
    """
    Collect Basic Pokemon usable as attackers, grouped by energy type.

    Cards carrying a ``Rule`` are excluded: those are the ex/Tera/ACE SPEC
    style cards whose extra-prize and deck-building restrictions would make the
    fabricated decks harder to keep legal for no benefit here.

    Entries are deduplicated **by name**, keeping the lowest card ID. The
    four-copy limit the engine enforces is per name, and the same Pokemon is
    reprinted across sets under different IDs, so five distinct IDs can still
    be an illegal deck.

    :return: Mapping from energy type to ``(card id, card name)`` pairs, one
        per distinct name.
    """
    by_name: dict[str, dict[str, tuple[int, str]]] = defaultdict(dict)
    with CARD_DATA.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row[STAGE_COLUMN] != "Basic Pokémon":
                continue
            if row.get("Rule", "n/a") not in ("", "n/a"):
                continue
            if row["Type"] not in TYPES:
                continue
            name = row["Card Name"]
            card_id = int(row["Card ID"])
            existing = by_name[row["Type"]].get(name)
            if existing is None or card_id < existing[0]:
                by_name[row["Type"]][name] = (card_id, name)
    return {
        energy_type: sorted(entries.values())
        for energy_type, entries in by_name.items()
    }


def build_deck(attackers: list[tuple[int, str]], energy_id: int) -> list[int]:
    """
    Assemble one legal 60-card deck from a set of attackers.

    :param attackers: ``(card id, name)`` pairs to build around; the first
        :data:`POKEMON_PER_DECK` are used.
    :param energy_id: Basic Energy card ID matching the attackers' type.
    :return: 60 card IDs.
    :raises ValueError: If too few attackers were supplied.
    """
    if len(attackers) < POKEMON_PER_DECK:
        raise ValueError(
            f"need {POKEMON_PER_DECK} distinct attackers, got {len(attackers)}"
        )
    deck: list[int] = []
    for card_id, _name in attackers[:POKEMON_PER_DECK]:
        deck.extend([card_id] * COPIES_PER_POKEMON)
    deck.extend([energy_id] * ENERGY_PER_DECK)
    return deck


def write_corpus(out_dir: Path, variants: int, seed: int) -> list[Path]:
    """
    Write one archetype folder per energy type, each holding several lists.

    Variants within an archetype share the type and therefore the strategy, and
    differ only in which attackers they use -- the same relationship real deck
    lists of one archetype have, and what makes archetype-level scoring the
    right granularity.

    :param out_dir: Corpus root; ``<out_dir>/<archetype>/list<N>.csv``.
    :param variants: Deck lists per archetype.
    :param seed: Seed for the attacker draw.
    :return: The written paths.
    """
    rng = random.Random(seed)
    grouped = load_attackers()
    written: list[Path] = []
    for energy_type, (name, energy_id) in TYPES.items():
        pool = sorted(grouped.get(energy_type, []))
        if len(pool) < POKEMON_PER_DECK:
            continue
        folder = out_dir / name
        folder.mkdir(parents=True, exist_ok=True)
        for variant in range(variants):
            attackers = rng.sample(pool, POKEMON_PER_DECK)
            deck = build_deck(attackers, energy_id)
            path = folder / f"list{variant}.csv"
            path.write_text("\n".join(str(card) for card in deck) + "\n")
            written.append(path)
    return written


def main() -> None:
    """
    Build the corpus and verify every deck starts a battle in the engine.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="corpus root")
    parser.add_argument("--variants", type=int, default=3, help="lists per archetype")
    parser.add_argument("--seed", type=int, default=0, help="attacker-draw seed")
    args = parser.parse_args()

    written = write_corpus(args.out, args.variants, args.seed)
    print(f"wrote {len(written)} decks under {args.out}")

    # The engine is the authority on legality, so prove each deck starts rather
    # than trusting the construction rules restated above.
    from src.env.battle_handle import BattleHandle
    from src.env.decks.deck import load_deck

    for path in written:
        deck = load_deck(str(path))
        handle = BattleHandle()
        handle.start(deck, deck)
        handle.finish()
    print(f"verified {len(written)} decks start in the engine")


if __name__ == "__main__":
    main()
