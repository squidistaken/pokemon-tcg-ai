"""Metagame metrics: archetype cores, diversity, win-rates, card usage/correlation."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from src.env.observation.card_database import CardDatabase

    from ..manifest import Manifest


def archetype_core_cards(
    decks: list[list[int]],
    names: list[str],  # noqa: ARG001
    archetypes: list[str],
    db: CardDatabase | None = None,
    threshold: float = 0.6,
) -> dict:
    """
    Identify the "core" cards and count the "tech"/variable slots per archetype.

    For each archetype, a card is core if it appears (>=1 copy) in more than
    ``threshold`` of that archetype's deck lists; it is a tech/variable slot
    if it appears in at least one list but in ``threshold`` or fewer of them.

    :param decks: List of decks, each a list of card IDs (duplicates = copies).
    :param names: Deck file stems, in the same order as ``decks``.
    :param archetypes: Archetype label per deck, in the same order as ``decks``.
    :param db: Optional ``CardDatabase`` for readable card names in the output.
    :param threshold: Inclusion fraction above which a card counts as core.
    :return: Mapping ``archetype -> {"n_decks", "core_cards", "n_core",
             "n_tech"}`` where ``core_cards`` is a list of
             ``(card_id, name, inclusion_fraction)`` sorted by inclusion desc.
    """
    by_arch: dict[str, list[int]] = defaultdict(list)
    for i, arch in enumerate(archetypes):
        by_arch[arch].append(i)

    result: dict = {}
    for arch, idxs in by_arch.items():
        n = len(idxs)
        presence_counts: Counter = Counter()
        for i in idxs:
            for cid in set(decks[i]):
                presence_counts[cid] += 1

        core = []
        n_tech = 0
        for cid, cnt in presence_counts.items():
            frac = cnt / n
            if frac > threshold:
                name = db.card_name(cid) if db is not None else str(cid)
                core.append((cid, name, frac))
            else:
                n_tech += 1
        core.sort(key=lambda t: t[2], reverse=True)
        result[arch] = {
            "n_decks": n,
            "core_cards": core,
            "n_core": len(core),
            "n_tech": n_tech,
        }
    return result


def metagame_diversity(archetypes: list[str], decks: list[list[int]]) -> dict:
    """
    Summarize how concentrated or diverse the metagame is.

    :param archetypes: Archetype label per deck.
    :param decks: List of decks, each a list of card IDs (for the distinct-card
                  count across the whole corpus).
    :return: Dict with ``n_archetypes``, ``top_archetype``, ``top_share``,
             ``shannon_entropy`` (nats), ``effective_archetypes``
             (``exp(entropy)``) and ``distinct_cards``.
    """
    counts = Counter(archetypes)
    total = sum(counts.values())
    freqs = np.array([c / total for c in counts.values()], dtype=np.float64)
    entropy = float(-np.sum(freqs * np.log(freqs)))  # nats; 0*log0 := 0
    top_arch, top_count = counts.most_common(1)[0]
    distinct_cards = len({cid for deck in decks for cid in deck})
    return {
        "n_archetypes": len(counts),
        "top_archetype": top_arch,
        "top_share": top_count / total,
        "shannon_entropy": entropy,
        "effective_archetypes": float(np.exp(entropy)),
        "distinct_cards": distinct_cards,
    }


def card_pool_coverage(decks: list[list[int]], db: CardDatabase) -> dict:
    """
    Measure how much of the engine's card pool the corpus actually uses.

    :param decks: List of decks, each a list of card IDs.
    :param db: Loaded :class:`CardDatabase` providing the exists flag.
    :return: Dict with ``distinct_used``, ``total_available``, ``n_unused`` and
             ``coverage`` (used / available, in ``[0, 1]``).
    """
    distinct_used = len({cid for deck in decks for cid in deck})
    total_available = int(db.card_features.numpy()[:, 11].sum())
    n_unused = total_available - distinct_used
    coverage = distinct_used / total_available if total_available else 0.0
    return {
        "distinct_used": distinct_used,
        "total_available": total_available,
        "n_unused": n_unused,
        "coverage": coverage,
    }


def parse_winrate(record: str | None) -> float | None:
    """
    Parse a manifest ``record`` string ("W-L-T") into a win-rate.

    :param record: Record string such as ``"4-1-0"``; ties are ignored.
    :return: ``W / (W + L)``, or None if the record is missing, malformed, or
             has no decisive games (``W + L == 0``).
    """
    if not record:
        return None
    parts = record.split("-")
    try:
        w, l = int(parts[0]), int(parts[1])
    except (ValueError, IndexError):
        return None
    if w + l == 0:  # no decisive games
        return None
    return w / (w + l)


def deck_winrates(manifest: Manifest, name: str) -> list[float]:
    """
    Every recorded win-rate for one deck — one per occurrence, not one per file.

    A list fourteen players piloted has fourteen records, and each is a real
    result; collapsing them to one would throw away thirteen samples and flatten
    the popularity weighting that ``observation_count`` exists to capture.

    :param manifest: The corpus manifest.
    :param name: Deck file stem.
    :return: Win-rate per observation that reports a usable record (possibly empty).
    """
    entry = manifest.decks.get(name)
    if entry is None:
        return []
    return [
        wr for o in entry.observations if (wr := parse_winrate(o.record)) is not None
    ]


def archetype_winrates(
    names: list[str],
    archetypes: list[str],
    manifest: Manifest,
    min_decks: int = 3,
) -> dict:
    """
    Aggregate deck win-rates by archetype to compare archetype quality.

    Each *occurrence* contributes a sample, so a widely-played list weighs more
    than a one-off brew with the same record.

    :param names: Deck file stems, aligned with ``archetypes``.
    :param archetypes: Archetype label per deck.
    :param manifest: The corpus manifest (see load_manifest).
    :param min_decks: Minimum observations with a valid record for an archetype to
                      be ranked; archetypes below this are dropped (and counted).
    :return: Dict with ``rows`` (list of ``(archetype, n_observations,
             pooled_winrate, mean_winrate, std_winrate)`` sorted by pooled win-rate
             desc), ``n_filtered`` (archetypes dropped by ``min_decks``) and
             ``min_decks``.
    """
    wins: dict[str, int] = defaultdict(int)
    losses: dict[str, int] = defaultdict(int)
    per_deck: dict[str, list[float]] = defaultdict(list)
    for name, arch in zip(names, archetypes, strict=False):
        entry = manifest.decks.get(name)
        if entry is None:
            continue
        for obs in entry.observations:
            wr = parse_winrate(obs.record)
            if wr is None:
                continue
            w, l = (
                int(p) for p in obs.record.split("-")[:2]
            )  # parse_winrate validated it
            wins[arch] += w
            losses[arch] += l
            per_deck[arch].append(wr)

    rows = []
    n_filtered = 0
    for arch, wrs in per_deck.items():
        if len(wrs) < min_decks:
            n_filtered += 1
            continue
        total_games = wins[arch] + losses[arch]
        pooled = wins[arch] / total_games if total_games else 0.0
        arr = np.array(wrs, dtype=np.float64)
        rows.append((arch, len(wrs), pooled, float(arr.mean()), float(arr.std())))
    rows.sort(key=lambda t: t[2], reverse=True)
    return {"rows": rows, "n_filtered": n_filtered, "min_decks": min_decks}


def deck_popularity_ranking(
    names: list[str],
    archetypes: list[str] | None,
    manifest: Manifest,
    top: int = 20,
) -> list:
    """
    Rank individual deck files by how often they were independently observed.

    ``observation_count`` is the corpus's popularity signal (see
    :mod:`scraper.manifest`): a list fourteen players brought to fourteen events
    outranks a one-off brew with the same 60 cards. This surfaces that ranking
    directly, at deck-file granularity rather than pooled by archetype.

    :param names: Deck file stems, in matrix order.
    :param archetypes: Archetype label per deck (matrix order), or None if no
                        manifest-derived labels are available.
    :param manifest: The corpus manifest (see load_manifest).
    :param top: Number of most-observed decks to return.
    :return: List of ``(name, archetype, observation_count, pooled_winrate)``
             sorted by ``observation_count`` desc, length <= ``top``.
             ``pooled_winrate`` is None if no observation reports a usable record.
             ``archetype`` is None if ``archetypes`` is None.
    """
    rows = []
    for i, name in enumerate(names):
        entry = manifest.decks.get(name)
        if entry is None:
            continue
        wins = losses = 0
        for obs in entry.observations:
            wr = parse_winrate(obs.record)
            if wr is None:
                continue
            w, l = (
                int(p) for p in obs.record.split("-")[:2]
            )  # parse_winrate validated it
            wins += w
            losses += l
        total_games = wins + losses
        pooled = wins / total_games if total_games else None
        arch = archetypes[i] if archetypes is not None else None
        rows.append((name, arch, entry.observation_count, pooled))
    rows.sort(key=lambda r: r[2], reverse=True)
    return rows[:top]


def card_usage_rates(
    decks: list[list[int]],
    names: list[str],  # noqa: ARG001 - kept for call-site parity with the other metric fns
    db: CardDatabase | None = None,
    top: int = 20,
) -> list:
    """
    Rank the most-included cards across the whole corpus.

    For each card: the fraction of decks running >=1 copy ("inclusion rate")
    and, among only those decks that run it, the average number of copies.

    :param decks: List of decks, each a list of card IDs (duplicates = copies).
    :param names: Deck file stems (only used for the corpus size / API parity).
    :param db: Optional ``CardDatabase`` for readable card names.
    :param top: Number of top cards to return, ranked by inclusion rate.
    :return: List of ``(card_id, name, inclusion_rate, avg_copies_when_run)``
             sorted by inclusion rate desc, length <= ``top``.
    """
    n_decks = len(decks)
    deck_count: Counter = Counter()  # decks running the card at all
    copy_total: Counter = Counter()  # total copies across those decks
    for deck in decks:
        card_copies = Counter(deck)
        for cid, copies in card_copies.items():
            deck_count[cid] += 1
            copy_total[cid] += copies

    rows = []
    for cid, dc in deck_count.items():
        name = db.card_name(cid) if db is not None else str(cid)
        rows.append((cid, name, dc / n_decks, copy_total[cid] / dc))
    rows.sort(key=lambda r: r[2], reverse=True)
    return rows[:top]


def card_placement_correlation(
    decks: list[list[int]],
    names: list[str],
    manifest: Manifest,
    db: CardDatabase | None = None,
    min_decks: int = 10,
    top: int = 15,
) -> dict:
    """
    Point-biserial correlation between running a card and deck win-rate.

    Each occurrence of a deck is one observation of that card set's win-rate, so a
    deck seen five times contributes five rows — which is what weights the
    correlation by how much the list was actually played.

    :param decks: List of decks, each a list of card IDs.
    :param names: Deck file stems, aligned with ``decks``; used to look up the
                  records in ``manifest``.
    :param manifest: The corpus manifest (see load_manifest).
    :param db: Optional ``CardDatabase`` for readable card names.
    :param min_decks: Minimum number of observations a card must appear in to qualify.
    :param top: How many top positive and top negative cards to return.
    :return: Dict with ``n_decks_scored``, ``n_cards_tested`` and
             ``positive`` / ``negative`` lists of
             ``(card_id, name, r, n_decks_with_card)``.
    """
    winrates = []
    kept_decks = []
    for name, deck in zip(names, decks, strict=False):
        for wr in deck_winrates(manifest, name):
            winrates.append(wr)
            kept_decks.append(deck)

    n = len(kept_decks)
    if not n:  # no source reported a record; averaging nothing yields NaNs
        return {
            "n_decks_scored": 0,
            "n_cards_tested": 0,
            "positive": [],
            "negative": [],
        }

    y = np.array(winrates, dtype=np.float64)
    y_centered = y - y.mean()
    y_ss = float(np.sum(y_centered**2))

    card_deckcount: Counter = Counter()
    for deck in kept_decks:
        for cid in set(deck):
            card_deckcount[cid] += 1

    rows = []
    for cid, cnt in card_deckcount.items():
        if cnt < min_decks or cnt == n:  # need variation in x too
            continue
        x = np.array([1.0 if cid in deck else 0.0 for deck in kept_decks])
        x_centered = x - x.mean()
        denom = np.sqrt(float(np.sum(x_centered**2)) * y_ss)
        if denom == 0:
            continue
        r = float(np.sum(x_centered * y_centered) / denom)
        name = db.card_name(cid) if db is not None else str(cid)
        rows.append((cid, name, r, cnt))

    rows.sort(key=lambda t: t[2], reverse=True)
    return {
        "n_decks_scored": n,
        "n_cards_tested": len(rows),
        "positive": rows[:top],
        "negative": rows[-top:][::-1],
    }
