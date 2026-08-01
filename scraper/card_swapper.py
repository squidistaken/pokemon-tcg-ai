from __future__ import annotations

from .card_index import CardIndex, normalize_name

# Missing staple -> its closest in-pool substitute. EN_Card_Data.csv is missing
# several whole sets and is drastically under-populated even in sets it
# nominally supports (see scraper/README.md), so ordinary decks fail almost
# universally on a handful of recurring staples. Substitutes range from
# near-identical effects to "spirit of the card" analogs when nothing closer
# exists in the pool; hand-curated, not derived automatically.
DEFAULT_SWAP_MAP: dict[str, str] = {
    "Pal Pad": "Miracle Headset",  # near-identical: return 2 Supporters from discard to hand
    "Artazon": "Lumiose City",  # near-identical: search a Basic Pokémon onto the Bench
    "Nest Ball": "Buddy-Buddy Poffin",  # close: search Basic Pokémon (<=70 HP) onto the Bench
    "Super Rod": "Max Rod",  # loose: recovers Pokémon+Energy from discard, to hand not deck
    "Superior Energy Retrieval": "Energy Retrieval",  # loose: weaker version of the same role
    "Iono": "Judge",  # loose: fixed draw-4 instead of prizes-remaining
    "Professor's Research": "Naveen",  # loose: conditional draw-to-5 instead of unconditional draw-7
}


class CardSwapper:
    """Hardcoded substitute table for staples missing from the card pool.

    Every substitute is resolved against ``index`` up front, so a stale or
    misspelled mapping entry fails loudly at construction time rather than
    silently never firing during a scrape.
    """

    def __init__(self, index: CardIndex, mapping: dict[str, str] | None = None):
        """
        :param index: Card index the substitutes must already resolve against.
        :param mapping: Missing-card-name -> substitute-name table; defaults to
            :data:`DEFAULT_SWAP_MAP`.
        :raises ValueError: If a configured substitute isn't in ``index``.
        """
        self._swap: dict[str, tuple[str, int]] = {}
        for src_name, dst_name in (DEFAULT_SWAP_MAP if mapping is None else mapping).items():
            dst_id = index.resolve_id(dst_name)
            if dst_id is None:
                raise ValueError(f"card swap target {dst_name!r} not found in card pool")
            self._swap[normalize_name(src_name)] = (dst_name, dst_id)

    def lookup(self, name: str) -> tuple[str, int] | None:
        """
        :param name: Scraped card name that failed normal resolution.
        :return: ``(substitute name, substitute Card ID)``, or None if this
            card has no configured substitute.
        """
        return self._swap.get(normalize_name(name))
