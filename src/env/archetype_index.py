from collections.abc import Sequence
from pathlib import Path


class ArchetypeIndex:
    """
    Groups the deck corpus into archetypes and numbers the resulting matchups.

    The curriculum scores *archetype pairs* rather than individual deck lists.
    Two lists of one archetype play nearly identically, so per-list scores would
    mostly measure noise between near-duplicates, and the corpus is far too
    large to give every list pair enough episodes to score under a terminal
    win/loss reward. Archetypes reduce that to a few thousand matchups, and
    keep the count roughly fixed as the corpus grows, since new lists are
    overwhelmingly variants of archetypes that already exist.

    An archetype is the folder its decks live in: the corpus ships as
    ``decks/<archetype>/*.csv`` and
    :func:`~src.env.deck.resolve_deck_paths` already prefers those nested
    files, so the grouping needs no clustering step. Decks sitting loose in the
    pool root fall back to :data:`UNGROUPED`.

    A matchup is identified by a single ``pair_id``, ``agent * count +
    opponent``, which stays valid for the lifetime of the run regardless of how
    the level buffer reorders or evicts entries internally.
    """

    #: Archetype name used for decks that are not inside a per-archetype folder.
    UNGROUPED = "ungrouped"

    def __init__(
        self, names: Sequence[str], deck_indices: Sequence[Sequence[int]]
    ) -> None:
        """
        :param names: Archetype name per archetype, in archetype-index order.
        :param deck_indices: Positions in the deck pool belonging to each
            archetype, aligned with ``names``.
        :raises ValueError: If the two sequences disagree in length, if there
            are no archetypes, or if any archetype holds no decks.
        """
        if len(names) != len(deck_indices):
            raise ValueError(
                f"names ({len(names)}) and deck_indices ({len(deck_indices)}) must align"
            )
        if not names:
            raise ValueError("ArchetypeIndex requires at least one archetype")
        for name, indices in zip(names, deck_indices, strict=True):
            if not indices:
                raise ValueError(f"archetype {name!r} holds no decks")
        self._names = tuple(names)
        self._deck_indices = tuple(tuple(indices) for indices in deck_indices)

    @classmethod
    def from_paths(cls, paths: Sequence[str]) -> "ArchetypeIndex":
        """
        Group deck paths by the folder each one sits in.

        Archetypes are ordered by name so the numbering is reproducible across
        processes and runs given the same corpus.

        :param paths: Deck CSV paths, in the order they occupy in the pool.
        :return: Index over the archetypes those paths fall into.
        :raises ValueError: If ``paths`` is empty.
        """
        if not paths:
            raise ValueError("ArchetypeIndex.from_paths requires at least one path")
        grouped: dict[str, list[int]] = {}
        for position, path in enumerate(paths):
            grouped.setdefault(cls._archetype_of(path), []).append(position)
        names = sorted(grouped)
        return cls(names, [grouped[name] for name in names])

    @staticmethod
    def _archetype_of(path: str) -> str:
        """
        Archetype a deck path belongs to.

        :param path: Deck CSV path.
        :return: The parent folder's name, or :data:`UNGROUPED` for a deck that
            has no meaningful parent (a bare filename).
        """
        parent = Path(path).parent.name
        return parent or ArchetypeIndex.UNGROUPED

    @property
    def count(self) -> int:
        """
        Number of archetypes, written ``K`` in the design notes.

        :return: The archetype count.
        """
        return len(self._names)

    @property
    def names(self) -> tuple[str, ...]:
        """
        Archetype names, in archetype-index order.

        :return: The names as an immutable view.
        """
        return self._names

    @property
    def pair_count(self) -> int:
        """
        Number of distinct matchups, ``count ** 2``.

        :return: The size of the level space.
        """
        return self.count * self.count

    def decks_for(self, archetype: int) -> tuple[int, ...]:
        """
        Deck-pool positions belonging to one archetype.

        :param archetype: Archetype index.
        :return: The deck positions, as an immutable view.
        """
        return self._deck_indices[archetype]

    def pair_id(self, agent: int, opponent: int) -> int:
        """
        Stable identifier of one matchup.

        :param agent: Archetype index played by the agent's seat.
        :param opponent: Archetype index played by the opposing seat.
        :return: The matchup's identifier.
        :raises IndexError: If either archetype index is out of range.
        """
        if not 0 <= agent < self.count or not 0 <= opponent < self.count:
            raise IndexError(
                f"archetype indices must lie in [0, {self.count}), "
                f"got ({agent}, {opponent})"
            )
        return agent * self.count + opponent

    def unpair(self, pair_id: int) -> tuple[int, int]:
        """
        Split a matchup identifier back into its two archetypes.

        :param pair_id: Identifier produced by :meth:`pair_id`.
        :return: ``(agent archetype, opponent archetype)``.
        :raises IndexError: If ``pair_id`` is out of range.
        """
        if not 0 <= pair_id < self.pair_count:
            raise IndexError(
                f"pair_id must lie in [0, {self.pair_count}), got {pair_id}"
            )
        return divmod(pair_id, self.count)

    def name_of_pair(self, pair_id: int) -> str:
        """
        Human-readable matchup label, for logging and inspection.

        :param pair_id: Identifier produced by :meth:`pair_id`.
        :return: ``"<agent archetype> vs <opponent archetype>"``.
        """
        agent, opponent = self.unpair(pair_id)
        return f"{self._names[agent]} vs {self._names[opponent]}"
