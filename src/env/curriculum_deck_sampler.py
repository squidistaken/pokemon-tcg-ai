import random
from collections.abc import Sequence

import torch

from .archetype_index import ArchetypeIndex
from .curriculum_handles import CurriculumHandles

Deck = list[int]

#: Level identifier reported when no matchup has been drawn yet.
NO_LEVEL = -1


class CurriculumDeckSampler:
    """
    Deck sampler that draws each episode's matchup from the level curriculum.

    Implements the :class:`~src.env.deck_sampler.DeckSampler` protocol, so it
    drops into :class:`~src.env.tcg_env.TCGEnv` wherever
    :class:`~src.env.deck_sampler.PoolDeckSampler` would go. The difference is
    where the matchup comes from: instead of drawing two decks uniformly, it
    draws an *archetype pair* from the distribution the learner publishes
    through :class:`~src.env.curriculum_handles.CurriculumHandles`, then deals
    a concrete list uniformly from within each archetype. List-level diversity
    is preserved; only the archetype pairing is curated.

    The identifier of the drawn matchup is exposed as :attr:`level_id` so the
    environment can stamp it into every observation of the episode, which is
    what lets the learner attribute critic residuals back to the matchup that
    produced them.

    Before the learner has published anything the channel is empty, and the
    sampler falls back to a uniformly random archetype pair. That is the
    correct behaviour rather than an error: collection starts before the first
    update, so the first batch is necessarily uncurated.

    ``explore_prob`` extends that same fallback into an ongoing mechanism: with
    that probability, every episode (not just ones before the first publish)
    draws a uniformly random pair instead of one from the published
    distribution. This is what lets the learner-side buffer discover matchups
    lazily (:meth:`~src.env.level_buffer.LevelBuffer.commit`) when the corpus
    is larger than the buffer's capacity -- without it, only whatever handful
    of pairs happened to be drawn before the very first publish would ever be
    scored, and the buffer would never grow past that.
    """

    def __init__(
            self,
            decks: Sequence[Sequence[int]],
            archetypes: ArchetypeIndex,
            handles: CurriculumHandles,
            seed: int | None = None,
            explore_prob: float = 0.0,
    ) -> None:
        """
        :param decks: The deck pool, indexed by the positions ``archetypes``
            was built from.
        :param archetypes: Grouping of ``decks`` into archetypes.
        :param handles: Shared channel the learner publishes the distribution to.
        :param seed: Seed for the within-archetype list draw and the fallback.
        :param explore_prob: Probability of drawing a fresh, uniformly random
            archetype pair instead of one from the published distribution, so
            unseen matchups keep being discovered even after publishing starts.
            0 (default) never explores, matching the pre-existing behaviour.
        :raises ValueError: If the pool is empty, an archetype references a
            deck position the pool does not contain, or ``explore_prob`` is out
            of ``[0, 1]``.
        """
        if not decks:
            raise ValueError("CurriculumDeckSampler requires a non-empty deck pool")
        for archetype in range(archetypes.count):
            for position in archetypes.decks_for(archetype):
                if position >= len(decks):
                    raise ValueError(
                        f"archetype {archetypes.names[archetype]!r} references deck "
                        f"position {position}, but the pool holds {len(decks)} decks"
                    )
        if not 0.0 <= explore_prob <= 1.0:
            raise ValueError(f"explore_prob must be in [0, 1], got {explore_prob}")
        self._decks = [list(deck) for deck in decks]
        self._archetypes = archetypes
        self._handles = handles
        self._rng = random.Random(seed)
        self._explore_prob = explore_prob
        self._level_id = NO_LEVEL

    @property
    def level_id(self) -> int:
        """
        Identifier of the matchup drawn by the most recent :meth:`sample`.

        :return: The matchup identifier, or :data:`NO_LEVEL` before the first draw.
        """
        return self._level_id

    @property
    def archetypes(self) -> ArchetypeIndex:
        """
        The archetype grouping this sampler draws from.

        :return: The index it was constructed with.
        """
        return self._archetypes

    def sample(self) -> tuple[Deck, Deck]:
        """
        Draw the next episode's matchup and deal a concrete list to each seat.

        :return: The ``(deck0, deck1)`` card-ID lists, ordered as the engine's
            two seats. Note the environment randomizes which seat the agent
            occupies, so the curriculum's "agent archetype" is the first entry
            of the pair and lands on whichever seat the agent takes.
        """
        agent, opponent = self._draw_pair()
        self._level_id = self._archetypes.pair_id(agent, opponent)
        return self._deal(agent), self._deal(opponent)

    def seed(self, seed: int | None) -> None:
        """
        Reseed the within-archetype draw.

        :param seed: Seed value; None leaves the sampler untouched.
        """
        if seed is None:
            return
        self._rng.seed(seed)

    def _draw_pair(self) -> tuple[int, int]:
        """
        Draw an archetype pair, from the published distribution or fresh.

        :return: ``(agent archetype, opponent archetype)``.
        """
        if self._explore_prob > 0.0 and self._rng.random() < self._explore_prob:
            return (
                self._rng.randrange(self._archetypes.count),
                self._rng.randrange(self._archetypes.count),
            )
        size = int(self._handles.size[0].item())
        if size <= 0:
            return (
                self._rng.randrange(self._archetypes.count),
                self._rng.randrange(self._archetypes.count),
            )
        probabilities = self._handles.probabilities[:size]
        total = float(probabilities.sum().item())
        if not total > 0.0:
            slot = self._rng.randrange(size)
        else:
            slot = int(torch.multinomial(probabilities, num_samples=1).item())
        return self._archetypes.unpair(int(self._handles.pair_ids[slot].item()))

    def _deal(self, archetype: int) -> Deck:
        """
        Pick one concrete list from an archetype.

        :param archetype: Archetype index to deal from.
        :return: A copy of the chosen deck's card IDs.
        """
        positions = self._archetypes.decks_for(archetype)
        return list(self._decks[self._rng.choice(positions)])
