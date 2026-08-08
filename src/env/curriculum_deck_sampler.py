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
    a concrete list from within each archetype -- uniformly, or by ``weights``
    when the run configures ``env.deck_weighting``. List-level diversity is
    preserved; only the archetype pairing is curated.

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
        weights: Sequence[float] | None = None,
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
        :param weights: Per-deck weights aligned with ``decks``, biasing the
            list dealt from within the drawn archetype. None deals uniformly.
        :raises ValueError: If the pool is empty, an archetype references a
            deck position the pool does not contain, ``explore_prob`` is out of
            ``[0, 1]``, or the weights are malformed.
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
        self._torch_rng = torch.Generator()
        if seed is not None:
            self._torch_rng.manual_seed(seed)
        self._explore_prob = explore_prob
        self._weights = self._validate_weights(weights, archetypes, len(self._decks))
        self._archetype_weights = self._archetype_totals(self._weights, archetypes)
        self._level_id = NO_LEVEL

    @staticmethod
    def _archetype_totals(
        weights: list[float] | None, archetypes: ArchetypeIndex
    ) -> list[float] | None:
        """
        Sum per-deck weights into one weight per archetype.

        Used for the discovery draw, so an archetype is explored in proportion
        to the same evidence its lists are dealt by, rather than uniformly.

        :param weights: Validated per-deck weights, or None for uniform.
        :param archetypes: Grouping to sum within.
        :return: One weight per archetype, or None when the decks are uniform.
        """
        if weights is None:
            return None
        return [
            sum(weights[position] for position in archetypes.decks_for(archetype))
            for archetype in range(archetypes.count)
        ]

    @staticmethod
    def _validate_weights(
        weights: Sequence[float] | None,
        archetypes: ArchetypeIndex,
        pool_size: int,
    ) -> list[float] | None:
        """
        Validate and copy per-deck weights for the within-archetype draw.

        Every archetype is checked separately: the draw happens inside one
        archetype at a time, so a globally positive sum is not enough -- an
        archetype whose lists all weigh zero would raise from
        :meth:`random.Random.choices` mid-episode instead of at construction.

        :param weights: Weights aligned with the pool, or None for uniform.
        :param archetypes: Grouping the draw happens within.
        :param pool_size: Number of decks the weights must line up with.
        :return: A copied weight list, or None.
        :raises ValueError: If the weights are the wrong length, negative, or
            leave some archetype with no positive weight.
        """
        if weights is None:
            return None
        weights = [float(weight) for weight in weights]
        if len(weights) != pool_size:
            raise ValueError(
                f"weights length {len(weights)} does not match pool size {pool_size}"
            )
        if any(weight < 0.0 for weight in weights):
            raise ValueError("weights must be non-negative")
        for archetype in range(archetypes.count):
            positions = archetypes.decks_for(archetype)
            if sum(weights[position] for position in positions) <= 0.0:
                raise ValueError(
                    f"archetype {archetypes.names[archetype]!r} has no deck with a "
                    "positive weight, so no list could ever be dealt for it"
                )
        return weights

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
        Draw the next episode's matchup, assuming the agent takes seat 0.

        Prefer :meth:`sample_for_seat`: the environment flips a coin for the
        agent's seat, and a level is the *ordered* pair
        ``(agent archetype, opponent archetype)``.

        :return: The ``(deck0, deck1)`` card-ID lists.
        """
        return self.sample_for_seat(0)

    def sample_for_seat(self, agent_seat: int) -> tuple[Deck, Deck]:
        """
        Draw the next matchup and deal the agent's archetype to its own seat.

        A level identifies an ordered pair, ``pair_id(agent, opponent)`` and
        ``pair_id(opponent, agent)`` being different levels over an asymmetric
        matchup table. Dealing the drawn agent archetype positionally to seat 0
        therefore hands it to the *opponent* on the half of episodes where the
        environment seats the agent second, and credits that episode's outcome
        to the transposed level.

        :param agent_seat: Seat index (0 or 1) the agent occupies this episode.
        :return: The ``(deck0, deck1)`` card-ID lists in engine seat order.
        """
        agent, opponent = self._draw_pair()
        self._level_id = self._archetypes.pair_id(agent, opponent)
        agent_deck, opponent_deck = self._deal(agent), self._deal(opponent)
        if agent_seat == 0:
            return agent_deck, opponent_deck
        return opponent_deck, agent_deck

    def seed(self, seed: int | None) -> None:
        """
        Reseed the matchup draw and the within-archetype deal.

        :param seed: Seed value; None leaves the sampler untouched.
        """
        if seed is None:
            return
        self._rng.seed(seed)
        self._torch_rng.manual_seed(seed)

    def _draw_pair(self) -> tuple[int, int]:
        """
        Draw an archetype pair, from the published distribution or fresh.

        :return: ``(agent archetype, opponent archetype)``.
        """
        if self._explore_prob > 0.0 and self._rng.random() < self._explore_prob:
            return self._explore_pair()
        size = int(self._handles.size[0].item())
        if size <= 0:
            return self._explore_pair()
        probabilities = self._handles.probabilities[:size]
        total = float(probabilities.sum().item())
        if not total > 0.0:
            slot = self._rng.randrange(size)
        else:
            slot = int(
                torch.multinomial(
                    probabilities, num_samples=1, generator=self._torch_rng
                ).item()
            )
        return self._archetypes.unpair(int(self._handles.pair_ids[slot].item()))

    def _explore_pair(self) -> tuple[int, int]:
        """
        Draw a fresh archetype pair, outside the published distribution.

        This is the discovery path -- the only way a matchup that is not in the
        level buffer ever gets played, and so the only thing that decides which
        matchups the curriculum comes to know about. Drawing it uniformly gave a
        134-archetype corpus's rarest folder exactly the same chance as its
        most-played one, so discovery spent most of its budget on matchups the
        ladder almost never deals. With ``weights`` it follows the same
        distribution the list draw does: an archetype's chance is the sum of its
        lists' weights, which under ``deck_weighting=observation`` is its total
        observation count.

        :return: ``(agent archetype, opponent archetype)``.
        """
        if self._archetype_weights is None:
            return (
                self._rng.randrange(self._archetypes.count),
                self._rng.randrange(self._archetypes.count),
            )
        drawn = self._rng.choices(
            range(self._archetypes.count), weights=self._archetype_weights, k=2
        )
        return drawn[0], drawn[1]

    def _deal(self, archetype: int) -> Deck:
        """
        Pick one concrete list from an archetype.

        :param archetype: Archetype index to deal from.
        :return: A copy of the chosen deck's card IDs.
        """
        positions = self._archetypes.decks_for(archetype)
        if self._weights is None:
            return list(self._decks[self._rng.choice(positions)])
        weights = [self._weights[position] for position in positions]
        chosen = self._rng.choices(positions, weights=weights, k=1)[0]
        return list(self._decks[chosen])
