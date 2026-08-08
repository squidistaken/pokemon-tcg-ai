import random
from collections.abc import Sequence

from .deck_sampler import Deck, DeckSampler, sample_for_seat


class AgentDeckSampler:
    """
    Pins the agent to one deck and draws only its opponent from a pool.

    A symmetric pool sampler draws both seats from the same corpus, so with
    134 archetypes the agent faces 134 x 134 = 17,956 distinct matchups. A run
    of 50,000 episodes gives each of them under three episodes, which is not
    enough to learn any of them -- and the submitted agent pilots exactly one
    deck, so all but 1/134 of that experience is spent on lists it will never
    play.

    Fixing the agent's deck collapses the space to one matchup per opponent
    archetype while leaving the opposing field exactly as wide as the ladder
    the agent is scored against. The same 50,000 episodes then buy a few
    hundred per matchup instead of three.

    ``field_probability`` keeps a minority of episodes on a pool-drawn agent
    deck. Without it the card embeddings and per-card statics outside the
    pinned list stop receiving gradient entirely, and the observation encoder
    degrades on precisely the cards the *opponent* plays.

    Seat placement is explicit rather than positional: the environment flips a
    coin for the agent's seat before asking for decks, so a sampler that always
    returns the agent's deck first hands it to the opponent half the time.
    """

    def __init__(
        self,
        agent_deck: Sequence[int],
        field_sampler: DeckSampler,
        agent_label: str | None = None,
        field_probability: float = 0.0,
        seed: int | None = None,
    ) -> None:
        """
        :param agent_deck: The 60 card IDs the agent pilots.
        :param field_sampler: Draws the opposing field; its second deck is
            taken as the opponent, which is a uniform draw under every
            ``independent`` pool configuration.
        :param agent_label: Archetype label reported for the pinned deck, so
            per-archetype attribution keeps working. None leaves the pair
            unlabelled.
        :param field_probability: Share of episodes that ignore the pin and
            take both decks from ``field_sampler``, keeping the rest of the
            corpus in the gradient. 0 pins every episode.
        :param seed: Seed for the field/pin coin flip.
        :raises ValueError: If ``agent_deck`` is empty or the probability is
            outside [0, 1].
        """
        if not agent_deck:
            raise ValueError("AgentDeckSampler requires a non-empty agent deck")
        if not 0.0 <= field_probability <= 1.0:
            raise ValueError(
                f"field_probability must be in [0, 1], got {field_probability}"
            )
        self._agent_deck = list(agent_deck)
        self._field_sampler = field_sampler
        self._agent_label = agent_label
        self._field_probability = float(field_probability)
        self._rng = random.Random(seed)
        self._last_labels: tuple[str, str] | None = None

    @property
    def last_labels(self) -> tuple[str, str] | None:
        """
        Archetype labels of the last sampled pair, in ``(deck0, deck1)`` seat
        order so callers can index them by the agent's seat.

        :return: The seat-ordered labels, or None when unlabelled.
        """
        return self._last_labels

    @property
    def level_id(self) -> int:
        """
        Matchup identifier from the wrapped sampler, for a curriculum.

        :return: The field sampler's level, or ``NO_LEVEL`` when it keeps none.
        """
        # Imported here rather than at module scope: the curriculum sampler
        # pulls in torch, which this module otherwise does not need.
        from .curriculum_deck_sampler import NO_LEVEL

        return int(getattr(self._field_sampler, "level_id", NO_LEVEL))

    def sample(self) -> tuple[Deck, Deck]:
        """
        Draw a matchup without knowing the agent's seat.

        Present only to satisfy :class:`~src.env.deck_sampler.DeckSampler`;
        it assumes seat 0, which is what the protocol can express.

        :return: The ``(deck0, deck1)`` pair with the agent's deck on seat 0.
        """
        return self.sample_for_seat(0)

    def sample_for_seat(self, agent_seat: int) -> tuple[Deck, Deck]:
        """
        Draw a matchup with the pinned deck on the agent's actual seat.

        :param agent_seat: Seat index (0 or 1) the agent occupies this episode.
        :return: The ``(deck0, deck1)`` pair in engine seat order.
        """
        field_deck0, field_deck1 = sample_for_seat(self._field_sampler, agent_seat)
        field_labels = getattr(self._field_sampler, "last_labels", None)
        if self._rng.random() < self._field_probability:
            self._last_labels = field_labels
            return field_deck0, field_deck1

        opponent_deck = field_deck1
        opponent_label = field_labels[1] if field_labels is not None else None
        agent_label = self._agent_label
        if agent_label is None or opponent_label is None:
            self._last_labels = None
        elif agent_seat == 0:
            self._last_labels = (agent_label, opponent_label)
        else:
            self._last_labels = (opponent_label, agent_label)

        if agent_seat == 0:
            return list(self._agent_deck), opponent_deck
        return opponent_deck, list(self._agent_deck)

    def seed(self, seed: int | None) -> None:
        """
        Reseed the coin flip and the wrapped field sampler.

        :param seed: Seed value; None leaves both untouched.
        """
        if seed is None:
            return
        self._rng.seed(seed)
        self._field_sampler.seed(seed)
