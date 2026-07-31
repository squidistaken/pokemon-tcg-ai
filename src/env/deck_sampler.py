import random
from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

Deck = list[int]


@runtime_checkable
class DeckSampler(Protocol):
    """
    Chooses the ``(deck0, deck1)`` matchup used for one episode.
    """

    def sample(self) -> tuple[Deck, Deck]:
        """Return the ``(deck0, deck1)`` card-ID lists for the next episode."""
        ...

    def seed(self, seed: int | None) -> None:
        """Reseed the sampler's randomness; a None seed leaves it untouched."""
        ...


class FixedDeckSampler:
    """
    Always returns the same ``(deck0, deck1)`` pair.
    """

    def __init__(self, deck0: Sequence[int], deck1: Sequence[int]) -> None:
        """
        :param deck0: 60 card IDs for player 0.
        :param deck1: 60 card IDs for player 1.
        """
        self._deck0 = list(deck0)
        self._deck1 = list(deck1)

    def sample(self) -> tuple[Deck, Deck]:
        """
        :return: A fresh copy of the fixed ``(deck0, deck1)`` pair.
        """
        return list(self._deck0), list(self._deck1)

    def seed(self, seed: int | None) -> None:  # noqa: ARG002, PLR6301 - protocol method, deterministic
        """
        No-op: a fixed sampler has no randomness to seed.

        :param seed: Ignored.
        """
        return


class PoolDeckSampler:
    """
    Samples an episode matchup from a pool of decks.

    Instead of one fixed deck pair, every episode draws decks from a corpus so
    the policy is trained across many archetypes rather than overfitting one
    card pool.
    """

    def __init__(
        self,
        decks: Sequence[Sequence[int]],
        matchup: str = "mirror",
        mode: str = "uniform",
        seed: int | None = None,
        mirror_prob: float | None = None,
        weights: Sequence[float] | None = None,
    ) -> None:
        """
        :param decks: Pool of decks, each a list of 60 card IDs.
        :param matchup: ``"mirror"`` or ``"independent"``. Used only when
                        ``mirror_prob`` is None, to derive it.
        :param mode: ``"uniform"`` or ``"round_robin"``.
        :param seed: Seed for uniform sampling and the round-robin start offset.
        :param mirror_prob: Probability an episode is a mirror match; the rest
            are independent draws. None derives it from ``matchup``
            (mirror -> 1.0, independent -> 0.0), so the presets are the endpoints
            and any value between mixes them.
        :param weights: Per-deck sampling weights aligned with ``decks`` (e.g.
            tournament win-rate). None samples uniformly. Ignored under
            ``round_robin``, which is deterministic coverage.
        :raises ValueError: If the pool is empty, an option is unknown, or the
            weights are malformed.
        """
        if not decks:
            raise ValueError("PoolDeckSampler requires a non-empty deck pool")
        if mode not in ("uniform", "round_robin"):
            raise ValueError(
                f"unknown mode {mode!r}; expected 'uniform' or 'round_robin'"
            )
        if mirror_prob is None:
            if matchup not in ("mirror", "independent"):
                raise ValueError(
                    f"unknown matchup {matchup!r}; expected 'mirror' or 'independent'"
                )
            mirror_prob = 1.0 if matchup == "mirror" else 0.0
        elif not 0.0 <= mirror_prob <= 1.0:
            raise ValueError(f"mirror_prob must be in [0, 1], got {mirror_prob}")
        self._decks = [list(deck) for deck in decks]
        self._mirror_prob = float(mirror_prob)
        self._mode = mode
        self._weights = self._validate_weights(weights, len(self._decks))
        self._rng = random.Random(seed)

        # A random start offset keeps two round-robin workers from marching in
        # lockstep through identical decks; still deterministic given the seed.
        self._cursor = self._rng.randrange(len(self._decks))

    @staticmethod
    def _validate_weights(
        weights: Sequence[float] | None, pool_size: int
    ) -> list[float] | None:
        """
        Validate and copy per-deck sampling weights.

        :param weights: Weights aligned with the pool, or None for uniform.
        :param pool_size: Number of decks the weights must line up with.
        :return: A copied weight list, or None.
        :raises ValueError: If the weights are the wrong length, negative, or
            sum to zero.
        """
        if weights is None:
            return None
        weights = list(weights)
        if len(weights) != pool_size:
            raise ValueError(
                f"weights length {len(weights)} does not match pool size {pool_size}"
            )
        if any(weight < 0 for weight in weights):
            raise ValueError("weights must be non-negative")
        if sum(weights) <= 0:
            raise ValueError("weights must have a positive sum")
        return weights

    def sample(self) -> tuple[Deck, Deck]:
        """
        :return: The ``(deck0, deck1)`` pair for the next episode.
        """
        index0 = self._next_index()
        mirror = self._rng.random() < self._mirror_prob
        index1 = index0 if mirror else self._next_index()

        return list(self._decks[index0]), list(self._decks[index1])

    def _next_index(self) -> int:
        """
        :return: Index of the next deck under the configured draw mode.
        """
        if self._mode == "uniform":
            if self._weights is not None:
                return self._rng.choices(
                    range(len(self._decks)), weights=self._weights, k=1
                )[0]
            return self._rng.randrange(len(self._decks))
        index = self._cursor % len(self._decks)
        self._cursor += 1

        return index

    def seed(self, seed: int | None) -> None:
        """
        Reseed the draw RNG and re-derive the round-robin start offset.

        :param seed: Seed value; None leaves the sampler untouched.
        """
        if seed is None:
            return
        self._rng.seed(seed)
        self._cursor = self._rng.randrange(len(self._decks))


def build_deck_sampler(spec: dict[str, Any], seed: int | None = None) -> DeckSampler:
    """
    Build a :class:`DeckSampler` from a plain, picklable spec dict.

    :param spec: ``{"kind": "fixed", "deck0": [...], "deck1": [...]}``,
        ``{"kind": "pool", "decks": [[...], ...], "matchup": ..., "mode": ...,
        "mirror_prob": ..., "weights": [...]}`` (the last two optional), or
        ``{"kind": "curriculum", "decks": [[...], ...], "archetypes": ...,
        "handles": ...}``.
    :param seed: Seed forwarded to the pool and curriculum samplers (ignored
        for fixed).
    :return: A constructed deck sampler.
    :raises ValueError: If ``spec["kind"]`` is unknown.
    """
    kind = spec.get("kind")
    if kind == "fixed":
        return FixedDeckSampler(spec["deck0"], spec["deck1"])
    if kind == "curriculum":
        # Imported here rather than at module scope: the curriculum sampler
        # pulls in torch, and this module is otherwise dependency-free.
        from .curriculum_deck_sampler import CurriculumDeckSampler

        return CurriculumDeckSampler(
            decks=spec["decks"],
            archetypes=spec["archetypes"],
            handles=spec["handles"],
            seed=seed,
        )
    if kind == "pool":
        return PoolDeckSampler(
            decks=spec["decks"],
            matchup=spec.get("matchup", "mirror"),
            mode=spec.get("mode", "uniform"),
            seed=seed,
            mirror_prob=spec.get("mirror_prob"),
            weights=spec.get("weights"),
        )

    raise ValueError(f"unknown deck sampler kind {kind!r}; expected 'fixed' or 'pool'")
