from functools import partial
from pathlib import Path
from types import SimpleNamespace

from src.env.deck import load_deck
from src.policies.random_masked_policy import RandomMaskedPolicy
from src.training.env_factory import make_env
from src.training.evaluator import Evaluator, _archetype_metrics, _episode_archetype

DECK_PATH = str(Path(__file__).parents[1] / "decks" / "example.csv")


def _stub_env(
    deck_labels: tuple[str, str] | None, agent_seat: int = 0
) -> SimpleNamespace:
    """A transform-wrapped env stub exposing the attributes eval reads.

    :param deck_labels: The leaf env's ``deck_labels``, or None if unlabelled.
    :param agent_seat: The leaf env's ``agent_seat``.
    :return: A namespace whose ``base_env`` is the leaf, so unwrapping applies.
    """
    leaf = SimpleNamespace(deck_labels=deck_labels, agent_seat=agent_seat)
    return SimpleNamespace(base_env=leaf)


def test_archetype_metrics_summarizes_spread() -> None:
    """
    ``_archetype_metrics`` macro-averages per-archetype rates and reports spread.
    """
    # A: 1.0 win-rate (2/2), B: 0.0 (0/2). Macro mean 0.5, min 0, max 1.
    metrics = _archetype_metrics({"A": [2, 2], "B": [2, 0]})
    assert metrics["archetype_count"] == 2.0
    assert metrics["archetype_win_rate_mean"] == 0.5
    assert metrics["archetype_win_rate_min"] == 0.0
    assert metrics["archetype_win_rate_max"] == 1.0
    assert metrics["archetype_win_rate_std"] > 0.0  # the two archetypes disagree
    assert metrics["archetype_win_rate/A"] == 1.0
    assert metrics["archetype_win_rate/B"] == 0.0


def test_archetype_metrics_single_archetype_has_zero_spread() -> None:
    """
    One archetype gives a well-defined mean and a zero (not undefined) spread.
    """
    metrics = _archetype_metrics({"A": [4, 3]})
    assert metrics["archetype_win_rate_mean"] == 0.75
    assert metrics["archetype_win_rate_std"] == 0.0
    # Bottom quartile of a single archetype is that archetype.
    assert metrics["archetype_win_rate_worst_quartile"] == 0.75


def test_worst_quartile_averages_the_weakest_archetypes() -> None:
    """
    ``worst_quartile`` means the weakest 25% of archetypes, not the single min.
    """
    # 8 archetypes at rates 0.0..0.7; bottom quartile is the 2 weakest (0.0, 0.1).
    per_archetype = {f"a{i}": [10, i] for i in range(8)}  # rate i/10
    metrics = _archetype_metrics(per_archetype)
    assert metrics["archetype_win_rate_min"] == 0.0
    assert metrics["archetype_win_rate_worst_quartile"] == 0.05  # mean(0.0, 0.1)
    # It sits between the fragile min and the macro mean.
    assert (
        metrics["archetype_win_rate_min"]
        <= metrics["archetype_win_rate_worst_quartile"]
        <= metrics["archetype_win_rate_mean"]
    )


def test_archetype_metrics_empty_is_absent() -> None:
    """
    With nothing recorded (unlabelled pool) no archetype metrics are emitted.
    """
    assert _archetype_metrics({}) == {}


def test_episode_archetype_credits_the_opponent_seat() -> None:
    """
    An episode is attributed to the deck the agent *faced*, not the one it piloted.

    Evaluation holds the agent's deck fixed, so crediting its own archetype would
    put every episode in one bucket; which opponents it beats is the useful split.
    """
    # Agent sits in seat 1, so seat 0's "mirror-a" is the opponent it faced.
    assert (
        _episode_archetype(_stub_env(("mirror-a", "opp-b"), agent_seat=1)) == "mirror-a"
    )
    # And from the other seat, the opponent is seat 1's deck.
    assert _episode_archetype(_stub_env(("mirror-a", "opp-b"), agent_seat=0)) == "opp-b"


def test_episode_archetype_none_without_labels() -> None:
    """
    An unlabelled env yields no archetype, so the episode goes uncredited.
    """
    assert _episode_archetype(_stub_env(None)) is None


def _labelled_pool_factory(labels: list[str]):
    """Env factory over a labelled mirror pool of the example deck.

    :param labels: One archetype label per pooled deck.
    :return: A zero-arg factory building a masked TCG env on that pool.
    """
    deck = load_deck(DECK_PATH)
    spec = {
        "kind": "pool",
        "decks": [list(deck) for _ in labels],
        "matchup": "mirror",
        "mode": "round_robin",
        "labels": labels,
    }
    return partial(make_env, spec, 96, 0)


def test_evaluate_reports_per_archetype() -> None:
    """
    Over a labelled pool the report carries per-archetype rates and their spread.
    """
    evaluator = Evaluator(
        env_factory=_labelled_pool_factory(["arch-a", "arch-b"]),
        n_episodes=6,
        per_archetype=True,
    )
    try:
        metrics = evaluator.evaluate(RandomMaskedPolicy())
    finally:
        evaluator.close()
    assert metrics["archetype_count"] >= 1
    # Every archetype seen contributes a rate in [0, 1] and folds into the spread.
    per_arch = {k: v for k, v in metrics.items() if k.startswith("archetype_win_rate/")}
    assert per_arch
    assert all(0.0 <= v <= 1.0 for v in per_arch.values())
    assert metrics["archetype_win_rate_min"] <= metrics["archetype_win_rate_mean"]
    assert metrics["archetype_win_rate_mean"] <= metrics["archetype_win_rate_max"]


def test_evaluate_without_labels_omits_archetype_metrics() -> None:
    """
    A fixed (unlabelled) env yields only the aggregate metrics, even when the
    per-archetype breakdown is requested.
    """
    deck = load_deck(DECK_PATH)
    factory = partial(make_env, {"kind": "fixed", "deck0": deck, "deck1": deck}, 96, 0)
    evaluator = Evaluator(env_factory=factory, n_episodes=3, per_archetype=True)
    try:
        metrics = evaluator.evaluate(RandomMaskedPolicy())
    finally:
        evaluator.close()
    assert "win_rate" in metrics  # aggregate still reported
    assert not any(key.startswith("archetype_") for key in metrics)
