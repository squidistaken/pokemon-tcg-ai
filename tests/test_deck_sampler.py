from pathlib import Path

import pytest
from omegaconf import DictConfig, OmegaConf

from src.env.deck import load_deck, load_decks, resolve_deck_paths
from src.env.deck_sampler import (
    FixedDeckSampler,
    PoolDeckSampler,
    build_deck_sampler,
)
from src.training.env_factory import (
    _build_sampler_spec,
    _deck_labels,
    _deck_weights,
    _limit_pool_width,
    _record_winrate,
)

REPO_ROOT = Path(__file__).parents[1]
EXAMPLE_DECK = str(REPO_ROOT / "decks" / "example.csv")
# The deck corpus is organized into per-archetype subfolders under decks/.
CORPUS_DIR = REPO_ROOT / "decks"

# The scraped corpus is a pulled release artifact, not committed. Only the
# corpus-dependent tests are marked to skip when it is absent (e.g. on CI, which
# checks out just the committed example deck) — the pure-logic sampler tests
# below run everywhere, so CI still exercises them.
_HAS_CORPUS = bool(list(CORPUS_DIR.glob("*/*.csv"))) if CORPUS_DIR.is_dir() else False
requires_corpus = pytest.mark.skipif(
    not _HAS_CORPUS,
    reason="scraped deck corpus not installed; run ./scripts/fetch_decks.sh",
)


def _fake_pool(n: int) -> list[list[int]]:
    """Build ``n`` distinguishable 60-card decks (deck k is all card id k).

    :param n: Number of decks to build.
    :return: A pool of ``n`` decks, each 60 copies of its own index.
    """
    return [[k] * 60 for k in range(n)]


def test_load_deck_tolerates_header(tmp_path: Path) -> None:
    """
    ``load_deck`` skips a leading non-numeric header line (e.g. ``card_id``).
    """
    deck_file = tmp_path / "d.csv"
    deck_file.write_text("card_id\n" + "\n".join("5" for _ in range(60)) + "\n")
    assert load_deck(str(deck_file)) == [5] * 60


def test_load_deck_rejects_wrong_size(tmp_path: Path) -> None:
    """
    ``load_deck`` raises when a file does not resolve to exactly 60 cards.
    """
    deck_file = tmp_path / "d.csv"
    deck_file.write_text("\n".join("1" for _ in range(59)))
    with pytest.raises(ValueError):
        load_deck(str(deck_file))


def test_fixed_sampler_is_constant() -> None:
    """
    ``FixedDeckSampler`` returns the same ``(deck0, deck1)`` pair every call.
    """
    sampler = FixedDeckSampler([1] * 60, [2] * 60)
    for _ in range(5):
        deck0, deck1 = sampler.sample()
        assert deck0 == [1] * 60
        assert deck1 == [2] * 60


def test_fixed_sampler_returns_copies() -> None:
    """
    Mutating a sampled deck does not corrupt the sampler's stored decks.
    """
    sampler = FixedDeckSampler([1] * 60, [2] * 60)
    deck0, _ = sampler.sample()
    deck0[0] = 999
    assert sampler.sample()[0][0] == 1  # mutation did not leak back


def test_pool_sampler_mirror_matches_seats() -> None:
    """
    ``mirror`` matchup hands both seats the identical deck every episode.
    """
    sampler = PoolDeckSampler(_fake_pool(8), matchup="mirror", mode="uniform", seed=0)
    for _ in range(20):
        deck0, deck1 = sampler.sample()
        assert deck0 == deck1


def test_pool_sampler_uniform_covers_pool() -> None:
    """
    Uniform sampling eventually draws every deck in the pool.
    """
    sampler = PoolDeckSampler(_fake_pool(5), matchup="mirror", mode="uniform", seed=1)
    seen = {sampler.sample()[0][0] for _ in range(500)}
    assert seen == set(range(5))  # every deck eventually drawn


def test_pool_sampler_round_robin_cycles_evenly() -> None:
    """
    Round-robin sampling rotates through the pool in a fixed, even cycle.
    """
    pool = _fake_pool(4)
    sampler = PoolDeckSampler(pool, matchup="mirror", mode="round_robin", seed=0)
    drawn = [sampler.sample()[0][0] for _ in range(8)]
    # Two full passes over the pool, each deck exactly twice, contiguous cycle.
    assert sorted(drawn[:4]) == [0, 1, 2, 3]
    assert drawn[:4] == drawn[4:]


def test_mirror_prob_endpoints_match_presets() -> None:
    """
    ``mirror_prob`` 1.0/0.0 reproduce the mirror/independent presets exactly.
    """
    pool = _fake_pool(20)
    always_mirror = PoolDeckSampler(pool, mirror_prob=1.0, seed=0)
    always_indep = PoolDeckSampler(pool, mirror_prob=0.0, seed=0)
    assert all(d0 == d1 for d0, d1 in (always_mirror.sample() for _ in range(50)))
    assert any(d0 != d1 for d0, d1 in (always_indep.sample() for _ in range(50)))


def test_mirror_prob_mixes_matchups() -> None:
    """
    A fractional ``mirror_prob`` yields roughly that fraction of mirror games.
    """
    # ~half the episodes should be mirror (both seats equal) with prob 0.5;
    # in a 20-deck pool an independent draw coincidentally matches only ~1/20.
    sampler = PoolDeckSampler(_fake_pool(20), mirror_prob=0.5, seed=2)
    n = 2000
    mirror_frac = sum(d0 == d1 for d0, d1 in (sampler.sample() for _ in range(n))) / n
    assert 0.4 < mirror_frac < 0.65


def test_mirror_prob_overrides_matchup_and_validates() -> None:
    """
    An explicit ``mirror_prob`` overrides ``matchup`` and is range-checked.
    """
    # mirror_prob wins over the matchup string when both are given.
    s = PoolDeckSampler(_fake_pool(8), matchup="mirror", mirror_prob=0.0, seed=0)
    assert any(d0 != d1 for d0, d1 in (s.sample() for _ in range(50)))
    with pytest.raises(ValueError):
        PoolDeckSampler(_fake_pool(2), mirror_prob=1.5)


def test_weights_bias_the_draw() -> None:
    """
    Per-deck weights skew the uniform draw toward the heavier decks.
    """
    # Deck 2 has ~50x the weight, so it should dominate the draw.
    sampler = PoolDeckSampler(
        _fake_pool(3), mode="uniform", seed=1, weights=[1.0, 1.0, 50.0]
    )
    counts = [0, 0, 0]
    for _ in range(600):
        counts[sampler.sample()[0][0]] += 1
    assert counts[2] > counts[0] + counts[1]


def test_weights_validation() -> None:
    """
    Malformed weights (wrong length, negative, zero-sum) are rejected.
    """
    with pytest.raises(ValueError):
        PoolDeckSampler(_fake_pool(3), weights=[1.0, 1.0])  # wrong length
    with pytest.raises(ValueError):
        PoolDeckSampler(_fake_pool(2), weights=[1.0, -1.0])  # negative
    with pytest.raises(ValueError):
        PoolDeckSampler(_fake_pool(2), weights=[0.0, 0.0])  # zero sum


def test_build_deck_sampler_passes_mix_and_weights() -> None:
    """
    ``build_deck_sampler`` forwards ``mirror_prob`` and ``weights`` from the spec.
    """
    pool = _fake_pool(4)
    s = build_deck_sampler(
        {
            "kind": "pool",
            "decks": pool,
            "mirror_prob": 0.0,
            "weights": [1.0, 1.0, 1.0, 9.0],
        },
        seed=0,
    )
    assert isinstance(s, PoolDeckSampler)
    counts = [0, 0, 0, 0]
    for _ in range(400):
        counts[s.sample()[0][0]] += 1
    assert counts[3] == max(counts)  # the heavily-weighted deck wins


def test_labels_default_to_none() -> None:
    """
    Without ``labels``, the sampler records no per-episode archetype labels.
    """
    sampler = PoolDeckSampler(_fake_pool(4), matchup="mirror", seed=0)
    assert sampler.last_labels is None
    sampler.sample()
    assert sampler.last_labels is None  # still None after a draw


def test_labels_track_the_sampled_pair() -> None:
    """
    ``last_labels`` reports the archetype of each seat's just-sampled deck.
    """
    labels = [f"arch{k}" for k in range(4)]
    sampler = PoolDeckSampler(
        _fake_pool(4), mode="round_robin", matchup="mirror", labels=labels, seed=0
    )
    for _ in range(8):
        (deck0, _), pair = sampler.sample(), sampler.last_labels
        assert pair is not None
        # Deck k is 60 copies of k, so the label index must match the deck id.
        assert pair == (labels[deck0[0]], labels[deck0[0]])  # mirror: both equal


def test_labels_can_differ_under_independent_matchup() -> None:
    """
    Independent draws label the two seats separately, so labels can differ.
    """
    labels = [f"arch{k}" for k in range(10)]
    sampler = PoolDeckSampler(
        _fake_pool(10), matchup="independent", labels=labels, seed=3
    )
    pairs = []
    for _ in range(50):
        sampler.sample()
        pairs.append(sampler.last_labels)
    assert any(p is not None and p[0] != p[1] for p in pairs)


def test_labels_length_is_validated() -> None:
    """
    Labels that do not line up one-per-deck are rejected at construction.
    """
    with pytest.raises(ValueError):
        PoolDeckSampler(_fake_pool(3), labels=["a", "b"])  # wrong length


def test_build_deck_sampler_forwards_labels() -> None:
    """
    ``build_deck_sampler`` forwards ``labels`` from the spec to the pool sampler.
    """
    s = build_deck_sampler(
        {"kind": "pool", "decks": _fake_pool(3), "labels": ["x", "y", "z"]}, seed=0
    )
    assert isinstance(s, PoolDeckSampler)
    s.sample()
    assert s.last_labels is not None
    assert s.last_labels[0] in {"x", "y", "z"}


def test_pool_sampler_independent_can_differ() -> None:
    """
    ``independent`` matchup draws each seat separately, so the decks can differ.
    """
    sampler = PoolDeckSampler(
        _fake_pool(10), matchup="independent", mode="uniform", seed=3
    )
    assert any(d0 != d1 for d0, d1 in (sampler.sample() for _ in range(50)))


def test_pool_sampler_seed_is_reproducible() -> None:
    """
    Two samplers with the same seed produce identical draw sequences.
    """
    pool = _fake_pool(12)
    a = PoolDeckSampler(pool, mode="uniform", seed=7)
    b = PoolDeckSampler(pool, mode="uniform", seed=7)
    assert [a.sample()[0][0] for _ in range(30)] == [
        b.sample()[0][0] for _ in range(30)
    ]


def test_pool_sampler_rejects_empty_pool() -> None:
    """
    Constructing a pool sampler over an empty deck list raises.
    """
    with pytest.raises(ValueError):
        PoolDeckSampler([], mode="uniform")


def test_pool_sampler_rejects_unknown_options() -> None:
    """
    Unknown ``matchup`` or ``mode`` values are rejected at construction.
    """
    with pytest.raises(ValueError):
        PoolDeckSampler(_fake_pool(2), matchup="bogus")
    with pytest.raises(ValueError):
        PoolDeckSampler(_fake_pool(2), mode="bogus")


def test_build_deck_sampler_dispatch() -> None:
    """
    ``build_deck_sampler`` maps the spec ``kind`` to the right sampler class.
    """
    fixed = build_deck_sampler({"kind": "fixed", "deck0": [1] * 60, "deck1": [2] * 60})
    assert isinstance(fixed, FixedDeckSampler)
    pool = build_deck_sampler({"kind": "pool", "decks": _fake_pool(3)}, seed=0)
    assert isinstance(pool, PoolDeckSampler)
    with pytest.raises(ValueError):
        build_deck_sampler({"kind": "nope"})


@requires_corpus
def test_resolve_and_load_scraped_pool() -> None:
    """
    The corpus directory resolves recursively to loadable 60-card deck CSVs.
    """
    paths = resolve_deck_paths(str(CORPUS_DIR))
    assert len(paths) > 100  # the per-archetype corpus, gathered recursively
    assert all(p.endswith(".csv") for p in paths)
    decks = load_decks(paths[:5])
    assert all(len(d) == 60 for d in decks)


@requires_corpus
def test_corpus_pool_excludes_loose_example_deck() -> None:
    """
    Loose CSVs in the corpus root (the example smoke deck) are not pooled.
    """
    paths = resolve_deck_paths(str(CORPUS_DIR))
    # example.csv sits directly in decks/; the corpus lives in archetype
    # subfolders, so the loose smoke deck must not leak into the pool.
    assert not any(Path(p).name == "example.csv" for p in paths)
    assert all(Path(p).parent != CORPUS_DIR for p in paths)


@requires_corpus
def test_resolve_single_archetype_folder() -> None:
    """
    Pointing the resolver at one archetype folder yields just its decks.
    """
    archetype = next(p for p in CORPUS_DIR.iterdir() if p.is_dir())
    paths = resolve_deck_paths(str(archetype))
    assert paths  # a single strategy folder resolves to its own decks
    assert all(Path(p).parent == archetype for p in paths)


def _env_cfg(**env_overrides) -> DictConfig:
    """Build a minimal ``env`` config for ``_build_sampler_spec`` tests.

    :param env_overrides: Keys merged into the ``env`` section (e.g. ``deck_pool``).
    :return: An OmegaConf config with ``seed`` and an ``env`` section.
    """
    base = {
        "seed": 0,
        "env": {
            "deck0": EXAMPLE_DECK,
            "deck1": EXAMPLE_DECK,
            "max_options": 96,
            "num_workers": 2,
            "encoder": "structured",
        },
    }
    base["env"].update(env_overrides)
    return OmegaConf.create(base)


def test_sampler_spec_defaults_to_fixed() -> None:
    """
    With no deck pool configured, the spec is a fixed single-matchup sampler.
    """
    spec = _build_sampler_spec(_env_cfg(), deck_split="train")
    assert spec["kind"] == "fixed"
    assert len(spec["deck0"]) == 60


@requires_corpus
def test_sampler_spec_builds_pool_from_directory() -> None:
    """
    A ``deck_pool`` directory produces a pool spec over the loaded corpus.
    """
    spec = _build_sampler_spec(_env_cfg(deck_pool=str(CORPUS_DIR)), deck_split="train")
    assert spec["kind"] == "pool"
    assert len(spec["decks"]) > 100
    assert spec["matchup"] == "mirror"
    assert spec["mode"] == "uniform"


@requires_corpus
def test_sampler_spec_holdout_is_disjoint() -> None:
    """
    Train and holdout splits share no decks, so eval measures unseen decks.
    """
    cfg = _env_cfg(deck_pool=str(CORPUS_DIR), deck_holdout_frac=0.2)
    train = _build_sampler_spec(cfg, deck_split="train")["decks"]
    holdout = _build_sampler_spec(cfg, deck_split="eval")["decks"]
    train_set = {tuple(d) for d in train}
    holdout_set = {tuple(d) for d in holdout}
    assert holdout_set  # non-empty holdout
    assert train_set.isdisjoint(holdout_set)  # unseen at train time


@requires_corpus
def test_eval_matchup_overrides_only_eval_split() -> None:
    """
    ``eval_deck_matchup`` changes the eval matchup without touching training.
    """
    cfg = _env_cfg(
        deck_pool=str(CORPUS_DIR),
        deck_matchup="independent",
        eval_deck_matchup="mirror",
        deck_holdout_frac=0.2,
    )
    assert _build_sampler_spec(cfg, deck_split="train")["matchup"] == "independent"
    assert _build_sampler_spec(cfg, deck_split="eval")["matchup"] == "mirror"


@requires_corpus
def test_eval_sampling_overrides_only_eval_split() -> None:
    """
    ``eval_deck_sampling`` changes the eval draw mode without touching training.
    """
    cfg = _env_cfg(
        deck_pool=str(CORPUS_DIR),
        deck_sampling="uniform",
        eval_deck_sampling="round_robin",
        deck_holdout_frac=0.2,
    )
    assert _build_sampler_spec(cfg, deck_split="train")["mode"] == "uniform"
    assert _build_sampler_spec(cfg, deck_split="eval")["mode"] == "round_robin"


@requires_corpus
def test_eval_sampling_falls_back_to_deck_sampling_when_unset() -> None:
    """
    Without an ``eval_deck_sampling`` key, eval inherits the training draw mode.
    """
    cfg = _env_cfg(
        deck_pool=str(CORPUS_DIR), deck_sampling="round_robin", deck_holdout_frac=0.2
    )
    assert _build_sampler_spec(cfg, deck_split="eval")["mode"] == "round_robin"


@requires_corpus
def test_eval_matchup_falls_back_to_deck_matchup_when_unset() -> None:
    """
    Without an ``eval_deck_matchup`` key, eval inherits the training matchup.
    """
    cfg = _env_cfg(
        deck_pool=str(CORPUS_DIR), deck_matchup="independent", deck_holdout_frac=0.2
    )
    # No eval_deck_matchup key -> eval inherits the training matchup.
    assert _build_sampler_spec(cfg, deck_split="eval")["matchup"] == "independent"


@requires_corpus
def test_mix_and_weighting_apply_to_train_not_eval() -> None:
    """
    ``deck_mirror_prob``/``deck_weighting`` shape the train draw but not eval.
    """
    cfg = _env_cfg(
        deck_pool=str(CORPUS_DIR),
        deck_mirror_prob=0.5,
        deck_weighting="winrate",
        deck_holdout_frac=0.2,
    )
    train = _build_sampler_spec(cfg, deck_split="train")
    eval_ = _build_sampler_spec(cfg, deck_split="eval")
    # Train spec carries the mix + weights; weights align with the train subset.
    assert train["mirror_prob"] == 0.5
    assert len(train["weights"]) == len(train["decks"])
    assert all(w > 0 for w in train["weights"])
    # Eval stays an unbiased uniform draw over the held-out pool.
    assert "mirror_prob" not in eval_
    assert "weights" not in eval_


@requires_corpus
def test_eval_spec_carries_archetype_labels() -> None:
    """
    The eval split attaches one archetype label per held-out deck; train omits them.
    """
    cfg = _env_cfg(deck_pool=str(CORPUS_DIR), deck_holdout_frac=0.2)
    eval_ = _build_sampler_spec(cfg, deck_split="eval")
    train = _build_sampler_spec(cfg, deck_split="train")
    assert "labels" in eval_
    assert len(eval_["labels"]) == len(eval_["decks"])
    assert all(isinstance(label, str) and label for label in eval_["labels"])
    assert "labels" not in train  # labels are an eval-only concern


@requires_corpus
def test_deck_labels_prefer_manifest_archetype() -> None:
    """
    ``_deck_labels`` reads the manifest archetype, falling back to the folder.
    """
    paths = resolve_deck_paths(str(CORPUS_DIR))[:20]
    labels = _deck_labels(paths)
    assert len(labels) == len(paths)
    assert all(isinstance(label, str) and label for label in labels)


def test_limit_pool_width_keeps_exactly_n_archetypes(tmp_path: Path) -> None:
    """
    Width filtering keeps decks from exactly ``width`` archetypes, deterministically.
    """
    paths: list[str] = []
    for archetype in range(4):
        folder = tmp_path / f"arch{archetype}"
        folder.mkdir()
        for deck in range(2):
            csv = folder / f"deck{deck}.csv"
            csv.write_text("\n".join("1" for _ in range(60)))
            paths.append(str(csv))
    idx = list(range(len(paths)))

    kept = _limit_pool_width(idx, paths, width=2, seed=0)
    archetypes = {Path(paths[i]).parent.name for i in kept}
    assert len(archetypes) == 2  # exactly the requested number of archetypes
    assert set(kept) <= set(idx)
    assert _limit_pool_width(idx, paths, width=2, seed=0) == kept  # deterministic


def test_limit_pool_width_rejects_below_one() -> None:
    """
    A width below one is rejected rather than silently emptying the pool.
    """
    with pytest.raises(ValueError, match="deck_pool_width"):
        _limit_pool_width([0], ["arch/a.csv"], width=0, seed=0)


@requires_corpus
def test_deck_pool_width_narrows_train_but_not_eval() -> None:
    """
    ``deck_pool_width`` shrinks the training pool while the held-out set is fixed.
    """
    full = _env_cfg(deck_pool=str(CORPUS_DIR), deck_holdout_frac=0.2)
    narrow = _env_cfg(deck_pool=str(CORPUS_DIR), deck_holdout_frac=0.2, deck_pool_width=5)
    assert (
        len(_build_sampler_spec(narrow, deck_split="train")["decks"])
        < len(_build_sampler_spec(full, deck_split="train")["decks"])
    )
    assert (
        len(_build_sampler_spec(narrow, deck_split="eval")["decks"])
        == len(_build_sampler_spec(full, deck_split="eval")["decks"])
    )


def test_deck_labels_fall_back_to_folder_name(tmp_path: Path) -> None:
    """
    Without a manifest, a deck is labelled by its parent (archetype) folder.
    """
    archetype_dir = tmp_path / "some-archetype"
    archetype_dir.mkdir()
    deck_file = archetype_dir / "d.csv"
    deck_file.write_text("\n".join("1" for _ in range(60)))
    assert _deck_labels([str(deck_file)]) == ["some-archetype"]


def test_record_winrate_parsing() -> None:
    """
    A ``"W-L-T"`` record parses to ``W/(W+L)``; unparseable/undecided is None.
    """
    assert _record_winrate("8-2-0") == 0.8
    assert _record_winrate("0-0-3") is None  # no decided games
    assert _record_winrate(None) is None
    assert _record_winrate("garbage") is None


@requires_corpus
def test_deck_weights_from_real_manifest() -> None:
    """
    Win-rate weighting over the real corpus yields one floored weight per deck.
    """
    paths = resolve_deck_paths(str(CORPUS_DIR))[:20]
    weights = _deck_weights(paths, "winrate")
    assert len(weights) == 20
    assert all(w >= 0.1 for w in weights)  # floor keeps every deck sampleable
    with pytest.raises(ValueError):
        _deck_weights(paths, "bogus")


@requires_corpus
def test_sampler_spec_zero_holdout_evaluates_on_full_pool() -> None:
    """
    With no holdout, train and eval both draw from the whole pool.
    """
    cfg = _env_cfg(deck_pool=str(CORPUS_DIR), deck_holdout_frac=0.0)
    train = _build_sampler_spec(cfg, deck_split="train")["decks"]
    holdout = _build_sampler_spec(cfg, deck_split="eval")["decks"]
    assert len(train) == len(holdout)  # both see the whole pool


@requires_corpus
def test_env_samples_different_decks_across_resets() -> None:
    """
    A pool-backed env draws a different deck across resets (per-episode sampling).
    """
    from src.env.tcg_env import TCGEnv

    paths = resolve_deck_paths(str(CORPUS_DIR))
    pool = load_decks(paths[:16])
    sampler = PoolDeckSampler(pool, matchup="mirror", mode="round_robin", seed=0)
    env = TCGEnv(max_options=96, seed=0, deck_sampler=sampler)
    try:
        seen = set()
        for _ in range(8):
            env.reset()
            seen.add(tuple(env._deck0))  # noqa: SLF001 - white-box check of sampled deck
        assert len(seen) > 1  # the curriculum rotates decks across episodes
    finally:
        env.close()
