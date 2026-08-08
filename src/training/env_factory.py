import json
import random
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from torchrl.data import Categorical, Composite
from torchrl.envs import EnvBase, TransformedEnv
from torchrl.envs.transforms import ActionMask

from cg.api import Observation
from src.env.deck import load_deck, resolve_deck_paths
from src.env.deck_sampler import build_deck_sampler
from src.env.observation_encoder import ObservationEncoder
from src.env.structured_observation_encoder import StructuredObservationEncoder
from src.env.tcg_env import TCGEnv

OpponentFactory = Callable[[], Callable[[Observation], list[int]]]

#: Floor added to the score-based ``deck_weighting`` schemes, so a deck the
#: manifest scores at zero still has a chance of being dealt.
_SCORE_WEIGHT_FLOOR = 0.1


def make_encoder(name: str, max_options: int) -> ObservationEncoder:
    """
    Build the observation encoder selected by name.

    :param name: Encoder name; ``structured`` is the only supported encoder.
    :param max_options: Padded option-space size of the environment.
    :return: A matching :class:`~src.env.observation_encoder.ObservationEncoder`.
    :raises ValueError: If the name is unknown.
    """
    if name == "structured":
        return StructuredObservationEncoder(max_options=max_options)
    raise ValueError(f"Unknown observation encoder '{name}'; expected 'structured'.")


def make_env(
    sampler_spec: dict[str, Any],
    max_options: int,
    seed: int,
    opponent_factory: OpponentFactory | None = None,
    encoder: str = "structured",
    deck_switch_steps: int = 0,
) -> EnvBase:
    """
    Build a single masked TCG environment instance.

    :param sampler_spec: Picklable deck-sampler spec (see
        :func:`~src.env.deck_sampler.build_deck_sampler`).
    :param max_options: Padded size of the option space.
    :param seed: Seed for this environment instance (and its deck sampler).
    :param opponent_factory: Builds the opponent for this instance (e.g. an
        OpponentPool for self-play); the default random opponent if None.
    :param encoder: Observation encoder name (see :func:`make_encoder`).
    :return: TransformedEnv with the ActionMask transform applied.
    """
    opponent = opponent_factory() if opponent_factory is not None else None
    deck_sampler = build_deck_sampler(sampler_spec, seed=seed)
    base_env = TCGEnv(
        max_options=max_options,
        seed=seed,
        opponent=opponent,
        encoder=make_encoder(encoder, max_options),
        deck_sampler=deck_sampler,
        deck_switch_steps=deck_switch_steps,
    )
    env = TransformedEnv(base_env, ActionMask())
    # TransformedEnv is created unlocked, and torchrl only caches its key lists
    # and step_mdp helper while the specs are locked. Left unlocked it rebuilds
    # all of them on every step, which measures as ~30% of env stepping time.
    # ActionMask mutates the mask in place rather than reassigning the spec, so
    # locking does not interfere with it.
    env.set_spec_lock_(True)
    return env


def _split_indices(
    pool_size: int, holdout_frac: float, seed: int
) -> tuple[list[int], list[int]]:
    """
    Deterministically split pool indices into train and holdout subsets.

    :param pool_size: Number of decks in the full pool.
    :param holdout_frac: Fraction of decks to reserve for evaluation ([0, 1)).
    :param seed: Seed for the deterministic shuffle.
    :return: ``(train_idx, holdout_idx)``. With ``holdout_frac == 0`` the holdout
        mirrors the full pool so evaluation still has decks to run on.
    """
    if not 0.0 <= holdout_frac < 1.0:
        raise ValueError(f"deck_holdout_frac must be in [0, 1), got {holdout_frac}")
    order = list(range(pool_size))
    random.Random(seed).shuffle(order)
    n_holdout = round(pool_size * holdout_frac)
    if holdout_frac > 0.0:
        # Always keep at least one deck on each side of the split when a nonzero
        # holdout is requested, so neither training nor eval is left empty.
        n_holdout = max(1, n_holdout)
    n_holdout = min(n_holdout, pool_size - 1) if pool_size > 1 else 0
    train_idx = order[n_holdout:]
    holdout_idx = order[:n_holdout] if n_holdout else list(range(pool_size))
    return train_idx, holdout_idx


def _load_pool_with_paths(paths: list[str]) -> tuple[list[list[int]], list[str]]:
    """
    Load every deck in ``paths``, keeping decks aligned to their source paths.

    :param paths: Deck CSV paths.
    :return: ``(decks, kept_paths)`` with invalid files skipped from both.
    :raises ValueError: If no valid deck could be loaded.
    """
    decks: list[list[int]] = []
    kept: list[str] = []
    for path in paths:
        try:
            decks.append(load_deck(path))
        except (ValueError, OSError):
            continue
        kept.append(path)
    if not decks:
        raise ValueError("no valid decks could be loaded from the given paths")
    return decks, kept


def _load_manifest_for(paths: list[str]) -> dict[str, dict]:
    """
    Load and merge the ``manifest.json`` sidecars covering ``paths``.

    Looks beside each deck so weighting works whether the pool points at
    ``decks`` or a single ``decks/<archetype>`` folder.

    :param paths: Deck CSV paths.
    :return: Merged ``stem -> metadata`` mapping; empty if no manifest is found.
    """
    manifest: dict[str, dict] = {}
    seen: set[Path] = set()
    for path in paths:
        deck_dir = Path(path).parent
        for candidate in (
            deck_dir / "manifest.json",
            deck_dir.parent / "manifest.json",
        ):
            if candidate in seen:
                continue
            seen.add(candidate)
            if candidate.exists():
                try:
                    data = json.loads(candidate.read_text())
                    decks = data.get("decks", data)
                    if isinstance(decks, dict):
                        manifest.update(decks)
                except (OSError, json.JSONDecodeError):
                    continue
    return manifest


def _record_winrate(record: str | None) -> float | None:
    """
    Parse a manifest ``record`` string (``"W-L-T"``) into a win-rate.

    :param record: Record string, or None.
    :return: ``W / (W + L)``, or None when unparseable or no games decided.
    """
    if not record:
        return None
    try:
        wins, losses, _ties = (int(part) for part in record.split("-"))
    except (ValueError, AttributeError):
        return None
    decided = wins + losses
    return wins / decided if decided > 0 else None


def _deck_labels(kept_paths: list[str]) -> list[str]:
    """
    Resolve each deck to its archetype label for per-archetype evaluation.

    :param kept_paths: Deck CSV paths, aligned with the decks being labelled.
    :return: One archetype label per path.
    """
    manifest = _load_manifest_for(kept_paths)
    labels: list[str] = []
    for path in kept_paths:
        entry = manifest.get(Path(path).stem, {})
        archetype = entry.get("archetype")
        labels.append(str(archetype) if archetype else Path(path).parent.name)
    return labels


def archetype_observations(kept_paths: list[str]) -> dict[str, float]:
    """
    Total manifest observations per archetype.

    The manifest counts observations per *list*; an archetype's total is the sum
    over its lists. This is the only place that summing happens -- the sampling
    weights in :func:`_deck_weights` stay strictly per-list.

    :param kept_paths: Deck CSV paths to aggregate over.
    :return: Summed observation count per archetype label.
    """
    manifest = _load_manifest_for(kept_paths)
    totals: dict[str, float] = {}
    for path in kept_paths:
        entry = manifest.get(Path(path).stem, {})
        archetype = entry.get("archetype")
        label = str(archetype) if archetype else Path(path).parent.name
        totals[label] = totals.get(label, 0.0) + _observation_weight(
            entry.get("observation_count")
        )
    return totals


def _limit_pool_width(
    idx: list[int],
    kept_paths: list[str],
    width: int,
    seed: int,
    selection: str = "observation",
) -> list[int]:
    """
    Restrict training-deck indices to a deterministic subset of archetypes.

    :param idx: Candidate deck indices (the training split).
    :param kept_paths: Deck paths aligned with the full pool, indexed by ``idx``.
    :param width: Number of archetypes to keep.
    :param seed: Seed for the deterministic archetype choice, used by the
        ``"random"`` selection only.
    :param selection: ``"observation"`` keeps the ``width`` most-observed
        archetypes; ``"random"`` keeps a seeded random subset.
    :return: The subset of ``idx`` whose decks belong to the chosen archetypes.
    :raises ValueError: If ``width`` is below 1 or ``selection`` is unknown.
    """
    if width < 1:
        raise ValueError(f"deck_pool_width must be >= 1, got {width}")
    if selection not in ("observation", "random"):
        raise ValueError(
            f"unknown deck_pool_selection {selection!r}; expected 'observation' "
            "or 'random'"
        )
    kept = [kept_paths[i] for i in idx]
    labels = _deck_labels(kept)
    archetypes = sorted(set(labels))
    if selection == "random":
        random.Random(seed).shuffle(archetypes)
    else:
        # Most-observed first, ties broken by name so the choice is reproducible
        # and the pools stay nested as width grows.
        observations = archetype_observations(kept)
        archetypes.sort(key=lambda name: (-observations.get(name, 0.0), name))
    chosen = set(archetypes[:width])
    return [i for i, label in zip(idx, labels, strict=True) if label in chosen]


def _deck_weights(kept_paths: list[str], scheme: str) -> list[float]:
    """
    Compute per-deck sampling weights from manifest metadata.

    :param kept_paths: Deck CSV paths, aligned with the decks being weighted.
    :param scheme: ``"observation"`` (weight by the manifest's
        ``observation_count``, so a list is dealt in proportion to how often it
        was actually played), ``"winrate"`` (weight by record ``W/(W+L)``) or
        ``"placing"`` (weight by ``1/placing``).
    :return: One weight per path.
    :raises ValueError: If the scheme is unknown.
    """
    if scheme not in ("observation", "winrate", "placing"):
        raise ValueError(
            f"unknown deck_weighting {scheme!r}; expected 'observation', "
            "'winrate' or 'placing'"
        )
    manifest = _load_manifest_for(kept_paths)
    weights: list[float] = []
    for path in kept_paths:
        entry = manifest.get(Path(path).stem, {})
        if scheme == "observation":
            weights.append(_observation_weight(entry.get("observation_count")))
            continue
        if scheme == "winrate":
            score = _record_winrate(entry.get("record")) or 0.0
        else:
            placing = entry.get("placing")
            score = 1.0 / placing if isinstance(placing, int) and placing > 0 else 0.0
        # A small floor so a deck the manifest scores at zero stays sampleable.
        weights.append(_SCORE_WEIGHT_FLOOR + score)
    return weights


def _observation_weight(observation_count: object) -> float:
    """
    Turn a manifest ``observation_count`` into a sampling weight.

    The count is used directly rather than floored onto a score, so the draw is
    exactly proportional to how many times the list was observed. A deck the
    manifest does not cover falls back to a single observation, which is the
    weight the corpus's most common entry carries anyway -- 76.7% of lists in
    ``heuristic-resolved`` were seen exactly once -- rather than a value that
    would rank an unknown list above or below the bulk of known ones.

    :param observation_count: The manifest's count, of unknown type.
    :return: The weight, at least 1.0.
    """
    if isinstance(observation_count, int) and observation_count > 0:
        return float(observation_count)
    return 1.0


def load_deck_pool(
    cfg: DictConfig, deck_split: str | None = None
) -> tuple[list[list[int]], list[str]]:
    """
    Resolve and load the configured deck pool, optionally one split of it.

    Shared by the environment factories and by the curriculum, which needs the
    same paths to derive archetypes from their folders. Both must ask for the
    same split and receive it in the same order, since the curriculum indexes
    decks by position.

    ``env.deck_pool_width`` is applied here, to the train split only, for that
    same reason: it used to be applied by :func:`_build_sampler_spec` alone, so
    a curriculum run -- which builds its own spec and its archetype index from
    this function -- silently trained on the full-width pool no matter what the
    width was set to. That is the one knob that shrinks the matchup space
    enough for levels to accumulate episodes, so ignoring it left the
    curriculum surveying a space it could never measure.

    :param cfg: Hydra configuration with an ``env.deck_pool`` entry.
    :param deck_split: ``"train"`` or ``"eval"`` to apply the holdout split,
        or None for the whole pool (which no width cap applies to either).
    :return: ``(decks, paths)``, aligned, with unreadable files dropped.
    :raises ValueError: If no pool is configured, it resolves to under two
        decks, or ``deck_split`` is unknown.
    """
    if deck_split not in (None, "train", "eval"):
        raise ValueError(
            f"unknown deck_split {deck_split!r}; expected 'train', 'eval' or None"
        )
    pool_spec = cfg.env.get("deck_pool")
    if not pool_spec:
        raise ValueError(
            "env.deck_pool must be set to load a deck pool; the fixed deck0/deck1 "
            "pair has no pool to resolve."
        )
    entries = [pool_spec] if isinstance(pool_spec, str) else list(pool_spec)
    paths = resolve_deck_paths([to_absolute_path(entry) for entry in entries])
    if len(paths) < 2:
        raise ValueError(
            f"deck_pool {pool_spec!r} resolved to {len(paths)} deck(s); expected a "
            f"multi-deck corpus. Run ./scripts/fetch_decks.sh to install it."
        )
    decks, kept_paths = _load_pool_with_paths(paths)
    if deck_split is None:
        return decks, kept_paths
    split_seed = int(cfg.env.get("deck_split_seed", 0))
    train_idx, holdout_idx = _split_indices(
        len(decks),
        holdout_frac=float(cfg.env.get("deck_holdout_frac", 0.0)),
        seed=split_seed,
    )
    width = cfg.env.get("deck_pool_width")
    if width is not None:
        # The held-out eval set stays fixed so a width sweep varies training
        # diversity against a constant yardstick.
        train_idx = _limit_pool_width(
            train_idx,
            kept_paths,
            int(width),
            split_seed,
            str(cfg.env.get("deck_pool_selection", "observation")),
        )
    if deck_split == "train":
        return [decks[i] for i in train_idx], [kept_paths[i] for i in train_idx]
    panel = cfg.env.get("eval_panel_size")
    if panel is not None:
        holdout_idx = _eval_panel(holdout_idx, kept_paths, int(panel))
    return [decks[i] for i in holdout_idx], [kept_paths[i] for i in holdout_idx]


def _eval_panel(
    holdout_idx: list[int], kept_paths: list[str], panel_size: int
) -> list[int]:
    """
    Cut the held-out pool down to a fixed panel of opponents.

    Evaluation otherwise deals a different held-out list every episode, so a
    50-episode round is 50 matchups played once each and the deck draw swamps
    the policy signal it is meant to measure. A fixed panel, drawn round-robin,
    plays the *same* matchups every round: the curve becomes comparable across
    rounds, and each opponent accumulates enough episodes to be read on its own.

    The panel is the most-observed held-out lists, so it is the field the agent
    is most likely to actually meet, and it is deterministic given the corpus.

    :param holdout_idx: Held-out deck indices.
    :param kept_paths: Deck paths aligned with the full pool.
    :param panel_size: Number of opponents to keep.
    :return: The panel's indices, ordered most-observed first.
    :raises ValueError: If ``panel_size`` is below one.
    """
    if panel_size < 1:
        raise ValueError(f"eval_panel_size must be >= 1, got {panel_size}")
    manifest = _load_manifest_for([kept_paths[i] for i in holdout_idx])
    ranked = sorted(
        holdout_idx,
        key=lambda index: (
            -_observation_weight(
                manifest.get(Path(kept_paths[index]).stem, {}).get("observation_count")
            ),
            kept_paths[index],
        ),
    )
    return ranked[:panel_size]


def _build_sampler_spec(cfg: DictConfig, deck_split: str) -> dict[str, Any]:
    """
    Build the picklable deck-sampler spec for the given split from the config.

    When ``cfg.env.deck_pool`` is unset the environment keeps its original
    single-matchup behaviour. When it is set, decks are loaded from the pool,
    split into train/holdout, and the requested subset is wrapped in a pool sampler.

    :param cfg: Hydra configuration with an ``env`` section and top-level seed.
    :param deck_split: ``"train"`` or ``"eval"`` — which subset the sampler draws
        from when a pool is configured.
    :return: Spec dict consumed by :func:`~src.env.deck_sampler.build_deck_sampler`.
    :raises ValueError: If ``env.agent_deck`` is set without ``env.deck_pool``,
        which would otherwise drop the pin silently.
    """
    pool_spec = cfg.env.get("deck_pool")
    if not pool_spec:
        if cfg.env.get("agent_deck"):
            raise ValueError(
                "env.agent_deck needs env.deck_pool: the pin only chooses which "
                "deck the agent takes, and the opposing field it leaves at full "
                "width is drawn from the pool. Without one there is no field to "
                "draw, and env.deck0/deck1 already fix both seats. Set a deck "
                "pool, or drop the pin and set env.deck0 instead."
            )
        return {
            "kind": "fixed",
            "deck0": load_deck(to_absolute_path(cfg.env.deck0)),
            "deck1": load_deck(to_absolute_path(cfg.env.deck1)),
        }

    # The split, the width cap and the eval panel all live in load_deck_pool, so
    # this spec and the curriculum's -- which calls it directly -- describe the
    # same pool.
    decks, kept_paths = load_deck_pool(cfg, deck_split=deck_split)
    matchup = cfg.env.get("deck_matchup", "mirror")
    spec: dict[str, Any] = {
        "kind": "pool",
        "decks": decks,
        "matchup": matchup,
        "mode": cfg.env.get("deck_sampling", "uniform"),
    }
    if deck_split == "eval":
        # Evaluation can/should use a different matchup than training.
        # ex. Training on `independent` (asymmetric) matchups is good for
        # robustness, but it makes the eval win-rate conflate piloting skill with deck luck.
        # Eval stays unweighted: with eval_panel_size the panel already fixes
        # which opponents appear, and weighting them on top would only vary how
        # often each is drawn -- round_robin gives every panel entry the same
        # count, which is what makes the rounds comparable.
        spec["matchup"] = cfg.env.get("eval_deck_matchup") or matchup
        # Draw strategy for eval, independent of training's. round_robin gives
        # deterministic even coverage of the held-out pool, so the per-archetype
        # breakdown is not at the mercy of which decks a uniform draw happened to
        # hit that round; null inherits deck_sampling.
        spec["mode"] = cfg.env.get("eval_deck_sampling") or spec["mode"]
        # Carry archetype labels so the evaluator can break the held-out win-rate
        # down per archetype and report generalization variance across them.
        spec["labels"] = _deck_labels(kept_paths)
        return _pin_agent_deck(cfg, spec, deck_split)

    mirror_prob = cfg.env.get("deck_mirror_prob")
    if mirror_prob is not None:
        spec["mirror_prob"] = float(mirror_prob)
    weighting = cfg.env.get("deck_weighting")
    if weighting:
        spec["weights"] = _deck_weights(kept_paths, str(weighting))
    return _pin_agent_deck(cfg, spec, deck_split)


def _pin_agent_deck(
    cfg: DictConfig, field_spec: dict[str, Any], deck_split: str
) -> dict[str, Any]:
    """
    Wrap a field spec so the agent always pilots a fixed deck.

    Evaluation reads ``env.eval_agent_deck`` and training ``env.agent_deck``, so
    the two can be set independently. That separation is the point: evaluation
    *should* hold the agent's deck fixed, because the curve is meant to measure
    one deck against a field the way a submission does, while training generally
    should not.

    Pinning the agent's deck during training makes the self-play opponent worse
    at the game. The opponent is a snapshot of the same network, and it is dealt
    field decks -- decks the pinned network barely practises -- so it misplays
    them, and the learner trains against an opponent it has itself crippled. The
    effect grows as the pin succeeds. ``agent_deck_field_prob`` softens it but
    cannot remove it. Leave ``env.agent_deck`` null and specialise, if at all,
    as a short fine-tune once the league is already strong.

    :param cfg: Hydra configuration with an ``env`` section.
    :param field_spec: The opposing-field sampler spec to wrap.
    :param deck_split: ``"train"`` or ``"eval"``.
    :return: The wrapped spec, or ``field_spec`` unchanged when no deck is set.
    """
    key = "eval_agent_deck" if deck_split == "eval" else "agent_deck"
    configured = cfg.env.get(key)
    if not configured:
        return field_spec
    path = Path(to_absolute_path(str(configured)))
    if not path.is_file():
        raise ValueError(f"env.{key} {path} does not exist.")
    field_probability = (
        0.0
        if deck_split == "eval"
        else float(cfg.env.get("agent_deck_field_prob", 0.0) or 0.0)
    )
    return {
        "kind": "agent_fixed",
        "agent_deck": load_deck(str(path)),
        "agent_label": _deck_labels([str(path)])[0],
        "field_probability": field_probability,
        "field": field_spec,
    }


def make_env_factories(
    cfg: DictConfig,
    opponent_factory: OpponentFactory | None = None,
    deck_split: str = "train",
    curriculum: Any = None,
    sampler_spec: dict[str, Any] | None = None,
    seed_offset: int = 0,
) -> list[Callable[[], EnvBase]]:
    """
    Build one environment factory per worker from the Hydra config.

    :param cfg: Hydra configuration with ``seed`` and an ``env`` section.
    :param opponent_factory: Opponent factory forwarded to every instance.
    :param deck_split: ``"train"`` (default) or ``"eval"``. Only affects runs
        with ``env.deck_pool`` set and a non-zero ``env.deck_holdout_frac``,
        where ``"eval"`` draws from the held-out, never-trained decks.
    :param curriculum: A :class:`~src.training.curriculum.Curriculum` whose
        published distribution the environments draw matchups from. None keeps
        the configured deck sampler. Ignored for the ``"eval"`` split, which
        must stay an unbiased read across held-out decks rather than following
        the training distribution.
    :param sampler_spec: Precomputed spec from :func:`_build_sampler_spec`, for
        a caller that already built one for this exact ``deck_split``.
        Built fresh when ``None``. Ignored when a ``curriculum`` drives the
        train split, which supplies its own spec.
    :param seed_offset: Added to every worker's seed. Non-zero only when a
        replacement set of factories is built for a restarted collector, where
        reusing the original seeds would re-deal the same sequence of matchups,
        seat assignments and opponent draws each worker already played.
    :return: List of ``cfg.env.num_workers`` picklable environment factories.
    """
    if curriculum is not None and cfg.env.get("agent_deck"):
        raise ValueError(
            "env.agent_deck and env.curriculum.enabled cannot both be set: a "
            "curriculum level is an ordered (agent, opponent) archetype pair, and "
            "pinning the agent's deck discards the agent half of every level it "
            "draws, so its scores would describe matchups that were never played. "
            "Pin the deck and leave the curriculum off, or drop the pin."
        )
    if curriculum is not None and deck_split != "eval":
        # The train split specifically, and via the same call the curriculum
        # used to build its archetype index: it addresses decks by position, so
        # a different subset or order here would silently mis-deal every level.
        # Loading the whole pool would also train on the held-out decks the
        # evaluator scores generalization against.
        decks, paths = load_deck_pool(cfg, deck_split="train")
        sampler_spec = {
            "kind": "curriculum",
            "decks": decks,
            "archetypes": curriculum.archetypes,
            "handles": curriculum.handles,
            "explore_prob": curriculum.explore_prob,
        }
        # The curriculum curates the archetype *pairing*; the list dealt from
        # within each archetype is still a pool draw, so it honours
        # deck_weighting exactly as the uncurated sampler does. Without this the
        # weighting is silently discarded whenever the curriculum is enabled.
        weighting = cfg.env.get("deck_weighting")
        if weighting:
            sampler_spec["weights"] = _deck_weights(paths, str(weighting))
    elif sampler_spec is None:
        sampler_spec = _build_sampler_spec(cfg, deck_split)
    encoder = cfg.env.get("encoder", "structured")
    deck_switch_steps = int(cfg.env.get("deck_switch_steps", 0))
    return [
        partial(
            make_env,
            sampler_spec,
            cfg.env.max_options,
            cfg.seed + seed_offset + worker,
            opponent_factory,
            encoder,
            deck_switch_steps,
        )
        for worker in range(cfg.env.num_workers)
    ]


def build_probe_specs(cfg: DictConfig) -> tuple[Composite, Categorical]:
    """
    Derive the observation/action specs from a throwaway environment instance.

    :param cfg: Hydra configuration with ``seed`` and an ``env`` section.
    :return: The environment's observation and action specs.
    """
    probe_env = make_env_factories(cfg)[0]()
    try:
        obs_spec = probe_env.observation_spec
        action_spec = probe_env.action_spec
    finally:
        probe_env.close()
    assert isinstance(action_spec, Categorical), "env action spec must be Categorical"
    return obs_spec, action_spec
