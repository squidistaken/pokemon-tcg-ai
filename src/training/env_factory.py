import json
import random
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from torchrl.envs import EnvBase, TransformedEnv
from torchrl.envs.transforms import ActionMask

from cg.api import Observation
from src.env.deck import load_deck, resolve_deck_paths
from src.env.deck_sampler import build_deck_sampler
from src.env.observation_encoder import ObservationEncoder
from src.env.structured_observation_encoder import StructuredObservationEncoder
from src.env.tcg_env import TCGEnv

OpponentFactory = Callable[[], Callable[[Observation], list[int]]]


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
    deck_sampler = build_deck_sampler(sampler_spec, seed=seed + 2)
    base_env = TCGEnv(
        max_options=max_options,
        seed=seed,
        opponent=opponent,
        encoder=make_encoder(encoder, max_options),
        deck_sampler=deck_sampler,
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
                    manifest.update(json.loads(candidate.read_text()))
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


def _deck_weights(kept_paths: list[str], scheme: str) -> list[float]:
    """
    Compute per-deck sampling weights from manifest metadata.

    A small floor is added so every deck keeps a nonzero chance.

    :param kept_paths: Deck CSV paths, aligned with the decks being weighted.
    :param scheme: ``"winrate"`` (weight by record ``W/(W+L)``) or ``"placing"``
        (weight by ``1/placing``).
    :return: One weight per path.
    :raises ValueError: If the scheme is unknown.
    """
    if scheme not in ("winrate", "placing"):
        raise ValueError(
            f"unknown deck_weighting {scheme!r}; expected 'winrate' or 'placing'"
        )
    manifest = _load_manifest_for(kept_paths)
    floor = 0.1
    weights: list[float] = []
    for path in kept_paths:
        entry = manifest.get(Path(path).stem, {})
        if scheme == "winrate":
            score = _record_winrate(entry.get("record")) or 0.0
        else:
            placing = entry.get("placing")
            score = 1.0 / placing if isinstance(placing, int) and placing > 0 else 0.0
        weights.append(floor + score)
    return weights


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
    """
    pool_spec = cfg.env.get("deck_pool")
    if not pool_spec:
        return {
            "kind": "fixed",
            "deck0": load_deck(to_absolute_path(cfg.env.deck0)),
            "deck1": load_deck(to_absolute_path(cfg.env.deck1)),
        }

    entries = [pool_spec] if isinstance(pool_spec, str) else list(pool_spec)
    paths = resolve_deck_paths([to_absolute_path(entry) for entry in entries])
    if len(paths) < 2:
        raise ValueError(
            f"deck_pool {pool_spec!r} resolved to {len(paths)} deck(s); expected a "
            f"multi-deck corpus. Run ./scripts/fetch_decks.sh to install it."
        )
    decks, kept_paths = _load_pool_with_paths(paths)
    train_idx, holdout_idx = _split_indices(
        len(decks),
        holdout_frac=float(cfg.env.get("deck_holdout_frac", 0.0)),
        seed=int(cfg.env.get("deck_split_seed", 0)),
    )
    idx = holdout_idx if deck_split == "eval" else train_idx
    matchup = cfg.env.get("deck_matchup", "mirror")
    spec: dict[str, Any] = {
        "kind": "pool",
        "decks": [decks[i] for i in idx],
        "matchup": matchup,
        "mode": cfg.env.get("deck_sampling", "uniform"),
    }
    if deck_split == "eval":
        # Evaluation can/should use a different matchup than training.
        # ex. Training on `independent` (asymmetric) matchups is good for
        # robustness, but it makes the eval win-rate conflate piloting skill with deck luck.
        # Eval also stays uniform over the held-out pool (no mix/weighting), so
        # the generalization curve is an unbiased read across unseen decks.
        spec["matchup"] = cfg.env.get("eval_deck_matchup") or matchup
        return spec

    mirror_prob = cfg.env.get("deck_mirror_prob")
    if mirror_prob is not None:
        spec["mirror_prob"] = float(mirror_prob)
    weighting = cfg.env.get("deck_weighting")
    if weighting:
        spec["weights"] = _deck_weights([kept_paths[i] for i in idx], str(weighting))
    return spec


def make_env_factories(
    cfg: DictConfig,
    opponent_factory: OpponentFactory | None = None,
    deck_split: str = "train",
) -> list[Callable[[], EnvBase]]:
    """
    Build one environment factory per worker from the Hydra config.

    :param cfg: Hydra configuration with ``seed`` and an ``env`` section.
    :param opponent_factory: Opponent factory forwarded to every instance.
    :param deck_split: ``"train"`` (default) or ``"eval"``. Only affects runs
        with ``env.deck_pool`` set and a non-zero ``env.deck_holdout_frac``,
        where ``"eval"`` draws from the held-out, never-trained decks.
    :return: List of ``cfg.env.num_workers`` picklable environment factories.
    """
    sampler_spec = _build_sampler_spec(cfg, deck_split)
    encoder = cfg.env.get("encoder", "structured")
    return [
        partial(
            make_env,
            sampler_spec,
            cfg.env.max_options,
            cfg.seed + worker,
            opponent_factory,
            encoder,
        )
        for worker in range(cfg.env.num_workers)
    ]
