import logging
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from torchrl.data import Categorical, Composite

from cg.api import Observation
from src.env.observation.observation_encoder import ObservationEncoder
from src.env.opponents.external_snapshot_opponent_pool import (
    ExternalSnapshotOpponentPool,
    valid_snapshot_paths,
)
from src.env.opponents.pfsp_opponent_pool import PFSPOpponentPool
from src.env.opponents.random_opponent import RandomOpponent
from src.env.opponents.snapshot_opponent_pool import SnapshotOpponentPool
from src.policies.greedy_policy_opponent import (
    load_greedy_opponent,
)
from src.training.env_factory import OpponentFactory, make_encoder

logger = logging.getLogger(__name__)


def build_eval_opponent_factory(
    cfg: DictConfig,
    obs_spec: Composite | None = None,
    action_spec: Categorical | None = None,
    checkpoint_dir: str | Path | None = None,
    opponent: str | None = None,
) -> OpponentFactory:
    """
    Build the fixed reference opponent the evaluator scores against.

    The reference must not move with the learner, or the ``eval/`` win-rate
    stops being comparable across the run.

    Supported ``eval_opponent`` values:

    ``"random"``
        :class:`~src.env.opponents.random_opponent.RandomOpponent`, the default and the
        only one that does not re-read a checkpoint every evaluation.

    ``"first_snapshot"``
        The oldest snapshot in ``checkpoint_dir``, as a
        :class:`~src.policies.greedy_policy_opponent.GreedyPolicyOpponent`.
        Falls back to ``RandomOpponent`` before the league has one, so an early
        eval still produces a readable number.

    ``"checkpoint"``
        The snapshot named by ``train.eval_opponent_checkpoint``. A relative
        path resolves against the original working directory and must exist at
        startup.

    ``"checkpoint_pool"``
        A frozen population sampled uniformly from the newest checkpoints in
        ``train.eval_opponent_checkpoint_dir``, scanned once at construction so
        learner snapshots cannot drift it.

    ``/path/to/snapshot.pt``
        The same frozen reference named inline, which is what lets one run score
        itself against several checkpoints at once.

    :param cfg: Hydra config with a ``train`` section and a top-level ``seed``.
    :param obs_spec: Environment observation spec, needed to rebuild the
        network for snapshot-backed opponents.
    :param action_spec: Environment action spec, same reason.
    :param checkpoint_dir: Directory scanned for ``snapshot_*.pt`` files when
        the opponent is ``"first_snapshot"``.
    :param opponent: Opponent to build, overriding ``cfg.train.eval_opponent``.
        Set by callers scoring against several references at once.
    :return: Factory building the reference opponent.
    :raises ValueError: If the opponent name is unsupported, or a
        snapshot-backed reference is missing its path, its checkpoint
        directory, or the specs needed to rebuild its network.
    """
    name = (
        opponent
        if opponent is not None
        else str(cfg.train.get("eval_opponent", "random"))
    )
    if name == "random":
        return partial(RandomOpponent, seed=int(cfg.seed))

    if name == "first_snapshot":
        if checkpoint_dir is None:
            raise ValueError(
                "eval_opponent=first_snapshot requires checkpoint_dir to be passed."
            )
        if obs_spec is None or action_spec is None:
            raise ValueError(
                "eval_opponent=first_snapshot needs obs/action specs to rebuild its "
                "network; pass them to build_eval_opponent_factory."
            )
        return partial(
            _load_oldest_snapshot_opponent,
            checkpoint_dir=Path(checkpoint_dir),
            cfg=cfg,
            obs_spec=obs_spec,
            action_spec=action_spec,
            fallback_seed=int(cfg.seed),
        )

    if name == "checkpoint":
        if obs_spec is None or action_spec is None:
            raise ValueError(
                "A checkpoint eval opponent needs obs/action specs to rebuild its "
                "network; pass them to build_eval_opponent_factory."
            )
        return _checkpoint_opponent_factory(
            cfg,
            obs_spec,
            action_spec,
            checkpoint=cfg.train.get("eval_opponent_checkpoint"),
            missing_message=(
                "eval_opponent='checkpoint' requires train.eval_opponent_checkpoint "
                "to point at a saved snapshot."
            ),
            not_found_prefix="eval_opponent_checkpoint",
        )

    if name == "checkpoint_pool":
        if obs_spec is None or action_spec is None:
            raise ValueError(
                "eval_opponent=checkpoint_pool needs obs/action specs to rebuild "
                "its checkpoint policies."
            )
        configured = cfg.train.get("eval_opponent_checkpoint_dir") or cfg.train.get(
            "opponent_checkpoint_dir"
        )
        if not configured:
            raise ValueError(
                "eval_opponent=checkpoint_pool requires "
                "train.eval_opponent_checkpoint_dir or train.opponent_checkpoint_dir."
            )
        directory = Path(to_absolute_path(str(configured))).resolve()
        pool_size = int(
            cfg.train.get("eval_opponent_pool_size") or cfg.train.get("pool_size", 5)
        )
        _validate_external_checkpoint_dir(directory, pool_size)
        return partial(
            _make_external_pool,
            checkpoint_dirs=(directory,),
            cfg=cfg,
            obs_spec=obs_spec,
            action_spec=action_spec,
            pool_size=pool_size,
            seed=int(cfg.seed),
            refresh=False,
        )

    if name.endswith(".pt"):
        if obs_spec is None or action_spec is None:
            raise ValueError(
                "eval_opponent checkpoint path requires obs_spec and action_spec."
            )
        checkpoint_path = Path(name)
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"eval_opponent checkpoint not found: {checkpoint_path}"
            )
        return partial(
            _load_checkpoint_opponent,
            checkpoint_path=checkpoint_path,
            cfg=cfg,
            obs_spec=obs_spec,
            action_spec=action_spec,
        )

    raise ValueError(
        f"Unsupported eval_opponent '{name}'; expected 'random', 'first_snapshot', "
        f"'checkpoint', 'checkpoint_pool', or a path to a .pt checkpoint."
    )


def _load_oldest_snapshot_opponent(
    checkpoint_dir: Path,
    cfg: DictConfig,
    obs_spec: Composite,
    action_spec: Categorical,
    fallback_seed: int,
) -> Callable[[Observation], list[int]]:
    """
    Load the oldest snapshot from ``checkpoint_dir`` as a greedy opponent.

    Backs the ``first_snapshot`` reference, where "first" means lowest-frame,
    not most recent: the point is a yardstick frozen early enough to stay
    fixed for the whole run. The config value keeps its name so metric series
    stay comparable with runs already logged.

    Falls back to :class:`~src.env.opponents.random_opponent.RandomOpponent` when no
    snapshot exists yet, so the first eval interval that fires before the
    league has been snapshotted still produces a readable win rate.

    Module-level so the enclosing :func:`~functools.partial` stays picklable.

    :param checkpoint_dir: Directory to scan for ``snapshot_*.pt`` files.
    :param cfg: Hydra config for network reconstruction.
    :param obs_spec: Environment observation spec.
    :param action_spec: Environment action spec.
    :param fallback_seed: Seed for the fallback ``RandomOpponent``.
    :return: A greedy opponent or a random fallback.
    """
    snapshots = sorted(checkpoint_dir.glob("snapshot_*.pt"))
    if not snapshots:
        logger.warning(
            "eval_opponent=first_snapshot but no snapshots in %s; "
            "falling back to RandomOpponent.",
            checkpoint_dir,
        )
        return RandomOpponent(seed=fallback_seed)
    path = snapshots[0]
    logger.info("Eval opponent: first snapshot %s", path.name)
    encoder = make_encoder(
        str(cfg.env.get("encoder", "structured")), int(cfg.env.max_options)
    )
    return load_greedy_opponent(path, cfg, obs_spec, action_spec, encoder, device="cpu")


def _load_checkpoint_opponent(
    checkpoint_path: Path,
    cfg: DictConfig,
    obs_spec: Composite,
    action_spec: Categorical,
) -> Callable[[Observation], list[int]]:
    """
    Load a specific checkpoint as a greedy opponent.

    Module-level so the enclosing :func:`~functools.partial` stays picklable.

    :param checkpoint_path: Path to a ``save_actor_critic`` snapshot.
    :param cfg: Hydra config for network reconstruction.
    :param obs_spec: Environment observation spec.
    :param action_spec: Environment action spec.
    :return: A :class:`~src.policies.greedy_policy_opponent.GreedyPolicyOpponent`.
    """
    logger.info("Eval opponent: checkpoint %s", checkpoint_path)
    encoder = make_encoder(
        str(cfg.env.get("encoder", "structured")), int(cfg.env.max_options)
    )
    return load_greedy_opponent(
        checkpoint_path, cfg, obs_spec, action_spec, encoder, device="cpu"
    )


def build_best_response_opponent_factory(
    cfg: DictConfig,
    obs_spec: Composite,
    action_spec: Categorical,
) -> OpponentFactory:
    """
    Build the fixed opponent for a best-response (exploitability) run.

    Exploitability is measured by freezing a trained agent and training a fresh
    learner to beat it: the collection opponent every worker faces is that one
    frozen checkpoint, and the learner's eval win-rate against it is the
    exploitability signal.

    :param cfg: Hydra config with ``train.best_response_checkpoint`` and the
        ``env``/``model`` sections used to rebuild the frozen network.
    :param obs_spec: Environment observation composite spec.
    :param action_spec: Environment action spec.
    :return: A picklable opponent factory playing the frozen checkpoint.
    :raises ValueError: If the checkpoint path is missing or does not exist.
    """
    return _checkpoint_opponent_factory(
        cfg,
        obs_spec,
        action_spec,
        checkpoint=cfg.train.get("best_response_checkpoint"),
        missing_message=(
            "best_response requires train.best_response_checkpoint to point at the "
            "saved agent whose exploitability is being measured."
        ),
        not_found_prefix="best_response_checkpoint",
    )


def _checkpoint_opponent_factory(
    cfg: DictConfig,
    obs_spec: Composite,
    action_spec: Categorical,
    checkpoint: Any,
    missing_message: str,
    not_found_prefix: str,
) -> OpponentFactory:
    """
    Validate a config-supplied checkpoint path and build its opponent factory.

    :param cfg: Hydra config used to rebuild the matching architecture.
    :param obs_spec: Environment observation composite spec.
    :param action_spec: Environment action spec.
    :param checkpoint: The config value for the checkpoint path (unset/empty
        when the required override was not passed).
    :param missing_message: Error raised when ``checkpoint`` is unset.
    :param not_found_prefix: Prefix for the error when the path does not
        resolve to a file, naming which config key it came from.
    :return: A picklable opponent factory playing the frozen checkpoint.
    :raises ValueError: If the checkpoint path is missing or does not exist.
    """
    if not checkpoint:
        raise ValueError(missing_message)
    path = Path(to_absolute_path(str(checkpoint)))
    if not path.is_file():
        raise ValueError(f"{not_found_prefix} {path} does not exist.")
    return partial(
        _make_checkpoint_opponent,
        cfg=cfg,
        obs_spec=obs_spec,
        action_spec=action_spec,
        checkpoint_path=path,
    )


def _make_checkpoint_opponent(
    cfg: DictConfig,
    obs_spec: Composite,
    action_spec: Categorical,
    checkpoint_path: Path,
) -> Callable[[Observation], list[int]]:
    """
    Load a frozen snapshot as a greedy opponent (fixed eval reference or the
    probed agent in a best-response run).

    :param cfg: Hydra config used to rebuild the matching architecture.
    :param obs_spec: Environment observation composite spec.
    :param action_spec: Environment action spec.
    :param checkpoint_path: Path to a ``save_actor_critic`` snapshot.
    :return: A greedy opponent playing that snapshot on CPU.
    """
    encoder = make_encoder(
        cfg.env.get("encoder", "structured"), int(cfg.env.max_options)
    )
    return load_greedy_opponent(
        checkpoint_path, cfg, obs_spec, action_spec, encoder, device="cpu"
    )


def build_opponent_factory(
    cfg: DictConfig,
    obs_spec: Composite,
    action_spec: Categorical,
    checkpoint_dir: str | Path,
) -> OpponentFactory | None:
    """
    Build the self-play opponent factory described by ``cfg.train``.

    Returns None when snapshotting is disabled, which leaves the environments
    on their built-in :class:`~src.env.opponents.random_opponent.RandomOpponent` — the
    exact random-baseline path, with no self-play machinery in the way.

    The returned factory is a module-level :func:`~functools.partial` over
    picklable arguments only, because ``ParallelEnv`` pickles it into each
    worker process, where it is called once to build that worker's league.

    :param cfg: Hydra config with a ``train`` section (``snapshot_interval``,
        ``pool_size``, ``warmup_opponent``) and ``env``/``model`` sections used
        to rebuild snapshot networks.
    :param obs_spec: Environment observation composite spec.
    :param action_spec: Environment action spec.
    :param checkpoint_dir: Directory the trainer writes snapshots to and the
        workers scan.
    :return: An opponent factory, or None when self-play is disabled.
    :raises ValueError: If ``warmup_opponent`` is not a supported name.
    """
    mode = str(cfg.train.get("opponent_pool_mode", "selfplay"))
    if mode not in ("selfplay", "frozen", "refresh"):
        raise ValueError(
            f"unknown opponent_pool_mode {mode!r}; expected 'selfplay', "
            "'frozen' or 'refresh'"
        )
    if mode != "selfplay":
        return _build_external_opponent_factory(
            cfg, obs_spec, action_spec, checkpoint_dir, mode
        )

    # Compatibility path: everything below is the established self-play
    # construction, including snapshot_interval=0 and the random warmup anchor.
    if int(cfg.train.get("snapshot_interval", 0)) <= 0:
        return None
    warmup = str(cfg.train.get("warmup_opponent", "random"))
    if warmup != "random":
        raise ValueError(
            f"Unsupported warmup_opponent '{warmup}'; only 'random' is implemented. "
            f"Facing only past selves from step 0 needs a snapshot that does not exist yet."
        )
    warmup_checkpoint = cfg.train.get("warmup_checkpoint")
    if warmup_checkpoint:
        warmup_checkpoint = Path(to_absolute_path(str(warmup_checkpoint)))
        if not warmup_checkpoint.is_file():
            raise ValueError(f"warmup_checkpoint {warmup_checkpoint} does not exist.")
    else:
        warmup_checkpoint = None
    sampling = str(cfg.train.get("opponent_sampling", "uniform"))
    if sampling not in ("uniform", "pfsp"):
        raise ValueError(
            f"unknown opponent_sampling {sampling!r}; expected 'uniform' or 'pfsp'"
        )
    return partial(
        _make_pool,
        checkpoint_dir=Path(checkpoint_dir),
        cfg=cfg,
        obs_spec=obs_spec,
        action_spec=action_spec,
        pool_size=int(cfg.train.get("pool_size", 5)),
        seed=int(cfg.seed),
        sampling=sampling,
        warmup_checkpoint=warmup_checkpoint,
    )


def _build_external_opponent_factory(
    cfg: DictConfig,
    obs_spec: Composite,
    action_spec: Categorical,
    checkpoint_dir: str | Path,
    mode: str,
) -> OpponentFactory:
    """Build an opt-in frozen or learner-refreshed checkpoint population."""
    sampling = str(cfg.train.get("opponent_sampling", "uniform"))
    if sampling != "uniform":
        raise ValueError(
            f"opponent_pool_mode={mode} requires opponent_sampling=uniform; "
            f"got {sampling!r}. PFSP remains available in selfplay mode."
        )
    configured = cfg.train.get("opponent_checkpoint_dir")
    if not configured:
        raise ValueError(
            f"opponent_pool_mode={mode} requires train.opponent_checkpoint_dir."
        )
    baseline_dir = Path(to_absolute_path(str(configured))).resolve()
    pool_size = int(cfg.train.get("pool_size", 5))
    _validate_external_checkpoint_dir(baseline_dir, pool_size)

    refresh = mode == "refresh"
    if refresh and int(cfg.train.get("snapshot_interval", 0)) <= 0:
        raise ValueError(
            "opponent_pool_mode=refresh requires train.snapshot_interval > 0 so "
            "learner checkpoints can enter the opponent population."
        )
    directories = [baseline_dir]
    if refresh:
        learner_dir = Path(checkpoint_dir).resolve()
        if learner_dir == baseline_dir:
            raise ValueError(
                "The baseline opponent directory and learner checkpoint directory "
                "must be distinct in refresh mode."
            )
        directories.append(learner_dir)

    return partial(
        _make_external_pool,
        checkpoint_dirs=tuple(directories),
        cfg=cfg,
        obs_spec=obs_spec,
        action_spec=action_spec,
        pool_size=pool_size,
        seed=int(cfg.seed),
        refresh=refresh,
    )


def _validate_external_checkpoint_dir(directory: Path, pool_size: int) -> None:
    """Fail in the parent process before workers load an undersized population."""
    if pool_size <= 0:
        raise ValueError(f"pool_size must be positive, got {pool_size}")
    if not directory.is_dir():
        raise ValueError(f"opponent checkpoint directory {directory} does not exist.")
    snapshots = valid_snapshot_paths(directory)
    if len(snapshots) < pool_size:
        raise ValueError(
            f"opponent checkpoint directory {directory} contains {len(snapshots)} "
            f"snapshot(s); expected at least pool_size={pool_size}."
        )


def _make_pool(
    checkpoint_dir: Path,
    cfg: DictConfig,
    obs_spec: Composite,
    action_spec: Categorical,
    pool_size: int,
    seed: int,
    sampling: str = "uniform",
    warmup_checkpoint: Path | None = None,
) -> SnapshotOpponentPool:
    """
    Construct one worker's self-play league.

    Module-level (not a closure) so the enclosing partial stays picklable.

    :param checkpoint_dir: Directory scanned for learner snapshots.
    :param cfg: Hydra config used to rebuild snapshot networks.
    :param obs_spec: Environment observation composite spec.
    :param action_spec: Environment action spec.
    :param pool_size: Number of most-recent snapshots kept in the league.
    :param seed: Seed for the league's member sampler.
    :param sampling: ``"uniform"`` or ``"pfsp"``; selects the pool class.
    :param warmup_checkpoint: Snapshot loaded as the league's permanent anchor
        in place of the random warmup, or None for the random warmup.
    :return: The league for this worker.
    """
    # One encoder shared by every snapshot this worker loads. Encoders are
    # stateless between calls (their scratch buffers are cloned on the way out)
    # and only the pool's active member ever runs, so sharing is safe. Building
    # one per snapshot instead would allocate a fresh set of buffers per league
    # member, and would re-arm each encoder's one-shot truncation warning.
    encoder = make_encoder(
        cfg.env.get("encoder", "structured"), int(cfg.env.max_options)
    )
    if warmup_checkpoint is not None:
        # A stationary strong anchor replaces the random warmup: it gives PFSP
        # a fixed reference to concentrate on, which a random opponent stops
        # providing as soon as the learner beats it. Loaded through the shared
        # encoder, exactly like any other league snapshot.
        warmup_opponents = [
            _load_snapshot(warmup_checkpoint, cfg, obs_spec, action_spec, encoder)
        ]
    else:
        warmup_opponents = [RandomOpponent(seed=seed)]
    common = {
        "checkpoint_dir": checkpoint_dir,
        "load_snapshot": partial(
            _load_snapshot,
            cfg=cfg,
            obs_spec=obs_spec,
            action_spec=action_spec,
            encoder=encoder,
        ),
        "warmup_opponents": warmup_opponents,
        "pool_size": pool_size,
        "seed": seed,
    }
    if sampling == "uniform":
        return SnapshotOpponentPool(**common)
    return PFSPOpponentPool(
        **common,
        weighting=str(cfg.train.get("pfsp_weighting", "hard")),
        exponent=float(cfg.train.get("pfsp_exponent", 2.0)),
        min_weight=float(cfg.train.get("pfsp_min_weight", 0.05)),
        prior_games=float(cfg.train.get("pfsp_prior_games", 2.0)),
    )


def _make_external_pool(
    checkpoint_dirs: tuple[Path, ...],
    cfg: DictConfig,
    obs_spec: Composite,
    action_spec: Categorical,
    pool_size: int,
    seed: int,
    refresh: bool,
) -> ExternalSnapshotOpponentPool:
    """Construct one worker's external checkpoint population."""
    encoder = make_encoder(
        cfg.env.get("encoder", "structured"), int(cfg.env.max_options)
    )
    return ExternalSnapshotOpponentPool(
        checkpoint_dirs=checkpoint_dirs,
        load_snapshot=partial(
            _load_snapshot,
            cfg=cfg,
            obs_spec=obs_spec,
            action_spec=action_spec,
            encoder=encoder,
        ),
        pool_size=pool_size,
        refresh=refresh,
        seed=seed,
    )


def _load_snapshot(
    checkpoint_path: Path,
    cfg: DictConfig,
    obs_spec: Composite,
    action_spec: Categorical,
    encoder: ObservationEncoder,
) -> Callable[[Observation], list[int]]:
    """
    Load one snapshot into a greedy opponent inside the calling worker.

    :param checkpoint_path: Snapshot written by
        :func:`~src.policies.greedy_policy_opponent.save_actor_critic`.
    :param cfg: Hydra config used to rebuild the matching architecture.
    :param obs_spec: Environment observation composite spec.
    :param action_spec: Environment action spec.
    :param encoder: Observation encoder, shared with this worker's other
        league members; built inside the worker so nothing encoder-shaped has
        to cross the process boundary.
    :return: A league opponent playing that snapshot, under
        ``train.opponent_action_selection``.
    """
    # Opponents always infer on CPU: they run inside the env workers, which are
    # CPU-only, and a per-worker CUDA context would be far costlier than the
    # small MLP forward it would accelerate.
    return load_greedy_opponent(
        checkpoint_path,
        cfg,
        obs_spec,
        action_spec,
        encoder,
        device="cpu",
        action_selection=str(
            cfg.get("train", {}).get("opponent_action_selection", "greedy") or "greedy"
        ),
    )
