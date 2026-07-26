import logging
from collections.abc import Callable
from functools import partial
from pathlib import Path

from omegaconf import DictConfig
from torchrl.data import Categorical, Composite

from cg.api import Observation
from src.env.observation_encoder import ObservationEncoder
from src.env.random_opponent import RandomOpponent
from src.env.snapshot_opponent_pool import SnapshotOpponentPool
from src.policies.greedy_policy_opponent import load_greedy_opponent
from src.training.env_factory import OpponentFactory, make_encoder

logger = logging.getLogger(__name__)


def build_eval_opponent_factory(cfg: DictConfig) -> OpponentFactory:
    """
    Build the fixed reference opponent the evaluator scores against.

    The reference must not move with the learner, or the ``eval/`` win-rate
    stops being comparable across the run — that comparability is the whole
    point of evaluating separately from collection.

    :param cfg: Hydra config with a ``train`` section and a top-level ``seed``.
    :return: Factory building the reference opponent.
    :raises ValueError: If ``eval_opponent`` names an unsupported opponent.
    """
    name = str(cfg.train.get("eval_opponent", "random"))
    if name != "random":
        raise ValueError(
            f"Unsupported eval_opponent '{name}'; only 'random' is implemented. "
            f"A snapshot-backed reference needs a checkpoint to score against."
        )
    return partial(RandomOpponent, seed=int(cfg.seed))


def build_opponent_factory(
        cfg: DictConfig,
        obs_spec: Composite,
        action_spec: Categorical,
        checkpoint_dir: str | Path,
) -> OpponentFactory | None:
    """
    Build the self-play opponent factory described by ``cfg.train``.

    Returns None when snapshotting is disabled, which leaves the environments
    on their built-in :class:`~src.env.random_opponent.RandomOpponent` — the
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
    if int(cfg.train.get("snapshot_interval", 0)) <= 0:
        return None
    warmup = str(cfg.train.get("warmup_opponent", "random"))
    if warmup != "random":
        raise ValueError(
            f"Unsupported warmup_opponent '{warmup}'; only 'random' is implemented. "
            f"Facing only past selves from step 0 needs a snapshot that does not exist yet."
        )
    return partial(
        _make_pool,
        checkpoint_dir=Path(checkpoint_dir),
        cfg=cfg,
        obs_spec=obs_spec,
        action_spec=action_spec,
        pool_size=int(cfg.train.get("pool_size", 5)),
        seed=int(cfg.seed),
    )


def _make_pool(
        checkpoint_dir: Path,
        cfg: DictConfig,
        obs_spec: Composite,
        action_spec: Categorical,
        pool_size: int,
        seed: int,
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
    :return: The league for this worker.
    """
    # One encoder shared by every snapshot this worker loads. Encoders are
    # stateless between calls (their scratch buffers are cloned on the way out)
    # and only the pool's active member ever runs, so sharing is safe. Building
    # one per snapshot instead would allocate a fresh set of buffers per league
    # member, and would re-arm each encoder's one-shot truncation warning.
    encoder = make_encoder(cfg.env.get("encoder", "structured"), int(cfg.env.max_options))
    return SnapshotOpponentPool(
        checkpoint_dir=checkpoint_dir,
        load_snapshot=partial(
            _load_snapshot,
            cfg=cfg,
            obs_spec=obs_spec,
            action_spec=action_spec,
            encoder=encoder,
        ),
        warmup_opponents=[RandomOpponent(seed=seed)],
        pool_size=pool_size,
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
    :return: A greedy opponent playing that snapshot.
    """
    # Opponents always infer on CPU: they run inside the env workers, which are
    # CPU-only, and a per-worker CUDA context would be far costlier than the
    # small MLP forward it would accelerate.
    return load_greedy_opponent(
        checkpoint_path, cfg, obs_spec, action_spec, encoder, device="cpu"
    )
