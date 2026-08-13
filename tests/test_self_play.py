import json
import pickle
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from src.checkpoint_registry import read_checkpoint_records
from src.env.opponents.random_opponent import RandomOpponent
from src.env.opponents.snapshot_opponent_pool import SnapshotOpponentPool
from src.policies.greedy_policy_opponent import GreedyPolicyOpponent, save_actor_critic
from src.policies.ppo_actor import build_actor_critic, build_ppo_actor_critic
from src.training.callbacks import SnapshotCallback
from src.training.env_factory import make_env_factories
from src.training.evaluator import Evaluator
from src.training.self_play import (
    build_best_response_opponent_factory,
    build_eval_opponent_factory,
    build_opponent_factory,
)
from tests.conftest import structured_env_cfg


def fake_loader(path: Path) -> RandomOpponent:  # noqa: ARG001
    """
    Stand-in for snapshot loading that ignores the file's contents.

    Lets the league's discovery/eviction logic be tested without paying for a
    real network rebuild per file.

    :param path: Snapshot path (unused).
    :return: A fresh random opponent standing in for the snapshot.
    """
    return RandomOpponent(seed=0)


def write_snapshot_files(directory: Path, frame_counts: list[int]) -> list[Path]:
    """
    Create placeholder snapshot files named the way the callback names them.

    :param directory: Directory to create the files in.
    :param frame_counts: Frame count to embed in each filename.
    :return: The created paths, in creation order.
    """
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for frames in frame_counts:
        path = directory / f"snapshot_{frames:012d}.pt"
        path.write_bytes(b"placeholder")
        paths.append(path)
    return paths


def selfplay_cfg(tmp_path: Path, structured_model_cfg, snapshot_interval: int = 100):
    """
    Build a full config exercising the self-play code path.

    :param tmp_path: Directory used for snapshots.
    :param structured_model_cfg: Model section from the shared fixture.
    :param snapshot_interval: Frames between snapshots; 0 disables self-play.
    :return: Merged config with ``env``, ``model`` and ``train`` sections.
    """
    cfg = OmegaConf.merge(
        structured_env_cfg(num_workers=1),
        structured_model_cfg,
        OmegaConf.create(
            {
                "train": {
                    "warmup_opponent": "random",
                    "snapshot_interval": snapshot_interval,
                    "checkpoint_dir": str(tmp_path),
                    "pool_size": 3,
                }
            }
        ),
    )
    return cfg


def test_pool_faces_only_warmup_until_a_snapshot_appears(tmp_path) -> None:
    """
    An empty checkpoint directory leaves the league at its warmup opponent, so
    early training has a real signal instead of erroring on a missing snapshot.
    """
    pool = SnapshotOpponentPool(
        checkpoint_dir=tmp_path / "missing",
        load_snapshot=fake_loader,
        warmup_opponents=[RandomOpponent(seed=0)],
        pool_size=3,
        seed=0,
    )
    pool.on_reset()
    assert pool.snapshot_count == 0
    assert isinstance(pool.active, RandomOpponent)


def test_pool_discovers_snapshots_written_after_construction(tmp_path) -> None:
    """
    The league picks up snapshots that appear after it was built — the whole
    point of the disk channel, since ParallelEnv workers cannot be handed new
    objects mid-run.
    """
    pool = SnapshotOpponentPool(
        checkpoint_dir=tmp_path,
        load_snapshot=fake_loader,
        warmup_opponents=[RandomOpponent(seed=0)],
        pool_size=3,
        seed=0,
    )
    pool.on_reset()
    assert pool.snapshot_count == 0

    write_snapshot_files(tmp_path, [100, 200])
    pool.on_reset()
    assert pool.snapshot_count == 2


def test_pool_keeps_only_the_newest_snapshots(tmp_path) -> None:
    """
    The league is capped at ``pool_size`` snapshots, evicting the oldest, while
    the warmup opponent survives as a permanent fixed reference.
    """
    pool = SnapshotOpponentPool(
        checkpoint_dir=tmp_path,
        load_snapshot=fake_loader,
        warmup_opponents=[RandomOpponent(seed=0)],
        pool_size=2,
        seed=0,
    )
    write_snapshot_files(tmp_path, [100, 200, 300, 400])
    pool.on_reset()
    assert pool.snapshot_count == 2


def test_pool_skips_unreadable_snapshot_and_retries_later(tmp_path) -> None:
    """
    A snapshot caught mid-write is skipped rather than crashing the episode,
    and is picked up on a later scan once it loads.
    """
    failing_paths = {tmp_path / "snapshot_000000000100.pt"}

    def flaky_loader(path: Path) -> RandomOpponent:
        """
        Fail for paths still marked unreadable, succeed otherwise.

        :param path: Snapshot path.
        :return: A random opponent standing in for the snapshot.
        """
        if path in failing_paths:
            raise OSError("snapshot is mid-write")
        return RandomOpponent(seed=0)

    write_snapshot_files(tmp_path, [100])
    pool = SnapshotOpponentPool(
        checkpoint_dir=tmp_path,
        load_snapshot=flaky_loader,
        warmup_opponents=[RandomOpponent(seed=0)],
        pool_size=3,
        seed=0,
    )
    pool.on_reset()
    assert pool.snapshot_count == 0

    failing_paths.clear()
    pool.on_reset()
    assert pool.snapshot_count == 1


def test_snapshot_callback_writes_on_interval(
    tmp_path, structured_model_cfg, structured_obs_spec, action_spec
) -> None:
    """
    Snapshots are written once per interval, leaving no partial ``.tmp`` files
    behind for a scanning worker to trip over.
    """
    actor_critic = build_actor_critic(
        structured_model_cfg, structured_obs_spec, action_spec
    )
    callback = SnapshotCallback(actor_critic, tmp_path, interval=100)
    callback.on_train_start({})
    for frames in (50, 100, 150, 200):
        callback.on_rollout_end(frames, {})

    assert sorted(path.name for path in tmp_path.glob("*.pt")) == [
        "snapshot_000000000100.pt",
        "snapshot_000000000200.pt",
    ]
    assert list(tmp_path.glob("*.tmp")) == []


def test_snapshot_callback_does_not_rewrite_final_snapshot(
    tmp_path, structured_model_cfg, structured_obs_spec, action_spec
) -> None:
    """
    A run ending exactly on a snapshot boundary does not write that snapshot
    twice; a run ending between boundaries still persists its final policy.
    """
    actor_critic = build_actor_critic(
        structured_model_cfg, structured_obs_spec, action_spec
    )
    registry = tmp_path / "logs" / "checkpoint_keys.csv"
    logged: list[tuple[Path, str, int]] = []
    callback = SnapshotCallback(
        actor_critic,
        tmp_path / "checkpoints",
        interval=100,
        checkpoint_loggers=[
            lambda path, digest, frames: logged.append((path, digest, frames))
        ],
        registry_path=registry,
        repo_root=tmp_path,
    )
    callback.on_rollout_end(100, {})
    assert logged == []
    assert not registry.exists()

    callback.on_train_end({"frames": 100})
    assert len(list((tmp_path / "checkpoints").glob("*.pt"))) == 1
    assert len(logged) == 1
    records = read_checkpoint_records(registry)
    assert len(records) == 1
    assert records[0].frames == 100

    callback.on_train_end({"frames": 100})
    assert len(logged) == 1
    assert len(read_checkpoint_records(registry)) == 1


def test_registry_failure_does_not_block_checkpoint_loggers(
    tmp_path,
    structured_model_cfg,
    structured_obs_spec,
    action_spec,
    monkeypatch,
    caplog,
) -> None:
    """A local registry failure does not prevent later publication callbacks."""
    actor_critic = build_actor_critic(
        structured_model_cfg, structured_obs_spec, action_spec
    )
    logged: list[tuple[Path, str, int]] = []
    callback = SnapshotCallback(
        actor_critic,
        tmp_path / "checkpoints",
        interval=0,
        checkpoint_loggers=[
            lambda path, digest, frames: logged.append((path, digest, frames))
        ],
        registry_path=tmp_path / "logs" / "checkpoint_keys.csv",
        repo_root=tmp_path,
    )

    def fail_registry_append(*args, **kwargs) -> None:  # noqa: ARG001
        raise OSError("registry unavailable")

    monkeypatch.setattr(
        "src.training.callbacks.snapshot_callback.append_checkpoint_record",
        fail_registry_append,
    )

    with caplog.at_level("ERROR"):
        callback.on_train_end({"frames": 123})

    assert len(logged) == 1
    assert logged[0][0].name == "snapshot_000000000123.pt"
    assert logged[0][2] == 123
    assert "continuing with remaining callbacks" in caplog.text


def test_final_checkpoint_embeds_config_and_emits_hash_key(
    tmp_path,
    structured_model_cfg,
    structured_obs_spec,
    action_spec,
    capsys,
) -> None:
    """Even interval=0 writes a self-describing final checkpoint and short key."""
    actor_critic = build_actor_critic(
        structured_model_cfg, structured_obs_spec, action_spec
    )
    logged: list[tuple[Path, str, int]] = []
    callback = SnapshotCallback(
        actor_critic,
        tmp_path / "checkpoints",
        interval=0,
        checkpoint_loggers=[
            lambda path, digest, frames: logged.append((path, digest, frames))
        ],
        registry_path=tmp_path / "logs" / "checkpoint_keys.csv",
        repo_root=tmp_path,
    )
    run_config = OmegaConf.to_container(structured_model_cfg, resolve=True)
    assert isinstance(run_config, dict)
    run_config["env"] = {
        "encoder": "structured",
        "max_options": 96,
        "deck0": "decks/example.csv",
    }
    callback.on_train_start(run_config)
    callback.on_train_end({"frames": 123})

    checkpoint = tmp_path / "checkpoints" / "snapshot_000000000123.pt"
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert payload["format_version"] == 1
    assert payload["frames"] == 123
    assert payload["config"]["env"]["max_options"] == 96
    metadata = json.loads(checkpoint.with_suffix(".json").read_text())
    assert metadata["sha256"] == logged[0][1]
    assert metadata["key"] == logged[0][1][:12]
    assert logged[0][0] == checkpoint
    assert logged[0][2] == 123
    record = read_checkpoint_records(tmp_path / "logs" / "checkpoint_keys.csv")[-1]
    assert record.frames == 123
    assert record.checkpoint_path == "checkpoints/snapshot_000000000123.pt"
    assert record.sha256 == metadata["sha256"]
    assert record.source == "training"
    output = capsys.readouterr().out
    assert f"checkpoint: {checkpoint}" in output
    assert f"checkpoint-key: {metadata['key']}" in output


def test_disabled_snapshotting_yields_no_opponent_factory(
    tmp_path, structured_model_cfg, structured_obs_spec, action_spec
) -> None:
    """
    ``snapshot_interval: 0`` leaves the environments on their built-in random
    opponent, keeping the baseline path free of self-play machinery.
    """
    cfg = selfplay_cfg(tmp_path, structured_model_cfg, snapshot_interval=0)
    factory = build_opponent_factory(cfg, structured_obs_spec, action_spec, tmp_path)
    assert factory is None


def test_missing_opponent_pool_mode_keeps_legacy_selfplay_factory(
    tmp_path, structured_model_cfg, structured_obs_spec, action_spec
) -> None:
    """Older configs still get the established random-anchored output pool."""
    cfg = selfplay_cfg(tmp_path, structured_model_cfg)
    assert "opponent_pool_mode" not in cfg.train

    factory = build_opponent_factory(cfg, structured_obs_spec, action_spec, tmp_path)
    assert factory is not None
    pool = factory()

    assert isinstance(pool, SnapshotOpponentPool)
    assert isinstance(pool.opponents[0], RandomOpponent)
    assert pool._checkpoint_dir.resolve() == tmp_path.resolve()  # noqa: SLF001


def test_warmup_checkpoint_missing_path_fails_during_factory_build(
    tmp_path, structured_model_cfg, structured_obs_spec, action_spec
) -> None:
    """A warmup_checkpoint that does not resolve to a file fails in the parent."""
    cfg = selfplay_cfg(tmp_path, structured_model_cfg)
    cfg.train.warmup_checkpoint = str(tmp_path / "missing.pt")

    with pytest.raises(ValueError, match="warmup_checkpoint .* does not exist"):
        build_opponent_factory(cfg, structured_obs_spec, action_spec, tmp_path)


def test_warmup_checkpoint_replaces_random_anchor(
    tmp_path, structured_model_cfg, structured_obs_spec, action_spec, monkeypatch
) -> None:
    """warmup_checkpoint becomes the league's anchor instead of RandomOpponent."""
    anchor = tmp_path / "anchor.pt"
    anchor.write_bytes(b"placeholder")

    def fake_load(checkpoint_path, cfg, obs_spec, action_spec, encoder):  # noqa: ARG001
        return lambda _obs: [0]

    monkeypatch.setattr("src.training.self_play._load_snapshot", fake_load)
    cfg = selfplay_cfg(tmp_path, structured_model_cfg)
    cfg.train.warmup_checkpoint = str(anchor)

    factory = build_opponent_factory(cfg, structured_obs_spec, action_spec, tmp_path)
    pool = factory()

    assert isinstance(pool, SnapshotOpponentPool)
    assert pool.snapshot_count == 0
    assert not isinstance(pool.opponents[0], RandomOpponent)


def test_unknown_opponent_pool_mode_fails_during_factory_build(
    tmp_path, structured_model_cfg, structured_obs_spec, action_spec
) -> None:
    """A mode typo cannot silently fall back to a different experiment."""
    cfg = selfplay_cfg(tmp_path, structured_model_cfg)
    cfg.train.opponent_pool_mode = "frezen"

    with pytest.raises(ValueError, match="unknown opponent_pool_mode"):
        build_opponent_factory(cfg, structured_obs_spec, action_spec, tmp_path)


@pytest.mark.parametrize("mode", ["frozen", "refresh"])
def test_external_modes_reject_pfsp_and_factories_are_picklable(
    tmp_path, structured_model_cfg, structured_obs_spec, action_spec, mode: str
) -> None:
    """New populations require literal uniform sampling and survive workers."""
    baseline = tmp_path / "baseline"
    learner = tmp_path / "learner"
    write_snapshot_files(baseline, list(range(10)))
    cfg = selfplay_cfg(learner, structured_model_cfg)
    cfg.train.opponent_pool_mode = mode
    cfg.train.opponent_checkpoint_dir = str(baseline)
    cfg.train.pool_size = 10
    cfg.train.opponent_sampling = "pfsp"

    with pytest.raises(ValueError, match="requires opponent_sampling=uniform"):
        build_opponent_factory(cfg, structured_obs_spec, action_spec, learner)

    cfg.train.opponent_sampling = "uniform"
    factory = build_opponent_factory(cfg, structured_obs_spec, action_spec, learner)
    assert factory is not None
    restored = pickle.loads(pickle.dumps(factory))
    assert restored.keywords["refresh"] is (mode == "refresh")


def test_external_modes_validate_checkpoint_sources(
    tmp_path, structured_model_cfg, structured_obs_spec, action_spec
) -> None:
    """Missing, undersized, and aliased sources fail in the parent process."""
    cfg = selfplay_cfg(tmp_path / "learner", structured_model_cfg)
    cfg.train.opponent_pool_mode = "frozen"
    cfg.train.opponent_sampling = "uniform"
    cfg.train.pool_size = 10
    cfg.train.opponent_checkpoint_dir = str(tmp_path / "missing")
    with pytest.raises(ValueError, match="does not exist"):
        build_opponent_factory(
            cfg, structured_obs_spec, action_spec, tmp_path / "learner"
        )

    baseline = tmp_path / "baseline"
    write_snapshot_files(baseline, list(range(9)))
    cfg.train.opponent_checkpoint_dir = str(baseline)
    with pytest.raises(ValueError, match="expected at least pool_size=10"):
        build_opponent_factory(
            cfg, structured_obs_spec, action_spec, tmp_path / "learner"
        )

    write_snapshot_files(baseline, [9])
    cfg.train.opponent_pool_mode = "refresh"
    with pytest.raises(ValueError, match="must be distinct"):
        build_opponent_factory(cfg, structured_obs_spec, action_spec, baseline)


def test_refresh_requires_periodic_learner_snapshots(
    tmp_path, structured_model_cfg, structured_obs_spec, action_spec
) -> None:
    """Refresh cannot be selected while learner snapshot publication is off."""
    baseline = tmp_path / "baseline"
    write_snapshot_files(baseline, list(range(10)))
    cfg = selfplay_cfg(tmp_path / "learner", structured_model_cfg, snapshot_interval=0)
    cfg.train.opponent_pool_mode = "refresh"
    cfg.train.opponent_checkpoint_dir = str(baseline)
    cfg.train.opponent_sampling = "uniform"
    cfg.train.pool_size = 10

    with pytest.raises(ValueError, match="snapshot_interval > 0"):
        build_opponent_factory(
            cfg, structured_obs_spec, action_spec, tmp_path / "learner"
        )


def test_opponent_factory_survives_pickling(
    tmp_path, structured_model_cfg, structured_obs_spec, action_spec
) -> None:
    """
    The factory is pickled into every ParallelEnv worker, so it must round-trip
    and still build a working league on the other side.
    """
    cfg = selfplay_cfg(tmp_path, structured_model_cfg)
    factory = build_opponent_factory(cfg, structured_obs_spec, action_spec, tmp_path)
    assert factory is not None

    pool = pickle.loads(pickle.dumps(factory))()
    assert isinstance(pool, SnapshotOpponentPool)
    pool.on_reset()
    assert pool.snapshot_count == 0


def test_real_snapshot_loads_back_into_the_league(
    tmp_path, structured_model_cfg, structured_obs_spec, action_spec
) -> None:
    """
    The full write/discover cycle works on a real checkpoint: what the callback
    saves is what a worker's league can rebuild and play.
    """
    cfg = selfplay_cfg(tmp_path, structured_model_cfg)
    actor_critic = build_actor_critic(
        structured_model_cfg, structured_obs_spec, action_spec
    )
    SnapshotCallback(actor_critic, tmp_path, interval=100).on_rollout_end(100, {})

    factory = build_opponent_factory(cfg, structured_obs_spec, action_spec, tmp_path)
    assert factory is not None
    pool = factory()
    pool.on_reset()
    assert pool.snapshot_count == 1


def test_league_members_share_one_encoder(
    tmp_path, structured_model_cfg, structured_obs_spec, action_spec
) -> None:
    """
    Every snapshot a worker loads reuses that worker's single encoder, rather
    than allocating a fresh set of scratch buffers (and a fresh one-shot
    truncation warning) per league member.
    """
    cfg = selfplay_cfg(tmp_path, structured_model_cfg)
    actor_critic = build_actor_critic(
        structured_model_cfg, structured_obs_spec, action_spec
    )
    callback = SnapshotCallback(actor_critic, tmp_path, interval=100)
    for frames in (100, 200, 300):
        callback.on_rollout_end(frames, {})

    factory = build_opponent_factory(cfg, structured_obs_spec, action_spec, tmp_path)
    assert factory is not None
    pool = factory()
    pool.on_reset()
    assert pool.snapshot_count == 3

    # Sharing is an internal arrangement with no public surface, so the
    # assertion has to reach for the private members to observe it.
    members = pool._opponents  # noqa: SLF001
    encoders = {
        id(member._encoder)  # noqa: SLF001
        for member in members
        if hasattr(member, "_encoder")
    }
    assert len(encoders) == 1


def test_eval_opponent_factory_builds_the_configured_reference(
    tmp_path, structured_model_cfg
) -> None:
    """
    The evaluator's opponent comes from ``train.eval_opponent`` explicitly,
    rather than from whatever the environment happens to default to.
    """
    cfg = selfplay_cfg(tmp_path, structured_model_cfg)
    cfg.seed = 7
    assert isinstance(build_eval_opponent_factory(cfg)(), RandomOpponent)


def test_unsupported_eval_opponent_is_rejected(tmp_path, structured_model_cfg) -> None:
    """
    An unimplemented reference fails loudly at build time. Falling back to the
    random opponent would silently score the run against something other than
    what the config asked for.
    """
    cfg = selfplay_cfg(tmp_path, structured_model_cfg)
    cfg.seed = 7
    cfg.train.eval_opponent = "snapshot"
    with pytest.raises(ValueError, match="eval_opponent"):
        build_eval_opponent_factory(cfg)


def test_checkpoint_eval_opponent_loads_a_frozen_snapshot(
    tmp_path, structured_model_cfg, structured_obs_spec, action_spec
) -> None:
    """
    ``eval_opponent=checkpoint`` scores the run against a frozen saved model.
    """
    cfg = selfplay_cfg(tmp_path, structured_model_cfg)
    actor_critic = build_actor_critic(
        structured_model_cfg, structured_obs_spec, action_spec
    )
    snapshot = save_actor_critic(actor_critic, tmp_path / "reference.pt")
    cfg.train.eval_opponent = "checkpoint"
    cfg.train.eval_opponent_checkpoint = str(snapshot)

    opponent = build_eval_opponent_factory(cfg, structured_obs_spec, action_spec)()
    assert isinstance(opponent, GreedyPolicyOpponent)


def test_checkpoint_eval_opponent_requires_a_path(
    tmp_path, structured_model_cfg, structured_obs_spec, action_spec
) -> None:
    """
    Selecting the checkpoint reference without a path fails loudly at build time.
    """
    cfg = selfplay_cfg(tmp_path, structured_model_cfg)
    cfg.train.eval_opponent = "checkpoint"
    with pytest.raises(ValueError, match="eval_opponent_checkpoint"):
        build_eval_opponent_factory(cfg, structured_obs_spec, action_spec)


def test_checkpoint_eval_opponent_requires_specs(
    tmp_path, structured_model_cfg
) -> None:
    """
    Rebuilding a checkpoint's network needs the env specs, so omitting them raises.
    """
    cfg = selfplay_cfg(tmp_path, structured_model_cfg)
    cfg.train.eval_opponent = "checkpoint"
    cfg.train.eval_opponent_checkpoint = str(tmp_path / "reference.pt")
    with pytest.raises(ValueError, match="specs"):
        build_eval_opponent_factory(cfg)


def test_checkpoint_eval_opponent_rejects_missing_file(
    tmp_path, structured_model_cfg, structured_obs_spec, action_spec
) -> None:
    """
    A checkpoint path that does not exist is caught at build time, not mid-eval.
    """
    cfg = selfplay_cfg(tmp_path, structured_model_cfg)
    cfg.train.eval_opponent = "checkpoint"
    cfg.train.eval_opponent_checkpoint = str(tmp_path / "does_not_exist.pt")
    with pytest.raises(ValueError, match="does not exist"):
        build_eval_opponent_factory(cfg, structured_obs_spec, action_spec)


def test_checkpoint_pool_eval_is_frozen_to_baseline_directory(
    tmp_path, structured_model_cfg, structured_obs_spec, action_spec
) -> None:
    """Aggregate evaluation never adds the refreshed learner population."""
    baseline = tmp_path / "baseline"
    learner = tmp_path / "learner"
    write_snapshot_files(baseline, list(range(10)))
    write_snapshot_files(learner, [999])
    cfg = selfplay_cfg(learner, structured_model_cfg)
    cfg.train.opponent_pool_mode = "refresh"
    cfg.train.opponent_checkpoint_dir = str(baseline)
    cfg.train.eval_opponent_checkpoint_dir = str(baseline)
    cfg.train.eval_opponent_pool_size = 10

    factory = build_eval_opponent_factory(
        cfg,
        structured_obs_spec,
        action_spec,
        opponent="checkpoint_pool",
    )

    assert factory.keywords["checkpoint_dirs"] == (baseline.resolve(),)
    assert factory.keywords["refresh"] is False
    assert factory.keywords["pool_size"] == 10


def test_best_response_opponent_loads_the_frozen_agent(
    tmp_path, structured_model_cfg, structured_obs_spec, action_spec
) -> None:
    """
    The best-response opponent is the frozen agent whose exploitability is probed.
    """
    cfg = selfplay_cfg(tmp_path, structured_model_cfg)
    actor_critic = build_actor_critic(
        structured_model_cfg, structured_obs_spec, action_spec
    )
    probed = save_actor_critic(actor_critic, tmp_path / "probed.pt")
    cfg.train.best_response_checkpoint = str(probed)

    factory = build_best_response_opponent_factory(
        cfg, structured_obs_spec, action_spec
    )
    assert isinstance(factory(), GreedyPolicyOpponent)


def test_best_response_opponent_survives_pickling(
    tmp_path, structured_model_cfg, structured_obs_spec, action_spec
) -> None:
    """
    The collection opponent is pickled into each worker, so it must round-trip.
    """
    cfg = selfplay_cfg(tmp_path, structured_model_cfg)
    actor_critic = build_actor_critic(
        structured_model_cfg, structured_obs_spec, action_spec
    )
    cfg.train.best_response_checkpoint = str(
        save_actor_critic(actor_critic, tmp_path / "probed.pt")
    )

    factory = build_best_response_opponent_factory(
        cfg, structured_obs_spec, action_spec
    )
    opponent = pickle.loads(pickle.dumps(factory))()
    assert isinstance(opponent, GreedyPolicyOpponent)


def test_best_response_requires_an_existing_checkpoint(
    tmp_path, structured_model_cfg, structured_obs_spec, action_spec
) -> None:
    """
    A missing or unset probed checkpoint fails loudly before training starts.
    """
    cfg = selfplay_cfg(tmp_path, structured_model_cfg)
    with pytest.raises(ValueError, match="best_response_checkpoint"):
        build_best_response_opponent_factory(cfg, structured_obs_spec, action_spec)
    cfg.train.best_response_checkpoint = str(tmp_path / "missing.pt")
    with pytest.raises(ValueError, match="does not exist"):
        build_best_response_opponent_factory(cfg, structured_obs_spec, action_spec)


@pytest.mark.parametrize("deterministic", [True, False])
def test_evaluator_scores_policy_against_fixed_opponent(
    structured_model_cfg, deterministic
) -> None:
    """
    Evaluation plays complete episodes against the fixed random opponent and
    reports outcome rates that partition the episodes, in both action-selection
    modes.
    """
    env_cfg = structured_env_cfg(num_workers=1)
    env_factory = make_env_factories(env_cfg)[0]
    probe_env = env_factory()
    try:
        obs_spec, action_spec = probe_env.observation_spec, probe_env.action_spec
    finally:
        probe_env.close()

    cfg = OmegaConf.merge(env_cfg, structured_model_cfg)
    policy = build_ppo_actor_critic(cfg, obs_spec, action_spec).get_policy_operator()

    evaluator = Evaluator(env_factory, n_episodes=2, deterministic=deterministic)
    try:
        metrics = evaluator.evaluate(policy)
    finally:
        evaluator.close()

    assert metrics["episodes"] == 2
    assert metrics["win_rate"] + metrics["draw_rate"] + metrics[
        "loss_rate"
    ] == pytest.approx(1.0)
    assert metrics["mean_episode_length"] > 0
