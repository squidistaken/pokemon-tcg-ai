import csv
from pathlib import Path

from omegaconf import OmegaConf

from src.policies.greedy_policy_opponent import save_actor_critic
from src.policies.ppo_actor import build_actor_critic
from src.training.callbacks import CrossPlayCallback
from tests.conftest import structured_env_cfg


def _cfg(structured_model_cfg):
    """Full env+model config the callback rebuilds snapshot networks from."""
    return OmegaConf.merge(structured_env_cfg(num_workers=1), structured_model_cfg)


def _write_snapshots(actor_critic, directory: Path, frame_counts) -> None:
    """Save the same actor-critic under snapshot-style filenames."""
    directory.mkdir(parents=True, exist_ok=True)
    for frames in frame_counts:
        save_actor_critic(actor_critic, directory / f"snapshot_{frames:012d}.pt")


def test_cross_play_callback_writes_matrix_and_elo(
    tmp_path, structured_model_cfg, structured_obs_spec, action_spec
) -> None:
    """
    The run-end round-robin writes a square matrix CSV and an Elo ranking CSV
    covering every checkpoint plus the final model.
    """
    cfg = _cfg(structured_model_cfg)
    actor_critic = build_actor_critic(structured_model_cfg, structured_obs_spec, action_spec)
    checkpoints = tmp_path / "ckpts"
    _write_snapshots(actor_critic, checkpoints, (100, 200))
    output = tmp_path / "out"

    callback = CrossPlayCallback(
        actor_critic, cfg, structured_obs_spec, action_spec,
        checkpoint_dir=checkpoints, output_dir=output, n_games=2, seed=0,
    )
    callback.on_train_end({"frames": 300})

    with (output / "crossplay_matrix.csv").open() as handle:
        rows = list(csv.reader(handle))
    header = rows[0]
    assert "current" in header  # the final model is entered alongside snapshots
    assert len(rows) - 1 == len(header) - 1  # square matrix (rows == columns)

    with (output / "crossplay_elo.csv").open() as handle:
        elo_rows = list(csv.reader(handle))
    assert elo_rows[0] == ["checkpoint", "elo"]
    assert len(elo_rows) - 1 == len(header) - 1  # one rating per checkpoint


def test_cross_play_callback_without_snapshots_is_a_noop(
    tmp_path, structured_model_cfg, structured_obs_spec, action_spec
) -> None:
    """
    With no snapshots on disk, both hooks return quietly and write nothing.
    """
    cfg = _cfg(structured_model_cfg)
    actor_critic = build_actor_critic(structured_model_cfg, structured_obs_spec, action_spec)
    output = tmp_path / "out"

    callback = CrossPlayCallback(
        actor_critic, cfg, structured_obs_spec, action_spec,
        checkpoint_dir=tmp_path / "empty", output_dir=output, n_games=2, seed=0,
    )
    callback.on_eval_end(50, {})
    callback.on_train_end({"frames": 100})
    assert not (output / "crossplay_matrix.csv").exists()


def test_cross_play_callback_eval_scores_against_latest(
    tmp_path, structured_model_cfg, structured_obs_spec, action_spec
) -> None:
    """
    Once a snapshot exists, the eval hook plays the current model against it
    without error (and without a W&B run, silently).
    """
    cfg = _cfg(structured_model_cfg)
    actor_critic = build_actor_critic(structured_model_cfg, structured_obs_spec, action_spec)
    checkpoints = tmp_path / "ckpts"
    _write_snapshots(actor_critic, checkpoints, (100,))

    callback = CrossPlayCallback(
        actor_critic, cfg, structured_obs_spec, action_spec,
        checkpoint_dir=checkpoints, output_dir=tmp_path / "out", n_games=2, seed=0,
    )
    callback.on_eval_end(100, {})  # completes without raising


def test_cross_play_callback_reuses_eval_sampler_across_calls(
    tmp_path, structured_model_cfg, structured_obs_spec, action_spec
) -> None:
    """
    on_eval_end scores against a persistent sampler so its round-robin/uniform
    cursor actually advances across evaluations, instead of a fresh,
    identically-seeded sampler replaying the same decks every time.
    """
    cfg = _cfg(structured_model_cfg)
    actor_critic = build_actor_critic(structured_model_cfg, structured_obs_spec, action_spec)
    checkpoints = tmp_path / "ckpts"
    _write_snapshots(actor_critic, checkpoints, (100,))

    callback = CrossPlayCallback(
        actor_critic, cfg, structured_obs_spec, action_spec,
        checkpoint_dir=checkpoints, output_dir=tmp_path / "out", n_games=1, seed=0,
    )
    sampler_before = callback._eval_sampler  # noqa: SLF001
    callback.on_eval_end(100, {})
    assert callback._eval_sampler is sampler_before  # noqa: SLF001


def test_select_checkpoints_caps_to_one(
    tmp_path, structured_model_cfg, structured_obs_spec, action_spec
) -> None:
    """
    max_checkpoints=1 returns exactly the most recent checkpoint, not two.
    """
    cfg = _cfg(structured_model_cfg)
    actor_critic = build_actor_critic(structured_model_cfg, structured_obs_spec, action_spec)
    callback = CrossPlayCallback(
        actor_critic, cfg, structured_obs_spec, action_spec,
        checkpoint_dir=tmp_path / "ckpts", output_dir=tmp_path / "out",
        n_games=1, max_checkpoints=1, seed=0,
    )
    paths = [Path(f"snapshot_{i:012d}.pt") for i in range(5)]

    selected = callback._select_checkpoints(paths)  # noqa: SLF001

    assert selected == [paths[-1]]
