import copy
import json
import logging
import os
from collections.abc import Callable, Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.checkpoint_registry import (
    CHECKPOINT_KEY_LENGTH,
    append_checkpoint_record,
    sha256_file,
)
from src.models.actor_critic import ActorCritic
from src.policies.greedy_policy_opponent import save_actor_critic

from .base import TrainingCallback

logger = logging.getLogger(__name__)

#: Zero-padding width of the frame counter in snapshot filenames, chosen so
#: lexicographic order matches numeric order for any realistic run length.
_FRAME_DIGITS = 12
CheckpointLogger = Callable[[Path, str, int], None]


class SnapshotCallback(TrainingCallback):
    """
    Freezes the learner into the self-play league at a fixed frame interval.

    This is the writing half of the self-play loop: every ``interval``
    collected frames it snapshots the actor-critic into ``checkpoint_dir``,
    where the environment workers'
    :class:`~src.env.snapshot_opponent_pool.SnapshotOpponentPool` instances
    discover it and start playing against it.

    Writes go to a temporary file that is then atomically renamed into place,
    so a worker scanning the directory concurrently never observes a partially
    written snapshot.

    Filenames embed the zero-padded frame count, so sorting them lexicographic-
    ally orders them by recency.
    """

    def __init__(
            self,
            actor_critic: ActorCritic,
            checkpoint_dir: str | Path,
            interval: int,
            checkpoint_loggers: Iterable[CheckpointLogger] = (),
            registry_path: str | Path | None = None,
            repo_root: str | Path | None = None,
    ) -> None:
        """
        :param actor_critic: Learner to snapshot; shared with the trainer, so
            each write captures the parameters as of that moment.
        :param checkpoint_dir: Directory snapshots are written to.
        :param interval: Frames between periodic self-play snapshots. ``0``
            disables periodic writes but still writes the final checkpoint.
        :param checkpoint_loggers: Optional sinks called after an atomic write,
            used to mirror checkpoints to services such as W&B.
        :param registry_path: Append-only CSV receiving the final checkpoint.
            None disables registry writes (primarily for direct unit-test use).
        :param repo_root: Root used to make repository checkpoint paths portable
            in the registry. Required when ``registry_path`` is set.
        """
        self._actor_critic = actor_critic
        self._checkpoint_dir = Path(checkpoint_dir)
        self._interval = interval
        self._checkpoint_loggers = list(checkpoint_loggers)
        self._registry_path = Path(registry_path) if registry_path is not None else None
        self._repo_root = Path(repo_root) if repo_root is not None else None
        if self._registry_path is not None and self._repo_root is None:
            raise ValueError("repo_root is required when registry_path is configured")
        self._last_snapshot_frames = 0
        self._last_checkpoint: Path | None = None
        self._last_digest: str | None = None
        self._last_published_frames: int | None = None
        self._checkpoint_config: dict[str, Any] | None = None

    def on_train_start(self, run_config: Mapping[str, Any]) -> None:
        """
        Create the checkpoint directory and retain the serving-time config.

        :param run_config: Resolved Hydra config; its inference subset is stored
            in versioned checkpoint files.
        """
        self._checkpoint_config = _inference_config(run_config)
        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)
        if self._interval > 0:
            logger.info("Self-play snapshots every %d frames -> %s", self._interval, self._checkpoint_dir)
        else:
            logger.info("Final checkpoint -> %s", self._checkpoint_dir)

    def on_rollout_start(self, step: int) -> None:
        """
        Unused; snapshots are taken after a rollout's update has been applied.

        :param step: Total frames collected before this rollout.
        """

    def on_rollout_end(self, step: int, metrics: Mapping[str, float]) -> None:  # noqa: ARG002
        """
        Snapshot the learner if ``interval`` frames have passed since the last.

        :param step: Total frames collected so far.
        :param metrics: Running training metrics; unused.
        """
        if self._interval <= 0 or step - self._last_snapshot_frames < self._interval:
            return
        self._last_snapshot_frames = step
        self._write(step)

    def on_eval_end(self, step: int, metrics: Mapping[str, float]) -> None:
        """
        Unused; evaluation does not change the learner.

        :param step: Frames collected at the time of the evaluation.
        :param metrics: Evaluation metrics; unused.
        """

    def on_train_end(self, summary: Mapping[str, float]) -> None:
        """
        Write a final snapshot so the finished policy is always on disk.

        Skipped when the run happened to end exactly on a snapshot boundary,
        which would otherwise rewrite the same parameters under the same name.

        :param summary: Aggregate run statistics, read for the frame count.
        """
        frames = int(summary.get("frames", self._last_snapshot_frames))
        if frames > self._last_snapshot_frames:
            self._write(frames)
        if self._last_checkpoint is not None and self._last_digest is not None:
            if self._last_published_frames != frames:
                if self._registry_path is not None:
                    assert self._repo_root is not None
                    try:
                        append_checkpoint_record(
                            self._registry_path,
                            self._last_checkpoint,
                            digest=self._last_digest,
                            frames=frames,
                            repo_root=self._repo_root,
                        )
                    except Exception:
                        logger.exception(
                            "Could not append final checkpoint %s to registry %s; "
                            "continuing with remaining callbacks",
                            self._last_checkpoint,
                            self._registry_path,
                        )
                for checkpoint_logger in self._checkpoint_loggers:
                    checkpoint_logger(self._last_checkpoint, self._last_digest, frames)
                self._last_published_frames = frames
            print(f"checkpoint: {self._last_checkpoint}")
            print(f"checkpoint-key: {self._last_digest[:CHECKPOINT_KEY_LENGTH]}")

    def _write(self, frames: int) -> None:
        """
        Atomically write one snapshot named after the frame count.

        :param frames: Frame count to embed in the filename.
        """
        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)
        final_path = self._checkpoint_dir / f"snapshot_{frames:0{_FRAME_DIGITS}d}.pt"
        # Same directory (os.replace is only atomic within a filesystem) and a
        # suffix the pool's "*.pt" scan does not match.
        staging_path = final_path.parent / f"{final_path.name}.tmp"
        save_actor_critic(
            self._actor_critic,
            staging_path,
            config=self._checkpoint_config,
            frames=frames,
        )
        os.replace(staging_path, final_path)
        digest = sha256_file(final_path)
        _write_metadata(final_path, digest, frames)
        self._last_snapshot_frames = frames
        self._last_checkpoint = final_path
        self._last_digest = digest
        logger.info(
            "Wrote checkpoint %s (key %s)",
            final_path.name,
            digest[:CHECKPOINT_KEY_LENGTH],
        )


def _inference_config(run_config: Mapping[str, Any]) -> dict[str, Any] | None:
    """
    Extract only config needed to reconstruct inference and choose its deck.

    :param run_config: Fully resolved training configuration.
    :return: Portable checkpoint config, or None for legacy/unit-test callers.
    """
    model = run_config.get("model")
    env = run_config.get("env")
    if not isinstance(model, Mapping) or not isinstance(env, Mapping):
        return None
    portable_env = {
        key: copy.deepcopy(env[key])
        for key in ("encoder", "max_options", "deck0")
        if key in env
    }
    return {"model": copy.deepcopy(dict(model)), "env": portable_env}


def _write_metadata(path: Path, digest: str, frames: int) -> None:
    """Atomically write the human-readable sidecar for a checkpoint."""
    metadata_path = path.with_suffix(".json")
    staging_path = metadata_path.parent / f"{metadata_path.name}.tmp"
    metadata = {
        "format_version": 1,
        "checkpoint": path.name,
        "sha256": digest,
        "key": digest[:CHECKPOINT_KEY_LENGTH],
        "frames": frames,
        "created_at": datetime.now(UTC).isoformat(),
    }
    staging_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    os.replace(staging_path, metadata_path)
