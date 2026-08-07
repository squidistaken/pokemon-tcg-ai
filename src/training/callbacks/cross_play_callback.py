import csv
import logging
import random
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from omegaconf import DictConfig
from torchrl.data import Categorical, Composite

import wandb
from src.env.battle_handle import BattleHandle
from src.env.deck_sampler import DeckSampler, build_deck_sampler
from src.env.snapshot_opponent_pool import SNAPSHOT_SUFFIX
from src.models.actor_critic import ActorCritic
from src.policies.greedy_policy_opponent import GreedyPolicyOpponent
from src.policies.ppo_actor import build_actor_critic
from src.training.cross_play import (
    Policy,
    bradley_terry_elo,
    crossplay_matrix,
    play_series,
)
from src.training.env_factory import _build_sampler_spec, make_encoder
from src.training.self_play import _load_snapshot

from .base import TrainingCallback

logger = logging.getLogger(__name__)


class CrossPlayCallback(TrainingCallback):
    """
    Cross-play evaluation of the learner against its own frozen history.

    Two views, both scoring policies through the greedy serving path on the
    held-out deck pool:

    * cheap: during training (:meth:`on_eval_end`) the current learner plays a short
      series against the latest snapshot.
    * expensive: at the end (:meth:`on_train_end`) every retained checkpoint (plus the
      final model) plays a full round-robin, and the win-rate matrix is fit to
      an order-free Bradley-Terry Elo ranking.
    """

    def __init__(
            self,
            actor_critic: ActorCritic,
            cfg: DictConfig,
            obs_spec: Composite,
            action_spec: Categorical,
            checkpoint_dir: str | Path,
            output_dir: str | Path,
            n_games: int = 20,
            max_checkpoints: int = 8,
            seed: int = 0,
            sampler_spec: dict[str, Any] | None = None,
    ) -> None:
        """
        :param actor_critic: The live learner, shared with the trainer; snapshot
            per use so scoring never mutates it.
        :param cfg: Hydra config, used to rebuild snapshot networks and the eval
            deck pool.
        :param obs_spec: Environment observation spec.
        :param action_spec: Environment action spec.
        :param checkpoint_dir: Directory the snapshots are written to.
        :param output_dir: Directory the matrix/Elo CSVs are written to.
        :param n_games: Games per pairing (per series).
        :param max_checkpoints: Cap on checkpoints entered into the run-end
            matrix; more than this are subsampled evenly across the run so the
            ranking still spans the whole trajectory without an N**2 blow-up.
        :param seed: Base seed for match seat assignment and deck draws.
        :param sampler_spec: Precomputed eval-split sampler spec, for a caller
            that already parsed the held-out deck pool and can share it instead of
            this callback re-parsing it. Built fresh when ``None``.
        """
        self._actor_critic = actor_critic
        self._cfg = cfg
        self._obs_spec = obs_spec
        self._action_spec = action_spec
        self._checkpoint_dir = Path(checkpoint_dir)
        self._output_dir = Path(output_dir)
        self._n_games = n_games
        self._max_checkpoints = max_checkpoints
        self._seed = seed
        self._encoder = make_encoder(
            cfg.env.get("encoder", "structured"), int(cfg.env.max_options)
        )
        self._sampler_spec = (
            sampler_spec if sampler_spec is not None else _build_sampler_spec(cfg, deck_split="eval")
        )
        # Persistent across on_eval_end calls so its round-robin/uniform cursor
        # actually advances through the held-out pool over the run.
        self._eval_sampler = build_deck_sampler(self._sampler_spec, seed=seed)

    def on_train_start(self, run_config: Mapping[str, Any]) -> None:
        """Unused; cross-play needs snapshots that do not exist yet at start."""

    def on_rollout_start(self, step: int) -> None:
        """Unused; cross-play runs at evaluation and run end."""

    def on_rollout_end(self, step: int, metrics: Mapping[str, float]) -> None:
        """Unused; cross-play runs at evaluation and run end."""

    def on_eval_end(self, step: int, metrics: Mapping[str, float]) -> None:  # noqa: ARG002
        """
        Score the current learner against the latest snapshot.

        Early evaluations are not charged for a match that cannot happen yet.

        :param step: Frames collected at this evaluation.
        :param metrics: Evaluation metrics; unused.
        """
        snapshots = self._snapshot_paths()
        if not snapshots:
            return
        latest = snapshots[-1]
        current = self._current_opponent()
        opponent = self._load(latest)
        result = play_series(
            BattleHandle(),
            current,
            opponent,
            self._eval_sampler,
            self._n_games,
            random.Random(self._seed + step),
        )
        logger.info(
            "Cross-play vs latest snapshot %s over %d game(s): score=%.3f",
            latest.name, result.scored, result.score,
        )
        self._wandb_log(step, {"crossplay/vs_latest_snapshot": result.score})

    def on_train_end(self, summary: Mapping[str, float]) -> None:
        """
        Round-robin every checkpoint (plus the final model) and rank by Elo.

        :param summary: Aggregate run statistics, read for the frame count.
        """
        checkpoints = self._select_checkpoints(self._snapshot_paths())
        if not checkpoints:
            logger.info("Cross-play skipped: no snapshots on disk to rank.")
            return
        policies: dict[str, Policy] = {path.stem: self._load(path) for path in checkpoints}
        policies["current"] = self._current_opponent()
        names, scores, games = crossplay_matrix(
            policies, self._sampler_factory, n_games=self._n_games, seed=self._seed
        )
        elo = bradley_terry_elo(names, scores, games)
        self._write_csvs(names, scores, elo)
        leader = max(names, key=lambda name: elo[name])
        logger.info(
            "Cross-play Elo leader over %d checkpoint(s): %s (%.0f)",
            len(names), leader, elo[leader],
        )
        step = int(summary.get("frames", 0))
        self._wandb_log(step, {f"crossplay/elo/{name}": elo[name] for name in names})
        self._wandb_summary(
            {"crossplay/best_checkpoint": leader, "crossplay/best_elo": elo[leader]}
        )

    def _snapshot_paths(self) -> list[Path]:
        """
        :return: Snapshot paths in the writer's lexicographic (chronological)
            order, or empty if the directory does not exist yet.
        """
        if not self._checkpoint_dir.is_dir():
            return []
        return sorted(self._checkpoint_dir.glob(f"*{SNAPSHOT_SUFFIX}"))

    def _select_checkpoints(self, paths: list[Path]) -> list[Path]:
        """
        Subsample the checkpoints down to ``max_checkpoints``, evenly spaced.

        The most recent is always kept, so the final model's nearest snapshot is
        represented even when the run wrote far more than the cap.

        :param paths: All snapshot paths, chronological.
        :return: An evenly spaced subset spanning the run.
        """
        if len(paths) <= self._max_checkpoints:
            return paths
        n = self._max_checkpoints
        if n == 1:
            return [paths[-1]]
        step = (len(paths) - 1) / (n - 1)
        indices = sorted({round(i * step) for i in range(n)} | {len(paths) - 1})
        return [paths[i] for i in indices]

    def _load(self, checkpoint_path: Path) -> Policy:
        """
        Load a snapshot as a greedy opponent on CPU.

        :param checkpoint_path: Snapshot path.
        :return: The greedy opponent.
        """
        return _load_snapshot(
            checkpoint_path, self._cfg, self._obs_spec, self._action_spec, self._encoder
        )

    def _current_opponent(self) -> Policy:
        """
        Copy the live learner's weights into a fresh greedy opponent.

        :return: A greedy opponent playing the current parameters.
        """
        snapshot = build_actor_critic(self._cfg, self._obs_spec, self._action_spec)
        snapshot.load_state_dict(self._actor_critic.state_dict())
        return GreedyPolicyOpponent(snapshot, self._encoder, device="cpu")

    def _sampler_factory(self) -> DeckSampler:
        """
        :return: A fresh deck sampler over the held-out eval pool.
        """
        return build_deck_sampler(self._sampler_spec, seed=self._seed)

    def _write_csvs(
            self,
            names: list[str],
            scores: Mapping[str, Mapping[str, float]],
            elo: Mapping[str, float],
    ) -> None:
        """
        Write the win-rate matrix and the Elo ranking as CSVs beside the run.

        :param names: Checkpoint names, matrix row/column order.
        :param scores: ``scores[a][b]`` = A's win-equivalent rate vs B.
        :param elo: Elo rating per checkpoint.
        """
        self._output_dir.mkdir(parents=True, exist_ok=True)
        matrix_path = self._output_dir / "crossplay_matrix.csv"
        with matrix_path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["", *names])
            for a in names:
                writer.writerow([a, *(f"{scores[a][b]:.4f}" for b in names)])
        elo_path = self._output_dir / "crossplay_elo.csv"
        with elo_path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["checkpoint", "elo"])
            for name in sorted(names, key=lambda n: elo[n], reverse=True):
                writer.writerow([name, f"{elo[name]:.1f}"])
        logger.info("Wrote cross-play matrix and Elo ranking to %s", self._output_dir)

    @staticmethod
    def _wandb_log(step: int, metrics: Mapping[str, float]) -> None:
        """
        Log scalars to the W&B run (started by the W&B callback), if one is active.

        :param step: Frame count x value.
        :param metrics: Scalars to log.
        """
        if wandb.run is not None:
            wandb.run.log(dict(metrics), step=step)

    @staticmethod
    def _wandb_summary(values: Mapping[str, Any]) -> None:
        """
        Record run-level values on the W&B run's summary, if one is active.

        :param values: Summary entries to record.
        """
        if wandb.run is not None:
            for key, value in values.items():
                wandb.run.summary[key] = value
