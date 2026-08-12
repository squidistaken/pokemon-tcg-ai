import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from tensordict import TensorDict

from src.curriculum.archetype_index import ArchetypeIndex
from src.curriculum.deck_sampler import NO_LEVEL
from src.curriculum.handles import CurriculumHandles
from src.curriculum.level_buffer import LevelBuffer

logger = logging.getLogger(__name__)


@dataclass
class _OpenEpisode:
    """
    Partial episode carried across batch boundaries for one collector row.

    Episodes here run far longer than a rollout, so most batches end mid-game.
    Committing what a batch happens to contain would make the visit count
    measure batches rather than episodes, and the maturity threshold would
    then be counting the wrong thing.

    :param level_id: Matchup this episode is being played on.
    :param residual_sum: Sum of critic residuals seen so far.
    :param steps: Number of steps contributing to ``residual_sum``.
    """

    level_id: int = NO_LEVEL
    residual_sum: float = 0.0
    steps: int = 0

    def reset(self) -> None:
        """
        Clear the accumulator for the next episode.
        """
        self.level_id = NO_LEVEL
        self.residual_sum = 0.0
        self.steps = 0


class Curriculum:
    """
    Learner-side owner of the level curriculum.

    Holds the :class:`~src.curriculum.level_buffer.LevelBuffer`, consumes each
    collected batch after advantage estimation, and republishes the resulting
    sampling distribution to the environment workers.

    Scoring lives here rather than in the workers because the signal is a
    critic residual, which the environments cannot compute. The workers only
    need the conclusion, which reaches them through
    :class:`~src.curriculum.handles.CurriculumHandles`. That asymmetry is
    specific to the level curriculum: opponent selection is driven by terminal
    rewards the environment already sees, so it stays worker-local and needs no
    channel at all.
    """

    def __init__(
        self,
        archetypes: ArchetypeIndex,
        handles: CurriculumHandles,
        buffer: LevelBuffer,
        num_workers: int,
        anchor_only_scoring: bool = False,
        explore_prob: float = 0.0,
    ) -> None:
        """
        :param archetypes: Grouping of the deck pool into archetypes.
        :param handles: Shared channel the distribution is published to.
        :param buffer: Level buffer to score into.
        :param num_workers: Collector rows, one accumulator each.
        :param anchor_only_scoring: Score only episodes played against the
            anchor opponent. Off by default: it removes the drift that a
            strengthening league induces in level scores, at the cost of
            discarding most episodes' scoring signal. Enable only if the
            logged diagnostic shows that drift is material.
        :param explore_prob: Forwarded to the workers' deck samplers, which
            read it off :attr:`explore_prob`; kept here only so it travels with
            the rest of the curriculum's construction. See
            :class:`~src.curriculum.deck_sampler.CurriculumDeckSampler`.
        """
        self._archetypes = archetypes
        self._handles = handles
        self._buffer = buffer
        self._anchor_only_scoring = anchor_only_scoring
        self._explore_prob = explore_prob
        self._open = [_OpenEpisode() for _ in range(num_workers)]
        # Probability the live distribution assigns to each pair_id, kept so
        # observe() can check the episodes that came back against the weights
        # the workers were supposed to have drawn them under.
        self._published: dict[int, float] = {}
        self._published_collision = 0.0
        self._drawn_fidelity_sum = 0.0
        self._drawn_episodes = 0
        # Only when the whole space fits: above capacity, prefilling every
        # matchup would immediately overflow it, and every level starting
        # unmeasured together is exactly the coverage sweep discovery relies on
        # not needing (see LevelBuffer.commit()). Larger corpora instead
        # discover levels lazily through commit()'s probation path, driven by
        # the workers' explore_prob.
        if archetypes.pair_count <= buffer.capacity:
            self._buffer.prefill(range(archetypes.pair_count))
        self.publish()

    @property
    def explore_prob(self) -> float:
        """
        :return: Probability a worker draws a fresh matchup instead of
            replaying from the published distribution.
        """
        return self._explore_prob

    @property
    def buffer(self) -> LevelBuffer:
        """
        :return: The level buffer being scored into.
        """
        return self._buffer

    @property
    def archetypes(self) -> ArchetypeIndex:
        """
        :return: The archetype grouping levels are drawn from.
        """
        return self._archetypes

    @property
    def handles(self) -> CurriculumHandles:
        """
        :return: The shared channel handed to the environment factories.
        """
        return self._handles

    def observe(self, data: TensorDict) -> None:
        """
        Score every episode that finished inside a collected batch.

        Expects the batch *after* advantage estimation, so ``value_target`` and
        ``state_value`` are present. Reads the raw pair rather than
        ``advantage`` deliberately: with ``average_gae`` enabled the advantages
        are standardized per batch, which would make the score a batch-relative
        quantity rather than a measure of critic bias.

        :param data: Collected batch shaped ``(workers, time)``.
        :raises KeyError: If advantage estimation has not been run on ``data``.
        """
        for key in ("value_target", "state_value"):
            if key not in data:
                raise KeyError(
                    f"{key!r} missing; run the advantage module before observe()"
                )
        rows, steps = data.batch_size[0], data.batch_size[1]
        levels = data["level_id"].reshape(rows, steps)
        residuals = (data["value_target"] - data["state_value"]).reshape(rows, steps)
        done = data["next", "done"].reshape(rows, steps)
        # Segment on `done` but decide outcomes on `terminated`: a run cut off
        # by the engine's selection cap ends the episode without producing a
        # result, and recording its zero reward would enter a phantom draw into
        # the matchup matrix.
        terminated = data["next", "terminated"].reshape(rows, steps)
        rewards = data["next", "reward"].reshape(rows, steps)
        anchors = data["opponent_is_anchor"].reshape(rows, steps)

        for row in range(rows):
            self._observe_row(
                self._open[row],
                levels[row],
                residuals[row],
                done[row],
                terminated[row],
                rewards[row],
                anchors[row],
            )

    def abandon_open_episodes(self) -> None:
        """
        Discard every partially observed episode without scoring it.

        Called when the collector's worker pool is rebuilt after one of its
        processes died: the replacement pool starts every row on a fresh
        battle, so the residuals banked against the old rows belong to games
        that will never report ``done``. Clearing them keeps that evidence from
        being committed under whichever level lands in the row next.
        """
        for accumulator in self._open:
            accumulator.reset()

    def publish(self) -> None:
        """
        Recompute the sampling distribution and hand it to the workers.
        """
        distribution = self._buffer.distribution()
        if distribution.size == 0:
            return
        pair_ids = self._buffer.pair_ids()
        self._handles.publish(
            torch.from_numpy(pair_ids),
            torch.from_numpy(np.ascontiguousarray(distribution, dtype=np.float32)),
        )
        # Snapshot what the workers will now be drawing under, so the next
        # batch's episodes can be checked against it.
        self._published = dict(
            zip(pair_ids.tolist(), distribution.tolist(), strict=True)
        )
        self._published_collision = float(np.square(distribution).sum())

    def _record_draw(self, level_id: int) -> None:
        """
        Note the probability the live distribution gave a level that came back.

        :param level_id: Matchup identifier of a finished episode.
        """
        probability = self._published.get(level_id)
        if probability is None or self._published_collision <= 0.0:
            return
        # Divide here rather than in metrics(): publish() replaces the
        # collision probability straight after observe(), so deferring the
        # division would score this batch's draws against the *next* batch's
        # distribution. Harmless while the distribution drifts slowly, wildly
        # wrong when it does not.
        self._drawn_fidelity_sum += probability / self._published_collision
        self._drawn_episodes += 1

    def _sampling_fidelity(self) -> float:
        """
        How closely the episodes actually played track the published weights.

        Averages the published probability of every level that came back and
        divides by the probability a genuine draw from that distribution would
        have averaged (its collision probability, ``sum(p^2)``). Normalizing
        this way makes the metric read 1.0 whether the distribution is sharp or
        flat, so one threshold holds for the whole run:

        * **order 1** -- workers are drawing from the published distribution.
          Readings somewhat below 1 are normal rather than a fault: a level is
          chosen at episode reset, which can precede the episode's end by a
          batch or two, so some episodes were drawn under a slightly older
          distribution. Measured around 0.6--0.8 in healthy runs.
        * ``~0.0`` -- draws are unrelated to it. The distribution is computed
          and published correctly but never reaches the samplers, which is
          invisible in every other counter here: visits, maturity and scores
          all keep advancing while the curriculum steers nothing.

        Alarm on a *sustained collapse toward zero*, not on any departure
        from 1.

        A ratio rather than a correlation because a batch holds only tens of
        episodes spread over a level space thousands wide; a histogram over
        that is almost entirely zeros, whereas a mean over the episodes that
        did come back is stable.

        :return: The ratio, or NaN before any episode has been attributed.
        """
        if self._drawn_episodes == 0:
            return float("nan")
        return self._drawn_fidelity_sum / self._drawn_episodes

    def metrics(self) -> dict[str, float]:
        """
        Curriculum statistics for the training callbacks.

        :return: Buffer counters, the share of episodes played against the
            anchor (the diagnostic for whether league drift is contaminating
            level scores), and the sampling-fidelity check described in
            :meth:`_sampling_fidelity`.
        """
        stats = self._buffer.stats()
        stats["sampling_fidelity"] = self._sampling_fidelity()
        self._drawn_fidelity_sum = 0.0
        self._drawn_episodes = 0
        return stats

    def save_state(self, path: str | Path) -> None:
        """
        Write the buffer's entries to disk.

        Deck selection needs win/loss tallies over a window of training, which
        a single end-of-run total cannot provide, so these are dumped
        periodically and differenced afterwards.

        :param path: Destination file.
        """
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "archetypes": list(self._archetypes.names),
            "buffer": self._buffer.state_dict(),
        }
        staging = destination.with_suffix(destination.suffix + ".tmp")
        torch.save(state, staging)
        staging.replace(destination)
        logger.debug("Wrote curriculum state to %s", destination)

    def load_state(self, path: str | Path) -> None:
        """
        Restore buffer entries written by :meth:`save_state`.

        A continued run that skips this rediscovers every matchup from nothing,
        throwing away the visit counts and win/loss tallies that decide which
        matchups the sampler prioritizes -- the curriculum's whole state.

        The saved archetype names are checked against the current index because
        a level is addressed by ``pair_id = agent * count + opponent``. That
        encoding is only meaningful against the archetype list it was computed
        from: load a state built over a different corpus and every entry silently
        refers to the wrong matchup, which no later error would reveal.

        :param path: File written by :meth:`save_state`.
        :raises ValueError: If the file's archetypes differ from this run's, or
            it holds more entries than this run's buffer can.
        """
        state = torch.load(Path(path), map_location="cpu", weights_only=False)
        saved_archetypes = list(state["archetypes"])
        current = list(self._archetypes.names)
        if saved_archetypes != current:
            raise ValueError(
                f"curriculum state at {path} was built over {len(saved_archetypes)} "
                f"archetypes and this run has {len(current)}; matchup ids are "
                f"positions in that list, so the entries would be misattributed. "
                f"Point env.curriculum.init_state at a state from a run over this "
                f"same deck corpus, or leave it unset to start fresh."
            )
        saved_entries = len(state["buffer"]["pair_id"])
        if saved_entries > self._buffer.capacity:
            raise ValueError(
                f"curriculum state at {path} holds {saved_entries} entries but "
                f"env.curriculum.capacity is {self._buffer.capacity}. Loading it "
                f"would push the buffer past the capacity its eviction policy "
                f"assumes. Raise the capacity to at least {saved_entries}."
            )
        self._buffer.load_state_dict(state["buffer"])
        self.publish()
        logger.info(
            "Restored curriculum: %d scored matchup(s) over %d archetypes from %s",
            self._buffer.size,
            len(current),
            path,
        )

    def _observe_row(
        self,
        accumulator: _OpenEpisode,
        levels: torch.Tensor,
        residuals: torch.Tensor,
        done: torch.Tensor,
        terminated: torch.Tensor,
        rewards: torch.Tensor,
        anchors: torch.Tensor,
    ) -> None:
        """
        Fold one collector row into the buffer, committing at episode ends.

        :param accumulator: This row's carried partial episode.
        :param levels: Matchup identifier per step.
        :param residuals: Critic residual per step.
        :param done: Episode-end flag per step.
        :param terminated: Natural-termination flag per step.
        :param rewards: Reward per step.
        :param anchors: Whether the anchor opponent played, per step.
        """
        start = 0
        for end in torch.nonzero(done).flatten().tolist():
            self._extend(accumulator, levels, residuals, start, end + 1)
            outcome = (
                self._outcome(float(rewards[end].item()))
                if bool(terminated[end].item()) and bool(anchors[end].item())
                else None
            )
            scored = not self._anchor_only_scoring or bool(anchors[end].item())
            if accumulator.steps > 0 and accumulator.level_id != NO_LEVEL:
                self._record_draw(accumulator.level_id)
            if accumulator.steps > 0 and accumulator.level_id != NO_LEVEL and scored:
                self._buffer.commit(
                    accumulator.level_id,
                    accumulator.residual_sum / accumulator.steps,
                    outcome,
                )
            accumulator.reset()
            start = end + 1
        self._extend(accumulator, levels, residuals, start, levels.shape[0])

    @staticmethod
    def _extend(
        accumulator: _OpenEpisode,
        levels: torch.Tensor,
        residuals: torch.Tensor,
        start: int,
        end: int,
    ) -> None:
        """
        Add a run of steps to a row's open episode.

        :param accumulator: Accumulator to extend.
        :param levels: Matchup identifier per step.
        :param residuals: Critic residual per step.
        :param start: First step index, inclusive.
        :param end: Last step index, exclusive.
        """
        if end <= start:
            return
        if accumulator.steps == 0:
            accumulator.level_id = int(levels[start].item())
        accumulator.residual_sum += float(residuals[start:end].sum().item())
        accumulator.steps += end - start

    @staticmethod
    def _outcome(reward: float) -> float:
        """
        Map a terminal reward to a win fraction.

        :param reward: Terminal reward from the agent's perspective.
        :return: 1.0 for a win, 0.0 for a loss, 0.5 for a draw.
        """
        if reward > 0.0:
            return 1.0
        if reward < 0.0:
            return 0.0
        return 0.5


def build_curriculum(cfg: DictConfig) -> Curriculum | None:
    """
    Construct the level curriculum described by ``cfg.env.curriculum``.

    Returns None when the curriculum is disabled, which leaves the environments
    on whichever deck sampler the config already selected.

    :param cfg: Hydra configuration with ``env`` and top-level ``seed``.
    :return: A curriculum, or None when disabled.
    :raises ValueError: If enabled without a deck pool, under a worker start
        method that cannot share memory, or if the corpus exceeds ``capacity``
        without ``explore_prob`` set to discover the rest lazily.
    """
    settings = cfg.env.get("curriculum")
    if not settings or not settings.get("enabled", False):
        return None
    if not cfg.env.get("deck_pool"):
        raise ValueError(
            "env.curriculum.enabled requires env.deck_pool: the curriculum scores "
            "archetype matchups, and a fixed deck0/deck1 pair has only one."
        )
    CurriculumHandles.require_shared_start_method(str(cfg.env.mp_start_method))

    # Deferred: env_factory imports from src.training, so a module-level import
    # here would close a cycle.
    from src.training.env_factory import load_deck_pool

    # The train split only: the held-out decks exist to measure generalization,
    # so scoring matchups over them would train on the evaluation set.
    _decks, paths = load_deck_pool(cfg, deck_split="train")
    archetypes = ArchetypeIndex.from_paths(paths)
    capacity = int(settings.get("capacity", 2000))
    explore_prob = float(settings.get("explore_prob", 0.0))
    if not 0.0 <= explore_prob <= 1.0:
        raise ValueError(
            f"env.curriculum.explore_prob must be in [0, 1], got {explore_prob}"
        )
    oversized = archetypes.pair_count > capacity
    if oversized and explore_prob <= 0.0:
        raise ValueError(
            f"the corpus yields {archetypes.count} archetypes and therefore "
            f"{archetypes.pair_count} matchups, above env.curriculum.capacity="
            f"{capacity}. Either raise the capacity to cover every matchup, or set "
            f"env.curriculum.explore_prob > 0 so the workers keep sampling fresh "
            f"matchups and unseen ones are discovered and scored lazily (see "
            f"LevelBuffer.commit()) instead of the whole space needing to fit up front."
        )
    buffer = LevelBuffer(
        capacity=capacity,
        score_temperature=float(settings.get("score_temperature", 0.9)),
        staleness_coefficient=float(settings.get("staleness_coefficient", 0.4)),
        min_visits=int(settings.get("min_visits", 5)),
        seed=int(cfg.seed),
    )
    logger.info(
        "Level curriculum over %d archetypes (%d matchups), capacity %d%s",
        archetypes.count,
        archetypes.pair_count,
        capacity,
        "" if not oversized else f", discovering lazily (explore_prob={explore_prob})",
    )
    curriculum = Curriculum(
        archetypes=archetypes,
        handles=CurriculumHandles.allocate(capacity),
        buffer=buffer,
        num_workers=int(cfg.env.num_workers),
        anchor_only_scoring=bool(settings.get("anchor_only_scoring", False)),
        explore_prob=explore_prob,
    )
    init_state = settings.get("init_state")
    if init_state:
        state_path = Path(to_absolute_path(str(init_state)))
        if not state_path.is_file():
            raise ValueError(f"env.curriculum.init_state {state_path} does not exist.")
        curriculum.load_state(state_path)
    return curriculum
