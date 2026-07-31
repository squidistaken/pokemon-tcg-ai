import shutil
from pathlib import Path

import pytest
import torch
from omegaconf import DictConfig, OmegaConf
from tensordict import TensorDict

from src.env.archetype_index import ArchetypeIndex
from src.env.curriculum_handles import CurriculumHandles
from src.env.level_buffer import LevelBuffer
from src.training.curriculum import Curriculum, build_curriculum
from src.training.env_factory import load_deck_pool
from tests.conftest import DECK_PATH, structured_env_cfg

WORKERS = 2
STEPS = 6


def make_buffer(capacity: int = 16, min_visits: int = 2, **kwargs) -> LevelBuffer:
    """
    Build a level buffer with test-scale settings.

    :param capacity: Maximum entries held.
    :param min_visits: Episodes before a score is trusted.
    :param kwargs: Overrides forwarded to :class:`LevelBuffer`.
    :return: The buffer under test.
    """
    return LevelBuffer(capacity=capacity, min_visits=min_visits, **kwargs)


def make_curriculum(archetype_count: int = 2, **kwargs) -> Curriculum:
    """
    Build a curriculum over a fabricated archetype grouping.

    :param archetype_count: Number of archetypes to fabricate.
    :param kwargs: Overrides forwarded to :class:`Curriculum`.
    :return: The curriculum under test.
    """
    index = ArchetypeIndex.from_paths(
        [f"decks/arch{n}/list{i}.csv" for n in range(archetype_count) for i in range(2)]
    )
    return Curriculum(
        archetypes=index,
        handles=CurriculumHandles.allocate(64),
        buffer=make_buffer(),
        num_workers=WORKERS,
        **kwargs,
    )


def batch(
    levels: list[list[int]],
    residuals: list[list[float]],
    done: list[list[bool]],
    rewards: list[list[float]] | None = None,
    anchors: list[list[bool]] | None = None,
    terminated: list[list[bool]] | None = None,
) -> TensorDict:
    """
    Assemble a post-advantage batch shaped like the collector's output.

    :param levels: Matchup identifier per worker per step.
    :param residuals: ``value_target - state_value`` per worker per step.
    :param done: Episode-end flag per worker per step.
    :param rewards: Reward per step; zeros if omitted.
    :param anchors: Anchor-opponent flag per step; all True if omitted.
    :param terminated: Natural-termination flag; equal to ``done`` if omitted.
    :return: TensorDict with the keys :meth:`Curriculum.observe` reads.
    """
    rows, steps = len(levels), len(levels[0])
    done_tensor = torch.tensor(done).reshape(rows, steps, 1)
    return TensorDict(
        {
            "level_id": torch.tensor(levels).reshape(rows, steps, 1),
            "value_target": torch.tensor(residuals).reshape(rows, steps, 1),
            "state_value": torch.zeros(rows, steps, 1),
            "opponent_is_anchor": torch.tensor(
                anchors if anchors is not None else [[True] * steps] * rows
            ).reshape(rows, steps, 1),
            "next": TensorDict(
                {
                    "done": done_tensor,
                    "terminated": (
                        torch.tensor(terminated).reshape(rows, steps, 1)
                        if terminated is not None
                        else done_tensor.clone()
                    ),
                    "reward": torch.tensor(
                        rewards if rewards is not None else [[0.0] * steps] * rows
                    ).reshape(rows, steps, 1),
                },
                batch_size=[rows, steps],
            ),
        },
        batch_size=[rows, steps],
    )


def test_score_uses_signed_mean_not_mean_magnitude() -> None:
    """
    A zero-mean matchup must score zero however large its swings.

    This is the central adaptation: with terminal +/-1 rewards a coin-flip
    matchup produces huge per-step residuals forever, so scoring the magnitude
    per step would pin the curriculum on the noisiest levels. Averaging the
    signed residual first cancels that.
    """
    buffer = make_buffer(min_visits=1)
    buffer.prefill([0, 1])

    for value in (1.0, -1.0, 1.0, -1.0):
        buffer.commit(0, value)
    for _ in range(4):
        buffer.commit(1, 0.4)

    noisy, biased = buffer.entries[0], buffer.entries[1]
    assert buffer.score(noisy) == pytest.approx(0.0, abs=1e-6)
    assert buffer.score(biased) == pytest.approx(0.4, abs=1e-6)


def test_unvisited_levels_outrank_measured_ones() -> None:
    """
    Optimistic initialization is what makes exploration happen without a branch.
    """
    buffer = make_buffer(min_visits=2)
    buffer.prefill([0, 1])
    for _ in range(5):
        buffer.commit(0, 0.9)

    distribution = buffer.distribution()

    assert buffer.score(buffer.entries[1]) == float("inf")
    assert distribution[1] > distribution[0]


def test_immature_levels_are_never_evicted() -> None:
    """
    Evicting on a one-episode score would churn levels out before measurement.
    """
    buffer = make_buffer(capacity=2, min_visits=3)
    buffer.prefill([0, 1])
    buffer.commit(0, 0.1)

    assert buffer.insert(2) is None
    assert {entry.pair_id for entry in buffer.entries} == {0, 1}

    for _ in range(3):
        buffer.commit(0, 0.1)
    assert buffer.insert(2) is not None


def test_staleness_lifts_neglected_levels() -> None:
    """
    Without the staleness term a low-scoring level would never be re-measured.
    """
    buffer = make_buffer(min_visits=1, staleness_coefficient=0.5)
    buffer.prefill([0, 1])
    buffer.commit(0, 0.0)
    buffer.commit(1, 0.0)
    for _ in range(50):
        buffer.commit(0, 0.0)

    assert buffer.distribution()[1] > buffer.distribution()[0]


def test_win_rate_matrix_reads_back_anchor_games() -> None:
    """
    The matrix deck selection reads must reflect only recorded anchor games.
    """
    buffer = make_buffer(min_visits=1)
    buffer.prefill([0, 1, 2, 3])
    for _ in range(3):
        buffer.commit(1, 0.0, outcome=1.0)
    buffer.commit(2, 0.0, outcome=0.0)

    matrix = buffer.win_rate_matrix(archetype_count=2)

    assert matrix[0, 1] == pytest.approx(1.0)
    assert matrix[1, 0] == pytest.approx(0.0)
    assert torch.isnan(torch.tensor(matrix[0, 0]))


def test_state_dict_round_trips() -> None:
    """
    Dumps must restore exactly, since deck selection differences two of them.
    """
    buffer = make_buffer(min_visits=1)
    buffer.prefill([0, 1])
    buffer.commit(0, 0.3, outcome=1.0)
    state = buffer.state_dict()

    restored = make_buffer(min_visits=1)
    restored.load_state_dict(state)

    assert restored.state_dict() == state
    assert restored.entries[0].wins == pytest.approx(1.0)


def test_observe_commits_only_finished_episodes() -> None:
    """
    A batch ending mid-episode must leave the visit uncounted until it ends.
    """
    curriculum = make_curriculum()
    buffer = curriculum.buffer

    curriculum.observe(
        batch(
            levels=[[0] * STEPS, [1] * STEPS],
            residuals=[[1.0] * STEPS, [1.0] * STEPS],
            done=[[False] * STEPS, [False] * STEPS],
        )
    )
    assert all(entry.visits == 0 for entry in buffer.entries)

    curriculum.observe(
        batch(
            levels=[[0] * STEPS, [1] * STEPS],
            residuals=[[1.0] * STEPS, [1.0] * STEPS],
            done=[[False] * (STEPS - 1) + [True]] * 2,
        )
    )
    assert buffer.entries[0].visits == 1
    assert buffer.entries[1].visits == 1


def test_episode_spanning_batches_averages_over_the_whole_episode() -> None:
    """
    The committed residual must cover every step of the episode, not the tail.

    Episodes here outlive a rollout, so an accumulator that reset at batch
    boundaries would silently score only whatever fragment the last batch held.
    """
    curriculum = make_curriculum()

    curriculum.observe(
        batch(
            levels=[[0] * STEPS] * WORKERS,
            residuals=[[0.0] * STEPS] * WORKERS,
            done=[[False] * STEPS] * WORKERS,
        )
    )
    curriculum.observe(
        batch(
            levels=[[0] * STEPS] * WORKERS,
            residuals=[[2.0] * STEPS] * WORKERS,
            done=[[False] * (STEPS - 1) + [True]] * WORKERS,
        )
    )

    # Twelve steps: six at 0.0 then six at 2.0, so the episode mean is 1.0.
    # Scoring only the closing batch would give 2.0.
    assert curriculum.buffer.entries[0].mean_residual == pytest.approx(1.0)


def test_truncated_episodes_do_not_enter_the_matchup_matrix() -> None:
    """
    A run cut off by the selection cap has no result to record.
    """
    curriculum = make_curriculum()

    curriculum.observe(
        batch(
            levels=[[0] * STEPS] * WORKERS,
            residuals=[[0.5] * STEPS] * WORKERS,
            done=[[False] * (STEPS - 1) + [True]] * WORKERS,
            terminated=[[False] * STEPS] * WORKERS,
            rewards=[[0.0] * STEPS] * WORKERS,
        )
    )

    entry = curriculum.buffer.entries[0]
    assert entry.visits == WORKERS, "the residual should still be scored"
    assert entry.games == 0, "but no phantom draw should reach the matrix"


def test_non_anchor_games_are_excluded_from_the_matrix() -> None:
    """
    Win rates must be comparable across training, so only the anchor counts.
    """
    curriculum = make_curriculum()

    curriculum.observe(
        batch(
            levels=[[0] * STEPS] * WORKERS,
            residuals=[[0.5] * STEPS] * WORKERS,
            done=[[False] * (STEPS - 1) + [True]] * WORKERS,
            rewards=[[0.0] * (STEPS - 1) + [1.0]] * WORKERS,
            anchors=[[False] * STEPS] * WORKERS,
        )
    )

    entry = curriculum.buffer.entries[0]
    assert entry.visits == WORKERS
    assert entry.games == 0


def test_anchor_only_scoring_skips_league_episodes() -> None:
    """
    With the flag on, only anchor episodes may move a level's score.
    """
    curriculum = make_curriculum(anchor_only_scoring=True)

    curriculum.observe(
        batch(
            levels=[[0] * STEPS, [1] * STEPS],
            residuals=[[0.5] * STEPS] * WORKERS,
            done=[[False] * (STEPS - 1) + [True]] * WORKERS,
            anchors=[[True] * STEPS, [False] * STEPS],
        )
    )

    assert curriculum.buffer.entries[0].visits == 1
    assert curriculum.buffer.entries[1].visits == 0


def test_observe_requires_advantage_keys() -> None:
    """
    Calling before the advantage module is a wiring bug, not a silent no-op.
    """
    curriculum = make_curriculum()
    data = batch(
        levels=[[0] * STEPS] * WORKERS,
        residuals=[[0.0] * STEPS] * WORKERS,
        done=[[False] * STEPS] * WORKERS,
    )
    del data["value_target"]

    with pytest.raises(KeyError, match="value_target"):
        curriculum.observe(data)


def test_publish_pushes_a_normalized_distribution() -> None:
    """
    Whatever the workers read must be a usable probability vector.
    """
    curriculum = make_curriculum()
    curriculum.publish()

    size = int(curriculum.handles.size[0])
    probabilities = curriculum.handles.probabilities[:size]

    assert size == curriculum.archetypes.pair_count
    assert float(probabilities.sum()) == pytest.approx(1.0, abs=1e-5)
    assert bool((probabilities > 0).all())


def build_corpus(root: Path, archetypes: int = 2, per_archetype: int = 2) -> Path:
    """
    Fabricate a per-archetype deck corpus laid out like the fetched one.

    :param root: Directory to build the corpus under.
    :param archetypes: Number of archetype folders.
    :param per_archetype: Deck files per archetype.
    :return: The corpus root.
    """
    corpus = root / "decks"
    for archetype in range(archetypes):
        folder = corpus / f"arch{archetype}"
        folder.mkdir(parents=True, exist_ok=True)
        for index in range(per_archetype):
            shutil.copy(DECK_PATH, folder / f"list{index}.csv")
    return corpus


def curriculum_cfg(tmp_path: Path, **overrides) -> DictConfig:
    """
    Build a config enabling the curriculum over a fabricated corpus.

    :param tmp_path: Directory the corpus is built under.
    :param overrides: Extra ``env.curriculum`` entries.
    :return: Merged config.
    """
    corpus = build_corpus(tmp_path)
    cfg = structured_env_cfg(num_workers=WORKERS)
    cfg.env.mp_start_method = "fork"
    cfg.env.deck_pool = str(corpus)
    cfg.env.curriculum = OmegaConf.create(
        {"enabled": True, "capacity": 64, "min_visits": 2, **overrides}
    )
    return cfg


def test_build_curriculum_covers_every_matchup(tmp_path: Path) -> None:
    """
    The buffer must start holding the whole level space, not a subset.
    """
    curriculum = build_curriculum(curriculum_cfg(tmp_path))

    assert curriculum is not None
    assert curriculum.archetypes.count == 2
    assert curriculum.buffer.size == 4


def test_curriculum_never_trains_on_held_out_decks(tmp_path: Path) -> None:
    """
    Held-out decks measure generalization, so the curriculum must not see them.

    The curriculum addresses decks by position into the pool it was built from,
    so the archetype index and the sampler must agree on the *same* split; a
    mismatch would both leak the evaluation set and mis-deal every level.
    """
    cfg = curriculum_cfg(tmp_path)
    cfg.env.deck_holdout_frac = 0.5
    cfg.env.deck_split_seed = 3

    train_decks, train_paths = load_deck_pool(cfg, deck_split="train")
    _eval_decks, eval_paths = load_deck_pool(cfg, deck_split="eval")
    curriculum = build_curriculum(cfg)

    assert curriculum is not None
    assert not set(train_paths) & set(eval_paths), "splits must be disjoint"
    # The index the curriculum built must describe exactly the decks the
    # sampler will be handed, or positions refer to different lists.
    assert curriculum.archetypes.count == ArchetypeIndex.from_paths(train_paths).count
    covered = {
        position
        for archetype in range(curriculum.archetypes.count)
        for position in curriculum.archetypes.decks_for(archetype)
    }
    assert covered == set(range(len(train_decks)))


def test_build_curriculum_disabled_by_default(tmp_path: Path) -> None:
    """
    Nothing changes unless the curriculum is explicitly turned on.
    """
    cfg = curriculum_cfg(tmp_path)
    cfg.env.curriculum.enabled = False

    assert build_curriculum(cfg) is None


def test_build_curriculum_rejects_undersized_capacity(tmp_path: Path) -> None:
    """
    Silently covering part of the level space would bias the curriculum.
    """
    cfg = curriculum_cfg(tmp_path, capacity=2)

    with pytest.raises(ValueError, match="capacity"):
        build_curriculum(cfg)


def test_build_curriculum_rejects_spawn(tmp_path: Path) -> None:
    """
    Under spawn the workers would never see a curriculum update.
    """
    cfg = curriculum_cfg(tmp_path)
    cfg.env.mp_start_method = "spawn"

    with pytest.raises(ValueError, match="fork"):
        build_curriculum(cfg)


def test_build_curriculum_requires_a_deck_pool(tmp_path: Path) -> None:
    """
    A fixed deck pair has exactly one matchup and nothing to curate.
    """
    cfg = curriculum_cfg(tmp_path)
    cfg.env.deck_pool = None

    with pytest.raises(ValueError, match="deck_pool"):
        build_curriculum(cfg)


def test_save_state_is_reloadable(tmp_path: Path) -> None:
    """
    Dumps are how deck selection recovers a window of training.
    """
    curriculum = make_curriculum()
    curriculum.buffer.commit(0, 0.5, outcome=1.0)
    destination = tmp_path / "state" / "curriculum_000000000512.pt"

    curriculum.save_state(destination)
    state = torch.load(destination, weights_only=False)

    assert destination.exists()
    assert state["buffer"]["wins"][0] == pytest.approx(1.0)
    assert state["archetypes"] == list(curriculum.archetypes.names)
