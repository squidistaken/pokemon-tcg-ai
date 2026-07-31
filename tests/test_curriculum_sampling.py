from functools import partial

import pytest
import torch
from torchrl.envs import ParallelEnv, TransformedEnv
from torchrl.envs.transforms import ActionMask
from torchrl.envs.utils import check_env_specs

from src.env.archetype_index import ArchetypeIndex
from src.env.curriculum_deck_sampler import NO_LEVEL, CurriculumDeckSampler
from src.env.curriculum_handles import CurriculumHandles
from src.env.tcg_env import TCGEnv
from src.policies.random_masked_policy import RandomMaskedPolicy
import torch.multiprocessing as torch_mp

from src.training.trainer import Trainer
from tests.test_tcg_env import DECK

CAPACITY = 64


def corpus_paths(archetypes: dict[str, int]) -> list[str]:
    """
    Build deck paths laid out the way the fetched corpus is.

    :param archetypes: Number of deck files to fabricate per archetype name.
    :return: Paths in ``decks/<archetype>/<name>.csv`` form.
    """
    return [
        f"decks/{name}/list{index}.csv"
        for name, count in archetypes.items()
        for index in range(count)
    ]


def make_index(archetypes: dict[str, int]) -> ArchetypeIndex:
    """
    Build an archetype index over a fabricated corpus layout.

    :param archetypes: Number of deck files per archetype name.
    :return: The index over those paths.
    """
    return ArchetypeIndex.from_paths(corpus_paths(archetypes))


def test_from_paths_groups_by_folder() -> None:
    """
    Archetypes come from the folder name, ordered so numbering is reproducible.
    """
    index = make_index({"gardevoir": 2, "charizard": 3})

    assert index.count == 2
    assert index.names == ("charizard", "gardevoir")
    assert index.decks_for(0) == (2, 3, 4)
    assert index.decks_for(1) == (0, 1)


def test_loose_decks_fall_back_to_ungrouped() -> None:
    """
    A deck with no archetype folder must still be placed, not dropped.
    """
    index = ArchetypeIndex.from_paths(["solo.csv", "decks/charizard/a.csv"])

    assert ArchetypeIndex.UNGROUPED in index.names


def test_pair_id_round_trips() -> None:
    """
    Every matchup identifier decodes back to the archetypes it came from.
    """
    index = make_index({"a": 1, "b": 1, "c": 1})

    seen = set()
    for agent in range(index.count):
        for opponent in range(index.count):
            pair_id = index.pair_id(agent, opponent)
            seen.add(pair_id)
            assert index.unpair(pair_id) == (agent, opponent)
    assert len(seen) == index.pair_count


def test_index_rejects_empty_archetypes() -> None:
    """
    An archetype with no decks would be sampleable but undealable.
    """
    with pytest.raises(ValueError, match="holds no decks"):
        ArchetypeIndex(["empty"], [[]])


def test_handles_publish_round_trips() -> None:
    """
    Published slots must read back, with the rest of the channel ignored.
    """
    handles = CurriculumHandles.allocate(CAPACITY)
    assert int(handles.size[0]) == 0

    handles.publish(torch.tensor([3, 7]), torch.tensor([0.25, 0.75]))

    assert int(handles.size[0]) == 2
    assert handles.pair_ids[:2].tolist() == [3, 7]
    assert handles.probabilities[:2].tolist() == pytest.approx([0.25, 0.75])


def test_handles_reject_oversized_publish() -> None:
    """
    Publishing past capacity must fail rather than truncate the distribution.
    """
    handles = CurriculumHandles.allocate(2)
    with pytest.raises(ValueError, match="capacity"):
        handles.publish(torch.arange(3), torch.ones(3) / 3)


def test_sampler_falls_back_before_the_first_publish() -> None:
    """
    Collection starts before the first update, so an empty channel is normal.
    """
    index = make_index({"a": 2, "b": 2})
    sampler = CurriculumDeckSampler([DECK] * 4, index, CurriculumHandles.allocate(CAPACITY), seed=0)

    assert sampler.level_id == NO_LEVEL
    deck0, deck1 = sampler.sample()

    assert len(deck0) == len(deck1) == 60
    assert 0 <= sampler.level_id < index.pair_count


def test_sampler_draws_only_published_levels() -> None:
    """
    Once a distribution is published, every draw must come from it.
    """
    index = make_index({"a": 2, "b": 2, "c": 2})
    handles = CurriculumHandles.allocate(CAPACITY)
    wanted = index.pair_id(0, 2)
    handles.publish(torch.tensor([wanted]), torch.tensor([1.0]))
    sampler = CurriculumDeckSampler([DECK] * 6, index, handles, seed=0)

    for _ in range(20):
        sampler.sample()
        assert sampler.level_id == wanted


def test_sampler_deals_from_the_drawn_archetypes() -> None:
    """
    The dealt lists must belong to the archetypes the level names.
    """
    index = make_index({"a": 2, "b": 2})
    decks = [[position] * 60 for position in range(4)]
    handles = CurriculumHandles.allocate(CAPACITY)
    handles.publish(torch.tensor([index.pair_id(1, 0)]), torch.tensor([1.0]))
    sampler = CurriculumDeckSampler(decks, index, handles, seed=0)

    for _ in range(20):
        deck0, deck1 = sampler.sample()
        assert deck0[0] in index.decks_for(1)
        assert deck1[0] in index.decks_for(0)


def test_sampler_follows_a_republished_distribution() -> None:
    """
    A worker must pick up the learner's next distribution, not cache the first.
    """
    index = make_index({"a": 1, "b": 1, "c": 1})
    handles = CurriculumHandles.allocate(CAPACITY)
    sampler = CurriculumDeckSampler([DECK] * 3, index, handles, seed=0)

    first = index.pair_id(0, 1)
    handles.publish(torch.tensor([first]), torch.tensor([1.0]))
    sampler.sample()
    assert sampler.level_id == first

    second = index.pair_id(2, 2)
    handles.publish(torch.tensor([second]), torch.tensor([1.0]))
    sampler.sample()
    assert sampler.level_id == second


def make_curriculum_env(
    index: ArchetypeIndex, handles: CurriculumHandles, seed: int
) -> TransformedEnv:
    """
    Build a masked env whose decks come from the curriculum channel.

    Module-level so it survives being sent to a ParallelEnv worker.

    :param index: Archetype grouping.
    :param handles: Shared curriculum channel.
    :param seed: Seed for this instance.
    :return: TransformedEnv with the ActionMask transform applied.
    """
    sampler = CurriculumDeckSampler([DECK] * 4, index, handles, seed=seed)
    return TransformedEnv(TCGEnv(max_options=128, seed=seed, deck_sampler=sampler), ActionMask())


def test_level_updates_reach_running_parallel_workers() -> None:
    """
    A publish made *after* the workers start must reach them.

    This is the transport's actual requirement, and the reason it must be
    shared memory rather than a copy: the learner republishes after every batch,
    long after the workers were forked. Publishing before the fork would prove
    nothing, since a private per-worker copy taken at fork time would serve the
    same values forever. So the first distribution is published first, and a
    second one after the workers are already running.
    """
    # Reproduce the process state training actually runs in: importing torchrl
    # sets the global start method to "spawn", and ParallelEnv starts workers
    # lazily, so this is what they would be created under.
    previous = torch_mp.get_start_method(allow_none=True)
    torch_mp.set_start_method("spawn", force=True)

    index = make_index({"a": 2, "b": 2})
    handles = CurriculumHandles.allocate(CAPACITY)
    first = index.pair_id(1, 0)
    handles.publish(torch.tensor([first]), torch.tensor([1.0]))

    env = ParallelEnv(
        num_workers=2,
        create_env_fn=[
            partial(make_curriculum_env, index, handles, 0),
            partial(make_curriculum_env, index, handles, 1),
        ],
        mp_start_method="fork",
    )
    try:
        env.reset()
        before = env.rollout(
            max_steps=4, policy=RandomMaskedPolicy(), break_when_any_done=False
        )

        second = index.pair_id(0, 1)
        handles.publish(torch.tensor([second]), torch.tensor([1.0]))
        env.reset()
        after = env.rollout(
            max_steps=4, policy=RandomMaskedPolicy(), break_when_any_done=False
        )
    finally:
        env.close()

    assert set(before["level_id"].reshape(-1).tolist()) == {first}
    assert set(after["level_id"].reshape(-1).tolist()) == {second}, (
        "workers kept serving the pre-fork distribution; the channel is not shared"
    )


def test_anchor_flag_defaults_true_without_a_league() -> None:
    """
    With a fixed opponent every episode is an anchor episode by definition.
    """
    env = TCGEnv(DECK, DECK, seed=0)
    reset = env.reset()
    env.close()

    assert bool(reset["opponent_is_anchor"].item()) is True
    assert int(reset["level_id"].item()) == NO_LEVEL


def test_curriculum_env_specs_stay_consistent() -> None:
    """
    The added keys must not break torchrl's spec contract.
    """
    # Reproduce the process state training actually runs in: importing torchrl
    # sets the global start method to "spawn", and ParallelEnv starts workers
    # lazily, so this is what they would be created under.
    previous = torch_mp.get_start_method(allow_none=True)
    torch_mp.set_start_method("spawn", force=True)

    index = make_index({"a": 2, "b": 2})
    handles = CurriculumHandles.allocate(CAPACITY)
    handles.publish(torch.tensor([index.pair_id(0, 1)]), torch.tensor([1.0]))
    env = make_curriculum_env(index, handles, seed=0)

    check_env_specs(env)
    env.close()


def test_rejects_non_fork_start_methods() -> None:
    """
    Spawned workers would get private copies and silently freeze the curriculum.
    """
    CurriculumHandles.require_shared_start_method("fork")
    for method in ("spawn", "forkserver"):
        with pytest.raises(ValueError, match="fork"):
            CurriculumHandles.require_shared_start_method(method)


def test_workers_track_a_republished_distribution_through_the_trainer_path() -> None:
    """
    The distribution must still reach workers built the way training builds them.

    :func:`test_level_updates_reach_running_parallel_workers` constructs the
    ``ParallelEnv`` directly and publishes a single slot at probability one, so
    it passes even when the channel is a per-worker *copy*: a copy holds the
    right values until the learner republishes, and that test's first publish
    happens before the fork.

    Training goes through :func:`~src.training.trainer.Trainer._make_vec_env`,
    which matters because importing torchrl sets the process-wide start method
    to ``spawn`` and ``ParallelEnv`` starts its workers lazily. Under ``spawn``
    the factories are pickled, every worker gets a private channel, and the
    curriculum silently freezes at whatever was published before collection --
    with ties among the initial ``inf`` scores broken by buffer position, that
    is a Zipf over ``pair_id`` rather than anything the learner computed.

    So this republishes *after* the workers are running and asserts the draws
    actually move, which a copied channel cannot do.
    """
    # Reproduce the process state training actually runs in: importing torchrl
    # sets the global start method to "spawn", and ParallelEnv starts workers
    # lazily, so this is what they would be created under.
    previous = torch_mp.get_start_method(allow_none=True)
    torch_mp.set_start_method("spawn", force=True)

    index = make_index({"a": 2, "b": 2})
    handles = CurriculumHandles.allocate(CAPACITY)
    pair_ids = torch.tensor([index.pair_id(j, k) for j in range(2) for k in range(2)])
    spread = torch.full((4,), 0.25)
    handles.publish(pair_ids, spread)

    trainer = Trainer(
        env_factories=[
            partial(make_curriculum_env, index, handles, 0),
            partial(make_curriculum_env, index, handles, 1),
        ],
        policy=RandomMaskedPolicy(),
        frames_per_batch=8,
        total_frames=8,
        mp_start_method="fork",
        serial_for_single=False,
    )
    env = trainer._make_vec_env()  # noqa: SLF001 - the construction under test
    try:
        env.reset()
        target = index.pair_id(1, 1)
        focused = torch.zeros(4)
        focused[pair_ids.tolist().index(target)] = 1.0
        handles.publish(pair_ids, focused)

        env.reset()
        drawn = env.rollout(
            max_steps=6, policy=RandomMaskedPolicy(), break_when_any_done=False
        )["level_id"]
    finally:
        env.close()
        torch_mp.set_start_method(previous or "fork", force=True)

    assert set(drawn.reshape(-1).tolist()) == {target}, (
        "workers kept serving the pre-fork distribution; the channel was copied, "
        "not shared, so the curriculum would be frozen for the whole run"
    )
