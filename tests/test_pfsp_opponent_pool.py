import pickle
from pathlib import Path

import pytest
from omegaconf import OmegaConf
from torchrl.envs import TransformedEnv
from torchrl.envs.transforms import ActionMask

from src.env.pfsp_opponent_pool import PFSPOpponentPool
from src.env.random_opponent import RandomOpponent
from src.env.snapshot_opponent_pool import SnapshotOpponentPool
from src.env.tcg_env import TCGEnv
from src.policies.random_masked_policy import RandomMaskedPolicy
from src.training.env_factory import make_env_factories
from src.training.self_play import build_opponent_factory
from tests.conftest import structured_env_cfg
from tests.test_self_play import fake_loader, write_snapshot_files
from tests.test_tcg_env import DECK


def make_pool(tmp_path: Path, pool_size: int = 3, **kwargs) -> PFSPOpponentPool:
    """
    Build a PFSP league backed by placeholder snapshot files.

    :param tmp_path: Directory scanned for snapshots.
    :param pool_size: Number of most-recent snapshots kept.
    :param kwargs: Overrides forwarded to :class:`PFSPOpponentPool`.
    :return: The league under test.
    """
    return PFSPOpponentPool(
        checkpoint_dir=tmp_path,
        load_snapshot=fake_loader,
        warmup_opponents=[RandomOpponent(seed=0)],
        pool_size=pool_size,
        seed=0,
        **kwargs,
    )


def play(pool: PFSPOpponentPool, episodes: int, loses_to: str | None = None) -> None:
    """
    Drive the pool through episodes, as the environment would.

    The simulated learner beats every member except ``loses_to``, so the
    recorded win rates separate cleanly.

    :param pool: League to drive.
    :param episodes: Number of episodes to play.
    :param loses_to: Member key the learner always loses to; None wins all.
    """
    for _ in range(episodes):
        pool.on_reset()
        pool.record_outcome(-1.0 if pool.active_key == loses_to else 1.0)


def weight_of(pool: PFSPOpponentPool) -> dict[str, float]:
    """
    Current sampling weight per member key.

    :param pool: League to read.
    :return: Mapping from member key to weight.
    """
    return dict(zip(pool.member_keys, pool.weights, strict=True))


def test_uniform_pool_keeps_uniform_weights(tmp_path: Path) -> None:
    """
    The base league must be unaffected by the weighting hook.
    """
    write_snapshot_files(tmp_path, [100, 200])
    pool = SnapshotOpponentPool(
        checkpoint_dir=tmp_path,
        load_snapshot=fake_loader,
        warmup_opponents=[RandomOpponent(seed=0)],
        pool_size=3,
        seed=0,
    )
    pool.on_reset()

    assert len(pool.opponents) == 3
    assert pool.weights == (1.0, 1.0, 1.0)


def test_unplayed_members_start_at_the_prior(tmp_path: Path) -> None:
    """
    With no games recorded every member sits at p = 0.5, so weights are equal.
    """
    write_snapshot_files(tmp_path, [100, 200])
    pool = make_pool(tmp_path)
    pool.on_reset()

    assert len(pool.weights) == 3
    assert pool.weights == pytest.approx((pool.weights[0],) * 3)


def test_losing_to_a_member_raises_its_weight(tmp_path: Path) -> None:
    """
    Under `hard`, a member that beats the learner outweighs the ones it beats.
    """
    write_snapshot_files(tmp_path, [100, 200])
    pool = make_pool(tmp_path, min_weight=0.01, prior_games=1.0)
    pool.on_reset()
    nemesis = pool.member_keys[-1]

    play(pool, episodes=400, loses_to=nemesis)

    weights = weight_of(pool)
    others = [weights[key] for key in pool.member_keys if key != nemesis]
    assert weights[nemesis] > max(others)
    assert pool.win_rates[nemesis] < min(
        pool.win_rates[key] for key in pool.member_keys if key != nemesis
    )


def test_min_weight_keeps_beaten_members_reachable(tmp_path: Path) -> None:
    """
    Members the learner always beats must keep a share, not decay to zero.

    Without the floor, ``(1 - p)`` drives the weight of a fully-beaten member
    toward zero, it stops being sampled, and its win rate freezes at a value
    that no longer describes the current policy.
    """
    floor = 0.05
    write_snapshot_files(tmp_path, [100])
    pool = make_pool(tmp_path, min_weight=floor, prior_games=1.0)

    play(pool, episodes=400)

    assert min(pool.win_rates.values()) > 0.95, "setup failed: members not saturated"
    assert min(pool.weights) == pytest.approx(floor, abs=0.01)
    assert all(weight > 0.0 for weight in pool.weights)


def test_even_weighting_peaks_at_a_level_matchup(tmp_path: Path) -> None:
    """
    Under `even`, a 50/50 member outweighs one that is always beaten.
    """
    write_snapshot_files(tmp_path, [100, 200])
    pool = make_pool(tmp_path, weighting="even", min_weight=0.01)
    pool.on_reset()
    level = pool.member_keys[-1]

    for index in range(400):
        pool.on_reset()
        if pool.active_key == level:
            pool.record_outcome(1.0 if index % 2 == 0 else -1.0)
        else:
            pool.record_outcome(1.0)

    weights = weight_of(pool)
    others = [weights[key] for key in pool.member_keys if key != level]
    assert weights[level] > max(others)


def test_draws_count_as_half(tmp_path: Path) -> None:
    """
    Drawn episodes must hold the win rate at 0.5, not push it to an extreme.
    """
    pool = make_pool(tmp_path)
    for _ in range(50):
        pool.on_reset()
        pool.record_outcome(0.0)

    assert pool.win_rates
    assert all(rate == pytest.approx(0.5) for rate in pool.win_rates.values())


def test_records_survive_eviction_and_reload(tmp_path: Path) -> None:
    """
    History is keyed by snapshot path, so an evicted member resumes its record.
    """
    write_snapshot_files(tmp_path, [100])
    pool = make_pool(tmp_path, pool_size=1)
    pool.on_reset()
    first = pool.member_keys[-1]

    play(pool, episodes=50, loses_to=first)
    recorded = pool.records[first]
    assert recorded[1] > 0

    write_snapshot_files(tmp_path, [200])
    pool.on_reset()

    assert first not in pool.member_keys
    assert pool.records[first] == recorded


def test_active_key_tracks_the_drawn_member(tmp_path: Path) -> None:
    """
    After a reset the recorded key must name the member that will play.
    """
    write_snapshot_files(tmp_path, [100, 200])
    pool = make_pool(tmp_path)

    for _ in range(10):
        pool.on_reset()
        index = pool.member_keys.index(pool.active_key)
        assert pool.opponents[index] is pool.active


def test_rejects_unknown_weighting(tmp_path: Path) -> None:
    """
    An unsupported scheme must fail at construction, not silently fall back.
    """
    with pytest.raises(ValueError, match="unknown weighting"):
        make_pool(tmp_path, weighting="softmax")


class RecordingOpponent(RandomOpponent):
    """
    Random opponent that logs every outcome the environment reports to it.
    """

    def __init__(self, seed: int | None = None) -> None:
        """
        :param seed: Seed for the underlying random play.
        """
        super().__init__(seed)
        self.outcomes: list[float] = []

    def record_outcome(self, reward: float) -> None:
        """
        Log one reported terminal reward.

        :param reward: Terminal reward from the agent's perspective.
        """
        self.outcomes.append(reward)


def test_env_reports_terminal_outcome_to_the_opponent() -> None:
    """
    A finished battle must reach the opponent's hook exactly once.
    """
    opponent = RecordingOpponent(seed=0)
    env = TransformedEnv(TCGEnv(DECK, DECK, seed=0, opponent=opponent), ActionMask())
    rollout = env.rollout(max_steps=5000, policy=RandomMaskedPolicy())
    terminated = bool(rollout["next", "terminated"][-1].item())
    env.close()

    assert terminated, "episode did not terminate; the outcome hook is untested"
    assert len(opponent.outcomes) == 1
    assert opponent.outcomes[0] in (1.0, -1.0, 0.0)


def selfplay_cfg(tmp_path: Path, structured_model_cfg, sampling: str):
    """
    Build a self-play config selecting a league sampling strategy.

    :param tmp_path: Directory used for snapshots.
    :param structured_model_cfg: Model section from the shared fixture.
    :param sampling: Value for ``train.opponent_sampling``.
    :return: Merged config with ``env``, ``model`` and ``train`` sections.
    """
    return OmegaConf.merge(
        structured_env_cfg(num_workers=1),
        structured_model_cfg,
        OmegaConf.create(
            {
                "train": {
                    "warmup_opponent": "random",
                    "snapshot_interval": 100,
                    "checkpoint_dir": str(tmp_path),
                    "pool_size": 3,
                    "opponent_sampling": sampling,
                }
            }
        ),
    )


def test_factory_builds_a_pfsp_pool_and_stays_picklable(
    tmp_path: Path, structured_model_cfg
) -> None:
    """
    ParallelEnv pickles the factory into each worker, so it must survive that.
    """
    cfg = selfplay_cfg(tmp_path, structured_model_cfg, sampling="pfsp")
    assert make_env_factories(cfg)

    factory = build_opponent_factory(
        cfg, obs_spec=None, action_spec=None, checkpoint_dir=tmp_path
    )
    assert factory is not None
    pickle.loads(pickle.dumps(factory))
    assert isinstance(factory(), PFSPOpponentPool)


def test_factory_builds_a_uniform_pool_by_default(
    tmp_path: Path, structured_model_cfg
) -> None:
    """
    Without the flag the league must stay the plain uniform pool.
    """
    cfg = selfplay_cfg(tmp_path, structured_model_cfg, sampling="uniform")
    factory = build_opponent_factory(
        cfg, obs_spec=None, action_spec=None, checkpoint_dir=tmp_path
    )
    assert factory is not None
    pool = factory()
    assert isinstance(pool, SnapshotOpponentPool)
    assert not isinstance(pool, PFSPOpponentPool)


def test_factory_rejects_unknown_sampling(tmp_path: Path, structured_model_cfg) -> None:
    """
    A typo in `opponent_sampling` must not silently train uniformly.
    """
    cfg = selfplay_cfg(tmp_path, structured_model_cfg, sampling="prioritised")
    with pytest.raises(ValueError, match="unknown opponent_sampling"):
        build_opponent_factory(
            cfg, obs_spec=None, action_spec=None, checkpoint_dir=tmp_path
        )
