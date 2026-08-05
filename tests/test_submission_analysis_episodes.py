from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime

import pytest

from submission_analysis.episodes import (
    EpisodeOutcome,
    OutcomeSummary,
    _result_for_reward,
    download_replays,
    resolve_submission_refs,
    summarize,
    to_outcomes,
)


@dataclass
class _FakeAgent:
    submission_id: int
    reward: float
    team_name: str
    index: int = 0


@dataclass
class _FakeEpisode:
    id: int
    create_time: datetime
    agents: list[_FakeAgent]


NOW = datetime(2026, 8, 1, 12, 0, 0, tzinfo=UTC)
OUR_REF = 55162026


@pytest.mark.parametrize(
    ("reward", "expected"),
    [(1.0, "win"), (-1.0, "loss"), (0.0, "draw")],
)
def test_result_for_reward(reward: float, expected: str) -> None:
    assert _result_for_reward(reward) == expected


def test_to_outcomes_matches_our_agent_by_submission_id() -> None:
    episode = _FakeEpisode(
        id=1,
        create_time=NOW,
        agents=[
            _FakeAgent(submission_id=OUR_REF, reward=1.0, team_name="us", index=1),
            _FakeAgent(
                submission_id=999, reward=-1.0, team_name="opponent-team", index=0
            ),
        ],
    )

    (outcome,) = to_outcomes(OUR_REF, [episode])

    assert outcome == EpisodeOutcome(
        episode_id=1,
        create_time=NOW,
        result="win",
        our_reward=1.0,
        opponent_team="opponent-team",
        opponent_reward=-1.0,
        our_index=1,
    )


def test_to_outcomes_skips_episode_missing_our_agent() -> None:
    """An episode with no agent matching our submission ref is not our data."""
    episode = _FakeEpisode(
        id=2,
        create_time=NOW,
        agents=[
            _FakeAgent(submission_id=111, reward=1.0, team_name="a"),
            _FakeAgent(submission_id=222, reward=-1.0, team_name="b"),
        ],
    )

    assert to_outcomes(OUR_REF, [episode]) == []


def test_to_outcomes_skips_episode_missing_an_opponent() -> None:
    """A malformed single-agent episode has nothing to compare against."""
    episode = _FakeEpisode(
        id=3,
        create_time=NOW,
        agents=[_FakeAgent(submission_id=OUR_REF, reward=1.0, team_name="us")],
    )

    assert to_outcomes(OUR_REF, [episode]) == []


def test_summarize_aggregates_overall_and_by_opponent() -> None:
    outcomes = [
        EpisodeOutcome(1, NOW, "win", 1.0, "alice", -1.0, our_index=1),
        EpisodeOutcome(2, NOW, "loss", -1.0, "alice", 1.0, our_index=1),
        EpisodeOutcome(3, NOW, "win", 1.0, "bob", -1.0, our_index=0),
        EpisodeOutcome(4, NOW, "draw", 0.0, "bob", 0.0, our_index=0),
    ]

    summary = summarize(OUR_REF, outcomes)

    assert summary == OutcomeSummary(
        submission_ref=OUR_REF,
        total=4,
        wins=2,
        losses=1,
        draws=1,
        win_rate=0.5,
        by_opponent={
            "alice": {"wins": 1, "losses": 1, "draws": 0},
            "bob": {"wins": 1, "losses": 0, "draws": 1},
        },
    )


def test_summarize_handles_no_outcomes() -> None:
    summary = summarize(OUR_REF, [])

    assert summary.total == 0
    assert summary.win_rate == 0.0
    assert summary.by_opponent == {}


def _install_fake_kaggle_api(
    monkeypatch: pytest.MonkeyPatch, replay_calls: list
) -> None:
    """Stand in for KaggleApi in download_replays()."""

    class _FakeApi:
        def authenticate(self) -> None:
            pass

        def competition_episode_replay(self, episode_id, path, quiet):  # noqa: ARG002, PLR6301
            replay_calls.append(episode_id)

    monkeypatch.setattr("submission_analysis.episodes.KaggleApi", _FakeApi)


def test_download_replays_writes_a_manifest_with_our_index_and_outcome(
    monkeypatch, tmp_path
) -> None:
    replay_calls: list = []
    _install_fake_kaggle_api(monkeypatch, replay_calls)
    outcomes = [
        EpisodeOutcome(1, NOW, "win", 1.0, "alice", -1.0, our_index=1),
        EpisodeOutcome(2, NOW, "loss", -1.0, "bob", 1.0, our_index=0),
    ]

    download_replays(OUR_REF, outcomes, tmp_path)

    assert replay_calls == [1, 2]
    manifest = json.loads((tmp_path / str(OUR_REF) / "manifest.json").read_text())
    assert manifest["1"]["our_index"] == 1
    assert manifest["1"]["result"] == "win"
    assert manifest["1"]["opponent_team"] == "alice"
    assert manifest["2"]["our_index"] == 0
    assert manifest["2"]["result"] == "loss"


def test_download_replays_merges_into_an_existing_manifest(
    monkeypatch, tmp_path
) -> None:
    """Re-running --download-replays adds to the manifest instead of
    clobbering episodes downloaded by an earlier run."""
    replay_calls: list = []
    _install_fake_kaggle_api(monkeypatch, replay_calls)
    download_replays(
        OUR_REF,
        [EpisodeOutcome(1, NOW, "win", 1.0, "alice", -1.0, our_index=1)],
        tmp_path,
    )

    download_replays(
        OUR_REF,
        [EpisodeOutcome(2, NOW, "loss", -1.0, "bob", 1.0, our_index=0)],
        tmp_path,
    )

    manifest = json.loads((tmp_path / str(OUR_REF) / "manifest.json").read_text())
    assert set(manifest.keys()) == {"1", "2"}


def test_download_replays_skips_episodes_already_on_disk(monkeypatch, tmp_path) -> None:
    """A replay is immutable once its episode is DONE, so a file already
    downloaded by a previous run shouldn't cost another API call."""
    replay_calls: list = []
    _install_fake_kaggle_api(monkeypatch, replay_calls)
    submission_dir = tmp_path / str(OUR_REF)
    submission_dir.mkdir(parents=True)
    (submission_dir / "episode-1-replay.json").write_text("{}", encoding="utf-8")
    outcomes = [
        EpisodeOutcome(1, NOW, "win", 1.0, "alice", -1.0, our_index=1),
        EpisodeOutcome(2, NOW, "loss", -1.0, "bob", 1.0, our_index=0),
    ]

    download_replays(OUR_REF, outcomes, tmp_path)

    assert replay_calls == [2]
    manifest = json.loads((submission_dir / "manifest.json").read_text())
    assert set(manifest.keys()) == {"1", "2"}


def test_resolve_submission_refs_uses_fetch_submissions_seam(monkeypatch) -> None:
    @dataclass
    class _FakeStatus:
        name: str

    @dataclass
    class _FakeSubmission:
        ref: int
        description: str
        file_name: str
        date: datetime
        status: _FakeStatus
        public_score: str
        error_description: str = ""

    submission = _FakeSubmission(
        ref=OUR_REF,
        description="agent",
        file_name="agent.tar.gz",
        date=NOW,
        status=_FakeStatus("COMPLETE"),
        public_score="500.0",
    )

    def _fake_fetch(competition: str) -> list[_FakeSubmission]:  # noqa: ARG001
        return [submission]

    monkeypatch.setattr("submission_analysis.episodes.fetch_submissions", _fake_fetch)

    refs = resolve_submission_refs("pokemon-tcg-ai-battle", most_recent_n=1)

    assert refs == [OUR_REF]
