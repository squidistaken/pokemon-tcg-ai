from __future__ import annotations

import json

from submission_analysis.loading import (
    discover_submission_dirs,
    load_card_index,
    load_deck,
    load_manifest,
    load_parsed_episodes,
    load_rating_history,
)


def test_load_card_index_resolves_a_known_card_and_attack() -> None:
    index = load_card_index()

    assert len(index.cards) > 0
    assert len(index.attacks) > 0


def test_load_deck_reads_one_card_id_per_line(tmp_path) -> None:
    deck_path = tmp_path / "deck.csv"
    deck_path.write_text("1\n2\n\n3\n", encoding="utf-8")

    assert load_deck(deck_path) == [1, 2, 3]


def test_discover_submission_dirs_requires_a_manifest(tmp_path) -> None:
    (tmp_path / "with_manifest").mkdir()
    (tmp_path / "with_manifest" / "manifest.json").write_text("{}", encoding="utf-8")
    (tmp_path / "without_manifest").mkdir()

    dirs = discover_submission_dirs(tmp_path)

    assert dirs == [tmp_path / "with_manifest"]


def test_discover_submission_dirs_missing_root_returns_empty(tmp_path) -> None:
    assert discover_submission_dirs(tmp_path / "does-not-exist") == []


def test_load_manifest_keys_by_int_episode_id(tmp_path) -> None:
    submission_dir = tmp_path / "12345"
    submission_dir.mkdir()
    (submission_dir / "manifest.json").write_text(
        json.dumps({"1": {"our_index": 0, "result": "win", "opponent_team": "alice"}}),
        encoding="utf-8",
    )

    manifest = load_manifest(submission_dir)

    assert manifest == {1: {"our_index": 0, "result": "win", "opponent_team": "alice"}}


def test_load_manifest_missing_file_returns_empty(tmp_path) -> None:
    assert load_manifest(tmp_path / "missing") == {}


def test_load_parsed_episodes_skips_replays_missing_from_manifest(tmp_path) -> None:
    submission_dir = tmp_path / "12345"
    submission_dir.mkdir()
    (submission_dir / "manifest.json").write_text(
        json.dumps(
            {
                "1": {"our_index": 0, "result": "win", "opponent_team": "alice"},
                "2": {"our_index": 1, "result": "loss", "opponent_team": "bob"},
            }
        ),
        encoding="utf-8",
    )
    # Only episode 1's replay was actually downloaded.
    (submission_dir / "episode-1-replay.json").write_text(
        json.dumps({"steps": []}), encoding="utf-8"
    )

    episodes = load_parsed_episodes([submission_dir])

    assert [episode.episode_id for episode in episodes] == [1]
    assert episodes[0].result == "win"
    assert episodes[0].opponent_team == "alice"


def test_load_rating_history_parses_scores_and_blanks(tmp_path) -> None:
    path = tmp_path / "history.csv"
    path.write_text(
        "fetched_at_utc,kaggle_ref,label,status,public_score\n"
        "2026-08-01T00:00:00+00:00,1,agent,COMPLETE,446.8\n"
        "2026-08-01T00:00:01+00:00,2,agent2,ERROR,\n",
        encoding="utf-8",
    )

    rows = load_rating_history(path)

    assert rows[0].public_score == 446.8
    assert rows[1].public_score is None


def test_load_rating_history_missing_file_returns_empty(tmp_path) -> None:
    assert load_rating_history(tmp_path / "missing.csv") == []
