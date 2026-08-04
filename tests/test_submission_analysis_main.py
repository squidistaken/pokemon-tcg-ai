from __future__ import annotations

from submission_analysis.__main__ import _label_slug, _submission_folder_name, main
from submission_analysis.loading import CardIndex
from submission_analysis.parser import ParsedEpisode


def test_run_deck_report_no_replays_returns_0_without_crashing(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr("submission_analysis.submissions.repo_root", lambda: tmp_path)

    exit_code = main(["deck-report"])

    assert exit_code == 0


def test_run_deck_report_full_flow_writes_report_and_plots(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr("submission_analysis.submissions.repo_root", lambda: tmp_path)
    submission_dir = tmp_path / "logs" / "replays" / "12345"
    submission_dir.mkdir(parents=True)
    (submission_dir / "manifest.json").write_text("{}", encoding="utf-8")

    episode = ParsedEpisode(
        episode_id=1,
        result="loss",
        opponent_team="alice",
        had_basic_pokemon=False,
        cards_seen={100},
        cards_played=set(),
    )
    monkeypatch.setattr(
        "submission_analysis.__main__.load_parsed_episodes",
        lambda dirs: [episode],  # noqa: ARG005
    )
    monkeypatch.setattr(
        "submission_analysis.__main__.load_card_index",
        lambda: CardIndex(cards={}, attacks={}),
    )
    output_dir = tmp_path / "outputs" / "submission_analysis" / "12345"

    exit_code = main(["deck-report", "--deck", str(tmp_path / "missing-deck.csv")])

    assert exit_code == 0
    assert list(output_dir.glob("report_*.txt"))
    assert (output_dir / "loss_causes.png").is_file()


def test_run_deck_report_creates_a_separate_folder_per_submission_by_default(
    tmp_path, monkeypatch
) -> None:
    """Submissions can carry different checkpoints/decks, so each gets its
    own report rather than being blended into one combined view."""
    monkeypatch.setattr("submission_analysis.submissions.repo_root", lambda: tmp_path)
    replays_dir = tmp_path / "logs" / "replays"
    for ref in ("111", "222"):
        submission_dir = replays_dir / ref
        submission_dir.mkdir(parents=True)
        (submission_dir / "manifest.json").write_text("{}", encoding="utf-8")

    episode = ParsedEpisode(episode_id=1, result="win", opponent_team="alice")
    monkeypatch.setattr(
        "submission_analysis.__main__.load_parsed_episodes",
        lambda dirs: [episode],  # noqa: ARG005
    )
    monkeypatch.setattr(
        "submission_analysis.__main__.load_card_index",
        lambda: CardIndex(cards={}, attacks={}),
    )

    exit_code = main(["deck-report"])

    assert exit_code == 0
    for ref in ("111", "222"):
        output_dir = tmp_path / "outputs" / "submission_analysis" / ref
        assert list(output_dir.glob("report_*.txt"))
    assert not (tmp_path / "outputs" / "submission_analysis" / "all").exists()


def test_run_deck_report_rating_history_plot_is_shared_not_per_submission(
    tmp_path, monkeypatch
) -> None:
    """The rating-history trend spans every submission `status` has ever
    fetched, so it belongs once at the top level, not copied into each
    submission's own folder."""
    monkeypatch.setattr("submission_analysis.submissions.repo_root", lambda: tmp_path)
    replays_dir = tmp_path / "logs" / "replays"
    for ref in ("111", "222"):
        submission_dir = replays_dir / ref
        submission_dir.mkdir(parents=True)
        (submission_dir / "manifest.json").write_text("{}", encoding="utf-8")

    history_path = tmp_path / "logs" / "kaggle_rating_history.csv"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.write_text(
        "fetched_at_utc,kaggle_ref,label,status,public_score\n"
        "2026-08-01T00:00:00+00:00,111,agent,COMPLETE,400.0\n",
        encoding="utf-8",
    )

    episode = ParsedEpisode(episode_id=1, result="win", opponent_team="alice")
    monkeypatch.setattr(
        "submission_analysis.__main__.load_parsed_episodes",
        lambda dirs: [episode],  # noqa: ARG005
    )
    monkeypatch.setattr(
        "submission_analysis.__main__.load_card_index",
        lambda: CardIndex(cards={}, attacks={}),
    )

    exit_code = main(["deck-report"])

    assert exit_code == 0
    assert (
        tmp_path / "outputs" / "submission_analysis" / "rating_history.png"
    ).is_file()
    for ref in ("111", "222"):
        assert not (
            tmp_path / "outputs" / "submission_analysis" / ref / "rating_history.png"
        ).exists()


def test_run_deck_report_submission_ref_filters_which_folders_get_created(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr("submission_analysis.submissions.repo_root", lambda: tmp_path)
    replays_dir = tmp_path / "logs" / "replays"
    for ref in ("111", "222"):
        submission_dir = replays_dir / ref
        submission_dir.mkdir(parents=True)
        (submission_dir / "manifest.json").write_text("{}", encoding="utf-8")

    episode = ParsedEpisode(episode_id=1, result="win", opponent_team="alice")
    monkeypatch.setattr(
        "submission_analysis.__main__.load_parsed_episodes",
        lambda dirs: [episode],  # noqa: ARG005
    )
    monkeypatch.setattr(
        "submission_analysis.__main__.load_card_index",
        lambda: CardIndex(cards={}, attacks={}),
    )

    exit_code = main(["deck-report", "--submission-ref", "111"])

    assert exit_code == 0
    filtered_dir = tmp_path / "outputs" / "submission_analysis" / "111"
    other_dir = tmp_path / "outputs" / "submission_analysis" / "222"
    assert list(filtered_dir.glob("report_*.txt"))
    assert not other_dir.exists()


def test_label_slug_normalizes_punctuation_and_case() -> None:
    assert _label_slug("mewtwo-sampled") == "mewtwo-sampled"
    assert _label_slug("N’s Reshiram") == "ns-reshiram"


def test_label_slug_truncates_long_labels() -> None:
    slug = _label_slug("curriculum-8h-s0 22M frames, ns-zoroark, extra detail")
    assert len(slug) <= 40
    assert slug.startswith("curriculum-8h-s0-22m-frames-ns-zoroark")


def test_submission_folder_name_uses_the_label_when_known() -> None:
    assert (
        _submission_folder_name("55162026", {55162026: "mewtwo-sampled"})
        == "55162026-mewtwo-sampled"
    )


def test_submission_folder_name_falls_back_to_the_bare_ref_when_unknown() -> None:
    assert _submission_folder_name("55162026", {}) == "55162026"


def test_run_deck_report_folder_name_includes_the_submission_label(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr("submission_analysis.submissions.repo_root", lambda: tmp_path)
    submission_dir = tmp_path / "logs" / "replays" / "55162026"
    submission_dir.mkdir(parents=True)
    (submission_dir / "manifest.json").write_text("{}", encoding="utf-8")

    history_path = tmp_path / "logs" / "kaggle_rating_history.csv"
    history_path.write_text(
        "fetched_at_utc,kaggle_ref,label,status,public_score\n"
        "2026-08-01T00:00:00+00:00,55162026,mewtwo-sampled,COMPLETE,500.0\n",
        encoding="utf-8",
    )

    episode = ParsedEpisode(episode_id=1, result="win", opponent_team="alice")
    monkeypatch.setattr(
        "submission_analysis.__main__.load_parsed_episodes",
        lambda dirs: [episode],  # noqa: ARG005
    )
    monkeypatch.setattr(
        "submission_analysis.__main__.load_card_index",
        lambda: CardIndex(cards={}, attacks={}),
    )

    exit_code = main(["deck-report"])

    assert exit_code == 0
    output_dir = (
        tmp_path / "outputs" / "submission_analysis" / "55162026-mewtwo-sampled"
    )
    assert list(output_dir.glob("report_*.txt"))
