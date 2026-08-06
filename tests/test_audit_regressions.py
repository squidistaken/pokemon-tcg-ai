"""
Regression tests for the issue #76 audit fixes.

Each test here pins behaviour that was wrong before the audit and that the
existing suite did not cover; see AUDIT.md for the reasoning behind each fix.
"""

from pathlib import Path

import pytest
import torch
from omegaconf import DictConfig, OmegaConf

import main
from src.env.battle_handle import BattleHandle
from src.env.deck import load_deck
from src.env.structured_observation_encoder import StructuredObservationEncoder
from src.policies.greedy_policy_opponent import GreedyPolicyOpponent
from src.policies.ppo_actor import build_actor_critic
from src.training.env_factory import load_deck_pool
from tests.conftest import DECK_PATH, MAX_OPTIONS

DECK = load_deck(DECK_PATH)


def _first_multiselect(deck: list[int], battles: int = 20, max_steps: int = 4000):
    """
    Drive battles until the engine offers a ``maxCount > 1`` selection.

    :param deck: Deck given to both seats.
    :param battles: Battles to play before giving up.
    :param max_steps: Selections to try per battle.
    :return: ``(handle, observation)`` at that selection, or ``(handle, None)``.
    """
    import random

    rng = random.Random(0)
    handle = BattleHandle()
    for _ in range(battles):
        observation = handle.start(deck, list(deck))
        for _ in range(max_steps):
            state = observation.current
            if state is not None and state.result != -1:
                break
            select = observation.select
            assert select is not None
            if select.maxCount > 1:
                return handle, observation
            count = rng.randint(select.minCount, select.maxCount)
            observation = handle.select(rng.sample(range(len(select.option)), count))
        handle.finish()
    return handle, None


def test_greedy_opponent_rescores_between_picks(
    structured_model_cfg: DictConfig, structured_obs_spec, action_spec
) -> None:
    """
    The league opponent re-scores the policy once per pick of a multi-select.
    """
    handle, observation = _first_multiselect(DECK)
    try:
        if observation is None:
            pytest.skip("no multi-select selection reached under random play")
        actor_critic = build_actor_critic(
            structured_model_cfg, structured_obs_spec, action_spec
        )
        opponent = GreedyPolicyOpponent(
            actor_critic, StructuredObservationEncoder(max_options=MAX_OPTIONS)
        )
        calls = 0
        original = actor_critic.policy_logits

        def counting_policy_logits(tensordict):
            nonlocal calls
            calls += 1
            return original(tensordict)

        actor_critic.policy_logits = counting_policy_logits
        picks = opponent(observation)
    finally:
        handle.finish()

    select = observation.select
    assert select is not None
    assert select.minCount <= len(picks) <= select.maxCount
    assert len(set(picks)) == len(picks), "engine rejects duplicate option indices"
    # One scoring pass per pick, plus the pass that decided to stop when the
    # opponent stopped short of maxCount.
    assert calls == len(picks) + (1 if len(picks) < select.maxCount else 0)
    assert calls >= 1


def test_greedy_select_matches_per_pick_loop() -> None:
    """
    The single-pass helper still resolves a selection the way it always did.
    """
    logits = torch.full((MAX_OPTIONS + 1,), -1e9)
    logits[:4] = torch.tensor([3.0, 2.0, 1.0, 0.5])
    logits[MAX_OPTIONS] = 1.5
    picks = GreedyPolicyOpponent.greedy_select(
        logits, n_options=4, min_count=1, max_count=3
    )
    assert picks == [0, 1]


def test_battle_start_names_the_violated_deck_rule() -> None:
    """
    An illegal deck is rejected by rule name, not by a bare engine error code.
    """
    # 60 copies of one Basic Pokemon: breaks only the 4-copy rule, so the
    # message must name that one rather than an earlier-checked violation.
    illegal = [721] * 60
    handle = BattleHandle()
    try:
        with pytest.raises(ValueError, match="more than 4 copies"):
            handle.start(illegal, list(DECK))
    finally:
        handle.finish()


def test_read_deck_csv_tolerates_a_header_row(tmp_path: Path, monkeypatch) -> None:
    """
    The submission's deck reader accepts the same files the training loader does.
    """
    deck_file = tmp_path / "deck.csv"
    deck_file.write_text("card_id\n" + "\n".join(str(card) for card in DECK) + "\n")
    monkeypatch.chdir(tmp_path)
    assert main.read_deck_csv() == DECK


def test_submission_runtime_serves_the_default_pointer_head(
    pointer_model_cfg: DictConfig, structured_obs_spec, action_spec
) -> None:
    """
    The Kaggle runtime rebuilds the *training default* architecture.

    ``conf/model/default.yaml`` selects the MLP backbone with
    ``model/head=pointer`` (``PointerHead``), where the adapter supplies the
    per-option tokens and the backbone adds none of its own. The runtime had no
    ``PointerHead`` at all — it raised "cannot rebuild policy head" — and its
    ``MLPBackbone`` overwrote the adapter-sourced ``produces_option_repr`` with
    the backbone's own ``option_tokens`` (false here), so it served no
    ``option_repr`` even once the head existed. The default-trained model could
    therefore not be packaged for the competition at all.
    """
    from omegaconf import OmegaConf

    from submission.runtime import Policy

    torch.manual_seed(23)
    actor_critic = build_actor_critic(
        pointer_model_cfg, structured_obs_spec, action_spec
    ).eval()
    portable = {
        "model": OmegaConf.to_container(pointer_model_cfg.model, resolve=True),
        "env": {"encoder": "structured", "max_options": MAX_OPTIONS},
        "inference": {"action_selection": "sample"},
    }
    policy = Policy({"state_dict": actor_critic.state_dict()}, portable)
    assert policy.model.backbone.produces_option_repr

    fixtures = torch.load(
        Path(__file__).parent / "fixtures" / "observations.pt", weights_only=False
    )
    for case_name in fixtures.keys():  # noqa: SIM118 - TensorDict iteration differs.
        case = fixtures[case_name]
        expected = actor_critic.policy_logits(case)
        actual = policy.model.policy_logits(case["observation"].to_dict())
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_truncated_episodes_are_not_counted_as_draws() -> None:
    """
    Win/draw rates cover decided episodes only.
    """
    from src.training.trainer import Trainer

    # 4 episodes: 1 win, 1 loss, 2 truncated (no result, reward 0).
    metrics = Trainer._metrics(  # noqa: SLF001
        frames=100, episodes=4, wins=1, draws=0, elapsed=1.0, truncations=2
    )
    assert metrics["win_rate"] == 0.5, "win rate is over the 2 decided episodes"
    assert metrics["draw_rate"] == 0.0, "truncations are not draws"
    assert metrics["truncation_rate"] == 0.5


def _pool_cfg(deck_pool: str, **overrides) -> DictConfig:
    """
    Build a minimal ``env`` config addressing a deck pool.

    :param deck_pool: Deck-pool spec.
    :param overrides: Extra ``env`` entries.
    :return: Config with ``env`` and a seed.
    """
    env = {"deck_pool": deck_pool, "deck_holdout_frac": 0.0, "deck_split_seed": 0}
    env.update(overrides)
    return OmegaConf.create({"seed": 0, "env": env})


def test_deck_pool_width_narrows_the_shared_train_split(tmp_path: Path) -> None:
    """
    ``deck_pool_width`` is honoured by ``load_deck_pool`` itself.
    """
    for archetype in range(4):
        archetype_dir = tmp_path / f"arch-{archetype}"
        archetype_dir.mkdir()
        (archetype_dir / "d.csv").write_text(
            "\n".join(str(archetype + 1) for _ in range(60))
        )

    full, full_paths = load_deck_pool(_pool_cfg(str(tmp_path)), deck_split="train")
    narrow, narrow_paths = load_deck_pool(
        _pool_cfg(str(tmp_path), deck_pool_width=2), deck_split="train"
    )
    assert len(full) == 4
    assert len(narrow) == 2
    # Decks and their paths must stay aligned: the curriculum's archetype index
    # is built from the paths and addresses the decks by position.
    assert len(narrow) == len(narrow_paths)
    assert set(narrow_paths) <= set(full_paths)


def test_in_place_pick_count_update_matches_a_full_re_encode() -> None:
    """
    The cached multi-select update writes the field a re-encode would write.
    """
    handle, observation = _first_multiselect(DECK)
    try:
        if observation is None:
            pytest.skip("no multi-select selection appeared in the sampled battles")
        encoder = StructuredObservationEncoder(max_options=MAX_OPTIONS)
        assert observation.current is not None
        seat = observation.current.yourIndex
        cached = encoder.encode(observation, seat, 0)
        for count in (1, 2, 3):
            expected = encoder.encode(observation, seat, count)
            assert encoder.update_already_chosen_option_count(cached, count)
            assert torch.equal(cached["globals"], expected["globals"])
    finally:
        handle.finish()


def test_submission_observation_schema_matches_the_competition_api() -> None:
    """
    ``submission/cg_api.py`` stays field-compatible with ``cg.api``.
    """
    import dataclasses

    from cg import api as competition_api
    from submission import cg_api as shipped

    for name in (
        "Card",
        "Pokemon",
        "PlayerState",
        "State",
        "Option",
        "SelectData",
        "Observation",
    ):
        expected = {
            field.name for field in dataclasses.fields(getattr(competition_api, name))
        }
        shipped_fields = {
            field.name for field in dataclasses.fields(getattr(shipped, name))
        }
        assert shipped_fields == expected, f"{name} has drifted from cg.api"

    for enum_name in ("AreaType", "OptionType"):
        expected_members = {
            member.name: int(member) for member in getattr(competition_api, enum_name)
        }
        shipped_members = {
            member.name: int(member) for member in getattr(shipped, enum_name)
        }
        assert shipped_members == expected_members, (
            f"{enum_name} has drifted from cg.api"
        )


def test_pending_episodes_are_skipped_not_scored_as_draws() -> None:
    """
    ``to_outcomes`` skips episodes whose rewards have not landed yet.
    """
    from dataclasses import dataclass
    from datetime import UTC, datetime

    from submission_analysis.episodes import summarize, to_outcomes

    @dataclass
    class _Agent:
        submission_id: int
        reward: float | None
        team_name: str
        index: int = 0

    @dataclass
    class _Episode:
        id: int
        create_time: datetime
        agents: list

    now = datetime(2026, 8, 1, tzinfo=UTC)
    ours = 55
    decided = _Episode(1, now, [_Agent(ours, 1.0, "us", 1), _Agent(99, -1.0, "them")])
    pending = _Episode(2, now, [_Agent(ours, None, "us", 1), _Agent(99, None, "them")])
    half = _Episode(3, now, [_Agent(ours, 1.0, "us", 1), _Agent(99, None, "them")])

    outcomes = to_outcomes(ours, [decided, pending, half])

    assert [outcome.episode_id for outcome in outcomes] == [1]
    summary = summarize(ours, outcomes)
    assert summary.total == 1, "pending episodes must not enter the denominator"
    assert summary.draws == 0, "a pending episode is not a draw"
    assert summary.win_rate == 1.0


def test_scout_reconstructs_only_the_scouted_teams_deck(monkeypatch, tmp_path) -> None:
    """
    ``scout_team`` attributes only the scouted team's own cards to that team.
    """
    import json
    from dataclasses import dataclass

    from cg.api import LogType
    from submission_analysis import scout

    @dataclass
    class _Agent:
        submission_id: int
        index: int

    @dataclass
    class _Episode:
        id: int
        agents: list

    @dataclass
    class _Submission:
        id: int
        public_score: str

    @dataclass
    class _Row:
        team_id: int
        team_name: str
        score: str

    their_card, opponent_card = 4242, 9999
    # Their side is index 1, so a naive both-sides union would pick up 9999.
    episode = _Episode(id=7, agents=[_Agent(999, 0), _Agent(55, 1)])

    def _log(player_index: int, card_id: int) -> dict:
        return {
            "type": int(LogType.PLAY),
            "playerIndex": player_index,
            "cardId": card_id,
        }

    # Both sides carry the same log batch, as a real replay does; the parser
    # selects by `playerIndex`, so side 0 yields the opponent's card and side 1
    # theirs. A fixture with logs on only one side would make this test vacuous.
    batch = [_log(0, opponent_card), _log(1, their_card)]
    step_entry = {"status": "DONE", "observation": {"logs": batch, "current": None}}
    replay = {"steps": [[dict(step_entry), dict(step_entry)]]}

    class _Api:
        """Minimal stand-in for the Kaggle client `scout_team` builds."""

        def __init__(self) -> None:
            self.submission = _Submission(id=55, public_score="1500")
            self.episode = episode
            self.replay = replay

        def authenticate(self) -> None:
            """No credentials needed for the fake."""

        def competition_team_submissions(self, team_id):  # noqa: ARG002
            return [self.submission]

        def competition_list_episodes(self, submission_id):  # noqa: ARG002
            return [self.episode]

        def competition_episode_replay(self, episode_id, path, quiet):  # noqa: ARG002
            Path(path).joinpath(f"episode-{episode_id}-replay.json").write_text(
                json.dumps(self.replay), encoding="utf-8"
            )

    monkeypatch.setattr(scout, "KaggleApi", _Api)

    team = scout.scout_team(
        _Row(team_id=1, team_name="them", score="1500"),
        episodes_per_team=1,
        replay_dir=tmp_path,
    )

    assert team is not None
    assert their_card in team.cards
    assert opponent_card not in team.cards, "opponent cards must not enter the deck"
    assert team.episodes_sampled == 1


def test_evolution_stats_group_reprints_of_the_same_card_by_name() -> None:
    """
    An evolution line split across two printings is counted as one line.
    """
    from cg.api import CardData, CardType
    from submission_analysis.deck_report import build_report
    from submission_analysis.loading import CardIndex
    from submission_analysis.parser import ParsedEpisode

    def _card(card_id: int, name: str, evolves_from: str | None = None) -> CardData:
        return CardData(
            cardId=card_id,
            name=name,
            cardType=CardType.POKEMON,
            retreatCost=1,
            hp=100,
            weakness=None,
            resistance=None,
            energyType=0,
            basic=evolves_from is None,
            stage1=evolves_from is not None,
            stage2=False,
            ex=False,
            megaEx=False,
            tera=False,
            aceSpec=False,
            evolvesFrom=evolves_from,
            skills=[],
            attacks=[],
        )

    # Two printings of the same Basic (100, 101) and one evolution off it.
    index = CardIndex(
        cards={
            100: _card(100, "Basic Mon"),
            101: _card(101, "Basic Mon"),
            200: _card(200, "Evolved Mon", evolves_from="Basic Mon"),
        },
        attacks={},
    )

    def _episode(episode_id: int, played: set[int], evolved: set[int]) -> ParsedEpisode:
        return ParsedEpisode(
            episode_id=episode_id,
            result="win",
            opponent_team="them",
            cards_played=played,
            evolutions_made=evolved,
        )

    episodes = [
        _episode(1, {100}, {200}),  # printing 100 played, evolved
        _episode(2, {101}, set()),  # printing 101 played, not evolved
    ]

    report = build_report(episodes, decklist=[100, 101, 200], card_index=index)

    (stat,) = report.evolution_stats
    assert stat.games_pre_evo_played == 2, "both printings count as the pre-evolution"
    assert stat.games_evolved == 1
    assert stat.conversion_rate == 0.5
