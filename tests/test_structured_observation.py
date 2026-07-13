import random
from collections.abc import Iterator
from pathlib import Path

import torch
from tensordict import TensorDict

from cg.api import AreaType, OptionType
from src.env.card_database import CardDatabase
from src.env.deck import load_deck
from src.env.tcg_env import TCGEnv

DECK = load_deck(str(Path(__file__).parents[1] / "decks" / "example.csv"))
FIXTURE_DIR = Path(__file__).parent / "fixtures"


def _random_rollout(seed: int, steps: int = 300) -> Iterator[tuple[TCGEnv, TensorDict]]:
    """
    Drive an environment through random legal actions, yielding the
    environment and current observation before each action is taken.

    Episodes are reset transparently on termination so the caller always
    sees `steps` consecutive selections. The environment is closed once
    the rollout is exhausted.

    :param seed: Seed for both the environment and the action RNG.
    :param steps: Number of random actions to take.
    :return: Iterator over `(env, obs_td)` pairs, one per selection.
    """
    env = TCGEnv(DECK, DECK, seed=seed)
    rng = random.Random(seed)
    obs_td = env.reset()
    try:
        for _ in range(steps):
            yield env, obs_td
            legal = obs_td["action_mask"].nonzero().reshape(-1).tolist()
            obs_td["action"] = torch.tensor(rng.choice(legal), dtype=torch.int64)
            obs_td = env.step(obs_td)["next"]
            if bool(obs_td["done"].any()):
                obs_td = env.reset()
    finally:
        env.close()


def test_zone_tables_consistent_with_state() -> None:
    """
    Zone masks and count features agree with the raw engine state at every
    visited selection.
    """
    for env, obs_td in _random_rollout(seed=4):
        state = env.current_state
        obs = obs_td["observation"]
        my_state = state.players[env.agent_seat]
        opp_state = state.players[1 - env.agent_seat]
        assert int(obs["my", "hand_mask"].sum()) == min(my_state.handCount, 30)
        assert int(obs["my", "discard_mask"].sum()) == len(my_state.discard)
        assert int(obs["opp", "discard_mask"].sum()) == len(opp_state.discard)
        assert int(obs["my", "prize_mask"].sum()) == len(my_state.prize)
        assert int(obs["opp", "prize_mask"].sum()) == len(opp_state.prize)
        expected_pokemon = (
            len(my_state.active) + min(len(my_state.bench), 8)
            + len(opp_state.active) + min(len(opp_state.bench), 8)
        )
        assert int(obs["pokemon", "mask"].sum()) == expected_pokemon


def test_option_table_aligned_with_action_mask() -> None:
    """
    Option rows are populated exactly for the offered options, and card
    references into the agent's hand resolve to the correct card IDs.
    """
    checked_hand_options = 0
    for env, obs_td in _random_rollout(seed=5):
        select = env.pending_select
        obs = obs_td["observation"]
        option_types = obs["options", "cats"][:, 0]
        n_options = min(len(select.option), 96)
        assert (option_types[:n_options] != 0).all()
        assert (option_types[n_options:] == 0).all()
        my_hand = env.current_state.players[env.agent_seat].hand
        for slot, option in enumerate(select.option[:96]):
            references_own_hand = (
                option.type in (OptionType.CARD, OptionType.PLAY)
                and (option.area is None or option.area == AreaType.HAND)
                and (option.playerIndex is None or option.playerIndex == env.agent_seat)
                and option.index is not None
            )
            if (
                references_own_hand
                and my_hand is not None
                and option.index is not None
                and option.index < len(my_hand)
            ):
                assert int(obs["options", "card_id"][slot]) == my_hand[option.index].id
                checked_hand_options += 1
            if option.type == OptionType.ATTACK:
                assert int(obs["options", "attack_id"][slot]) == option.attackId
    assert checked_hand_options > 0, "no hand-referencing options were exercised"


def test_fixture_observations_match_spec() -> None:
    """
    Every committed fixture case conforms to the environment's observation
    spec (regenerate via scripts/generate_obs_fixtures.py when it changes).
    """
    fixtures = torch.load(FIXTURE_DIR / "observations.pt", weights_only=False)
    env = TCGEnv(DECK, DECK)
    expected_cases = {
        "setup", "main_select", "card_select", "multi_select_partial",
        "deck_search", "yes_no", "attack_option", "energy_select", "terminal",
    }
    assert expected_cases.issubset(set(fixtures.keys()))
    # `fixtures` is a TensorDict, not a dict: iterating it directly walks the (empty)
    # batch dimension instead of the keys, so `.keys()` is required here and Ruff's
    # "remove .keys()" simplification must be suppressed.
    for case in fixtures.keys():  # noqa: SIM118
        case_td = fixtures[case]
        assert env.observation_spec.is_in(case_td.select("observation", "action_mask")), (
            f"fixture case '{case}' does not match the observation spec"
        )
    env.close()


def test_card_database_tables() -> None:
    """
    The static card tables index by raw ID, keep row 0 empty and cover every
    card in the example deck.
    """
    database = CardDatabase()
    assert (database.card_features[0] == 0).all()
    assert (database.card_cats[0] == 0).all()
    assert (database.attack_features[0] == 0).all()
    exists_column = database.card_features[:, -1]
    for card_id in DECK:
        assert exists_column[card_id] == 1.0
        assert database.card_name(card_id) != "<none>"
    fixture_tables = torch.load(FIXTURE_DIR / "card_tables.pt", weights_only=False)
    assert fixture_tables["card_features"].shape == database.card_features.shape
