import random
from pathlib import Path

import torch

from cg.api import AreaType, OptionType
from src.env.card_database import CardDatabase
from src.env.deck import load_deck
from src.env.tcg_env import TCGEnv

DECK = load_deck(str(Path(__file__).parents[1] / "decks" / "example.csv"))
FIXTURE_DIR = Path(__file__).parent / "fixtures"


def test_zone_tables_consistent_with_state() -> None:
    """
    Zone masks and count features agree with the raw engine state at every
    visited selection.
    """
    env = TCGEnv(DECK, DECK, seed=4)
    rng = random.Random(4)
    obs_td = env.reset()
    for _ in range(300):
        state = env._pending.current
        obs = obs_td["observation"]
        my_state = state.players[env._agent_seat]
        opp_state = state.players[1 - env._agent_seat]
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
        legal = obs_td["action_mask"].nonzero().reshape(-1).tolist()
        obs_td["action"] = torch.tensor(rng.choice(legal), dtype=torch.int64)
        obs_td = env.step(obs_td)["next"]
        if bool(obs_td["done"].any()):
            obs_td = env.reset()
    env.close()


def test_option_table_aligned_with_action_mask() -> None:
    """
    Option rows are populated exactly for the offered options, and card
    references into the agent's hand resolve to the correct card IDs.
    """
    env = TCGEnv(DECK, DECK, seed=5)
    rng = random.Random(5)
    obs_td = env.reset()
    checked_hand_options = 0
    for _ in range(300):
        select = env._pending.select
        obs = obs_td["observation"]
        option_types = obs["options", "cats"][:, 0]
        n_options = min(len(select.option), 96)
        assert (option_types[:n_options] != 0).all()
        assert (option_types[n_options:] == 0).all()
        my_hand = env._pending.current.players[env._agent_seat].hand
        for slot, option in enumerate(select.option[:96]):
            references_own_hand = (
                option.type in (OptionType.CARD, OptionType.PLAY)
                and (option.area is None or option.area == AreaType.HAND)
                and (option.playerIndex is None or option.playerIndex == env._agent_seat)
                and option.index is not None
            )
            if references_own_hand and my_hand is not None and option.index < len(my_hand):
                assert int(obs["options", "card_id"][slot]) == my_hand[option.index].id
                checked_hand_options += 1
            if option.type == OptionType.ATTACK:
                assert int(obs["options", "attack_id"][slot]) == option.attackId
        legal = obs_td["action_mask"].nonzero().reshape(-1).tolist()
        obs_td["action"] = torch.tensor(rng.choice(legal), dtype=torch.int64)
        obs_td = env.step(obs_td)["next"]
        if bool(obs_td["done"].any()):
            obs_td = env.reset()
    env.close()
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
    for case in fixtures.keys():
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
