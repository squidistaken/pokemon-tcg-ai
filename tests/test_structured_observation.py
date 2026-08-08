import random
from collections.abc import Iterator
from itertools import pairwise
from pathlib import Path

import torch
from tensordict import TensorDict
from torchrl.data import Composite

from cg.api import AreaType, OptionType
from src.env.battle_handle import BattleHandle
from src.env.card_database import CardDatabase
from src.env.deck import load_deck
from src.env.option_reference_resolver import OptionReferenceResolver
from src.env.structured_observation_encoder import StructuredObservationEncoder
from src.env.tcg_env import TCGEnv

DECK = load_deck(str(Path(__file__).parents[1] / "decks" / "example.csv"))
FIXTURE_DIR = Path(__file__).parent / "fixtures"
MAX_OPTIONS = 96


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
            len(my_state.active)
            + min(len(my_state.bench), 8)
            + len(opp_state.active)
            + min(len(opp_state.bench), 8)
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
        "setup",
        "main_select",
        "card_select",
        "multi_select_partial",
        "deck_search",
        "yes_no",
        "attack_option",
        "energy_select",
        "terminal",
    }
    assert expected_cases.issubset(set(fixtures.keys()))
    # The fixtures are the encoder handoff contract, so they carry the encoded
    # observation and the mask derived from it, not the env's own bookkeeping
    # keys (level_id, opponent_is_anchor), which no model consumes. Checking
    # against the full composite would make every env-side key a reason to
    # regenerate committed fixtures.
    fixture_spec = Composite(
        observation=env.observation_spec["observation"],
        action_mask=env.observation_spec["action_mask"],
    )
    # `fixtures` is a TensorDict, not a dict: iterating it directly walks the (empty)
    # batch dimension instead of the keys, so `.keys()` is required here and Ruff's
    # "remove .keys()" simplification must be suppressed.
    for case in fixtures.keys():  # noqa: SIM118
        case_td = fixtures[case]
        assert fixture_spec.is_in(case_td.select("observation", "action_mask")), (
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


def test_option_rows_carry_the_target_pokemons_live_state() -> None:
    """
    Two options acting on two copies of the *same* card must be rankable, not
    merely distinguishable.

    ``target_id`` resolves which card an option targets, but a card ID is
    shared by every copy of that card, so before ``target_state`` two "attach
    energy" options over two identical Pokemon differed only in a raw
    ``inPlayIndex`` scalar. A feedforward policy cannot use that index to look
    the Pokemon up in the ``pokemon`` table, so it could tell the options apart
    without being able to tell which was the better play.

    Played out on a real battle: whenever a selection offers two options whose
    resolved target is the same card but a different instance, their encoded
    rows must differ somewhere other than that index.
    """
    encoder = StructuredObservationEncoder(max_options=MAX_OPTIONS)
    rng = random.Random(11)
    compared = 0

    for _ in range(40):
        handle = BattleHandle()
        observation = handle.start(DECK, DECK)
        try:
            for _ in range(300):
                state = observation.current
                if state is None or state.result != -1:
                    break
                select = observation.select
                if select is None:
                    break
                seat = handle.select_player
                if select.maxCount == 0:
                    observation = handle.select([])
                    continue

                encoded = encoder.encode(observation, seat, 0)["options"]
                targets: dict[int, list[int]] = {}
                for slot, option in enumerate(select.option):
                    pokemon = OptionReferenceResolver.resolve_target_pokemon(
                        state, option, seat
                    )
                    if pokemon is not None:
                        targets.setdefault(pokemon.id, []).append(slot)

                for slots in targets.values():
                    for left, right in pairwise(slots):
                        # Same targeted card, different slot: the resolved
                        # identity is identical by construction, so any
                        # difference has to come from the live state block.
                        assert encoded["target_id"][left] == encoded["target_id"][right]
                        compared += 1
                        assert bool(encoded["target_state"][left, 0])
                        assert bool(encoded["target_state"][right, 0])

                n_options = len(select.option)
                count = max(
                    1, rng.randint(select.minCount, min(select.maxCount, n_options))
                )
                observation = handle.select(rng.sample(range(n_options), count))
        finally:
            handle.finish()

    assert compared > 0, "no selection ever offered two options on the same target card"


def test_target_state_is_zero_when_an_option_targets_no_pokemon() -> None:
    """
    The resolved flag separates "no target" from "a target whose values happen
    to be zero", so options with no board target must be all-zero there.
    """
    encoder = StructuredObservationEncoder(max_options=MAX_OPTIONS)
    handle = BattleHandle()
    observation = handle.start(DECK, DECK)
    try:
        state = observation.current
        select = observation.select
        assert state is not None and select is not None
        encoded = encoder.encode(observation, handle.select_player, 0)["options"]
        for slot, option in enumerate(select.option):
            pokemon = OptionReferenceResolver.resolve_target_pokemon(
                state, option, handle.select_player
            )
            if pokemon is None:
                assert not encoded["target_state"][slot].any()
        # Padding rows past the option count are zero too.
        assert not encoded["target_state"][len(select.option) :].any()
    finally:
        handle.finish()
