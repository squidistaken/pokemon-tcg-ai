"""
Generate observation fixtures for model development.

Plays random-policy games in :class:`TCGEnv` and captures one structured
observation TensorDict per named case (setup, main-phase selection, ongoing
multi-select, deck search, ...), so the model/agent code can be developed and
unit-tested against real data without running the engine. Also exports the
static :class:`CardDatabase` lookup tables.

Outputs (committed to the repository, see ``docs/observation_spec.md``)::

    tests/fixtures/observations.pt   TensorDict: case name -> {observation, action_mask}
    tests/fixtures/card_tables.pt    TensorDict: CardDatabase lookup tables

Load with ``torch.load(path, weights_only=False)``.

Run from the repository root::

    uv run python scripts/generate_obs_fixtures.py

The script inspects the environment's pending selection (via its public
``pending_select``/``already_chosen_option_count`` accessors) to classify
observations; it is a dev tool, not part of training.
"""

import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import torch
from tensordict import TensorDict

from cg.api import OptionType, SelectType
from src.env.decks.deck import load_deck
from src.env.observation.card_database import CardDatabase
from src.env.tcg_env import TCGEnv

FIXTURE_DIR = REPO_ROOT / "tests" / "fixtures"
MAX_STEPS_PER_EPISODE = 5000  # Don't like this but this is a slop script anyway
"""Safety cap on steps per random-policy episode, to bail out of a stuck game."""
CASE_NAMES = [
    "setup",
    "main_select",
    "card_select",
    "multi_select_partial",
    "deck_search",
    "yes_no",
    "attack_option",
    "energy_select",
    "terminal",
]


def classify(env: TCGEnv, is_reset: bool, is_terminal: bool) -> list[str]:
    """
    Name the fixture cases the environment's pending selection matches.

    :param env: Environment whose pending selection to classify.
    :param is_reset: Whether this observation came from a reset.
    :param is_terminal: Whether this observation is terminal.
    :return: List of matching case names (possibly empty).
    """
    if is_terminal:
        return ["terminal"]
    select = env.pending_select
    cases: list[str] = []
    if is_reset:
        cases.append("setup")
    if env.already_chosen_option_count >= 1:
        cases.append("multi_select_partial")
    if select.deck is not None:
        cases.append("deck_search")
    by_type = {
        SelectType.MAIN: "main_select",
        SelectType.CARD: "card_select",
        SelectType.YES_NO: "yes_no",
        SelectType.ENERGY: "energy_select",
    }
    if select.type in by_type:
        cases.append(by_type[select.type])
    if any(option.type == OptionType.ATTACK for option in select.option):
        cases.append("attack_option")
    return cases


def main() -> None:
    """
    Play random games until every fixture case is captured, then save.
    """
    deck = load_deck(str(REPO_ROOT / "decks" / "example.csv"))
    env = TCGEnv(deck, deck, seed=0)
    rng = random.Random(0)
    captured: dict[str, TensorDict] = {}
    episodes = 0
    while len(captured) < len(CASE_NAMES) and episodes < 500:
        obs_td = env.reset()
        episodes += 1
        is_reset = True
        for _ in range(MAX_STEPS_PER_EPISODE):
            done = bool(obs_td.get("done", torch.zeros(1, dtype=torch.bool)).any())
            for case in classify(env, is_reset, done):
                if case not in captured:
                    captured[case] = obs_td.select("observation", "action_mask").clone()
                    print(f"captured '{case}' (episode {episodes})")
            if done:
                break
            is_reset = False
            legal = obs_td["action_mask"].nonzero().reshape(-1).tolist()
            obs_td["action"] = torch.tensor(rng.choice(legal), dtype=torch.int64)
            obs_td = env.step(obs_td)["next"]
    env.close()

    missing = [case for case in CASE_NAMES if case not in captured]
    if missing:
        print(f"WARNING: no state matched: {missing} (after {episodes} episodes)")

    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    observations = TensorDict(captured, batch_size=torch.Size(()))
    torch.save(observations, FIXTURE_DIR / "observations.pt")
    torch.save(CardDatabase().as_tensordict(), FIXTURE_DIR / "card_tables.pt")
    print(f"wrote {FIXTURE_DIR / 'observations.pt'} ({len(captured)} cases)")
    print(f"wrote {FIXTURE_DIR / 'card_tables.pt'}")


if __name__ == "__main__":
    main()
