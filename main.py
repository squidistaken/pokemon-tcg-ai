import os

from cg.api import Observation, to_observation_class
from src.policies.greedy_policy_opponent import GreedyPolicyOpponent
from src.policies.inference import load_inference_agent

KAGGLE_AGENT_DIR = "/kaggle_simulations/agent/"
CHECKPOINT_PATH = "checkpoint/model.pt"
MODEL_CONFIG_PATH = "checkpoint/model_config.yaml"
# Sample from the learned distribution rather than always taking the
# highest-scoring options: a deterministic policy is a fixed function of the
# observed state, which an opponent can learn and reliably counter, whereas
# sampling only ever exposes it to probabilities. Flip to True to switch back
# to greedy (deterministic) action selection. This only affects the Kaggle
# inference path here, not training/self-play, which has its own default.
DETERMINISTIC_INFERENCE = False

_agent: GreedyPolicyOpponent | None = None


def _resolve_path(file_path: str) -> str:
    """
    Resolve a submission-bundle-relative path, local run or Kaggle sandbox.

    :param file_path: Path relative to the submission bundle's root.
    :return: ``file_path`` unchanged if it exists in the current working
        directory, otherwise the same path under the Kaggle agent directory.
    """
    if not os.path.exists(file_path):
        file_path = KAGGLE_AGENT_DIR + file_path
    return file_path


def read_deck_csv() -> list[int]:
    """
    Read deck.csv.

    Returns:
        list[int]: A list of card IDs in the deck.
    """
    file_path = _resolve_path("deck.csv")
    with open(file_path, "r") as file:
        csv = file.read().split("\n")
    deck = []
    for i in range(60):
        deck.append(int(csv[i]))
    return deck


def _load_agent() -> GreedyPolicyOpponent:
    """
    Build (once) and return the checkpointed agent used for inference.

    Lazily constructed on first use and cached in ``_agent``: ``agent()``
    is called once per selection for the whole match, so rebuilding the
    network and reloading weights on every call would be pure overhead.

    Returns:
        GreedyPolicyOpponent: Our submission agent, wrapping the loaded
            checkpoint and sampling from its learned distribution.

    Raises:
        FileNotFoundError: If the checkpoint or its config is missing. A
            submission without a trained model is a packaging mistake, not a
            state to silently degrade from; run
            ``scripts/export_submission_checkpoint.py`` first.
    """
    global _agent
    if _agent is None:
        checkpoint_path = _resolve_path(CHECKPOINT_PATH)
        model_config_path = _resolve_path(MODEL_CONFIG_PATH)
        if not os.path.exists(checkpoint_path) or not os.path.exists(model_config_path):
            raise FileNotFoundError(
                f"Missing checkpoint ({checkpoint_path!r}) or model config ({model_config_path!r}). "
                "Run scripts/export_submission_checkpoint.py before submitting."
            )
        _agent = load_inference_agent(
            checkpoint_path, model_config_path, deterministic=DETERMINISTIC_INFERENCE
        )
    return _agent


def agent(obs_dict: dict) -> list[int]:
    """
    Implement Your Pokémon Trading Card Game Agent.

    Each element in the returned list must be >= 0 and < len(obs.select.option).
    The list length must be between obs.select.minCount and obs.select.maxCount (inclusive), with no duplicate elements.

    Returns:
        list[int]: A list of option index.
    """
    obs: Observation = to_observation_class(obs_dict)
    if obs.select == None:
        # In the initial selection, the obs.select is None, and it is necessary to return the deck.
        # The deck is a list of 60 card IDs.
        # The deck must comply with the Pokémon Trading Card Game rules.
        return read_deck_csv()

    return _load_agent()(obs)
