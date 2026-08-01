"""Kaggle entry point for a self-contained, Torch-only PPO checkpoint."""

import json
import sys
from pathlib import Path

import torch

_SOURCE_FILE = globals().get("__file__")
if isinstance(_SOURCE_FILE, str):
    _BUNDLE_DIR = Path(_SOURCE_FILE).resolve().parent
elif sys.path:
    # Kaggle's get_last_callable() does not define __file__; immediately before
    # exec(), it appends the submitted main.py directory to sys.path.
    _BUNDLE_DIR = Path(sys.path[-1]).resolve()
else:
    raise RuntimeError("Cannot resolve the Kaggle submission bundle directory")
_BUNDLE_PATH = str(_BUNDLE_DIR)
_IS_PACKAGE = bool(globals().get("__package__"))
if not _IS_PACKAGE and (not sys.path or sys.path[0] != _BUNDLE_PATH):
    sys.path.insert(0, _BUNDLE_PATH)

if _IS_PACKAGE:
    from .cg_api import Observation, to_observation_class
    from .runtime import Policy
else:
    from cg_api import Observation, to_observation_class
    from runtime import Policy

# Kaggle calls agent() once per engine selection in the same interpreter. Keep
# the loaded weights and rebuilt network for that process instead of repeating
# both relatively expensive operations on every decision.
_CACHED_POLICY: Policy | None = None


def read_deck_csv() -> list[int]:
    """Read the 60-card deck bundled beside this file."""
    deck_path = _BUNDLE_DIR / "deck.csv"
    cards = [
        int(line)
        for line in deck_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(cards) != 60:
        raise ValueError(
            f"deck.csv must contain exactly 60 card IDs, found {len(cards)}"
        )
    return cards


def _load_policy() -> Policy:
    """Lazily rebuild and process-cache the bundled trained policy."""
    global _CACHED_POLICY
    if _CACHED_POLICY is not None:
        return _CACHED_POLICY

    config = json.loads((_BUNDLE_DIR / "model_config.json").read_text(encoding="utf-8"))
    try:
        payload = torch.load(
            _BUNDLE_DIR / "model.pt", map_location="cpu", weights_only=True
        )
    except TypeError:  # torch<2.0 does not support weights_only.
        payload = torch.load(_BUNDLE_DIR / "model.pt", map_location="cpu")
    _CACHED_POLICY = Policy(payload, config)
    return _CACHED_POLICY


def agent(obs_dict: dict) -> list[int]:
    """Return the deck during setup, otherwise a configured legal selection."""
    observation: Observation = to_observation_class(obs_dict)
    if observation.select is None:
        return read_deck_csv()
    return _load_policy()(observation)
