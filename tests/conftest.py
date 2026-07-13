from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf
from torchrl.data import Binary, Categorical, Composite, Unbounded

from src.env.flat_observation_encoder import FlatObservationEncoder
from src.training.ppo_trainer import PPOTrainer

DECK_PATH = str(Path(__file__).parents[1] / "decks" / "example.csv")
MAX_OPTIONS = 96
N_ACTIONS = MAX_OPTIONS + 1
FLAT_DIM = FlatObservationEncoder().dim


class PPOTrainerForTests(PPOTrainer):
    """
    Test-only PPOTrainer exposing selected internals for white-box tests.
    """

    @property
    def policy_for_test(self):
        """
        Collection policy used by the trainer.

        :return: The policy module passed to the base trainer.
        """
        return self._policy

    def update_for_test(self, data):
        """
        Run one algorithm update on a collected batch for testing.

        :param data: Collected batch from a TorchRL collector.
        :return: Loss values returned by the trainer update.
        """
        return self._update(data)


@pytest.fixture
def model_cfg() -> OmegaConf:
    """
    Minimal ``model`` config selecting the MLP backbone + linear head.

    Uses tiny layer widths so unit tests stay fast.

    :return: OmegaConf config with a ``model`` section.
    """
    return OmegaConf.create(
        {
            "model": {
                "embed_dim": 32,
                "backbone": {
                    "_target_": "src.models.backbone.MLPBackbone",
                    "num_cells": [32],
                    "activation": "tanh",
                    "in_keys": [["observation", "observation"]],
                },
                "head": {"_target_": "src.models.heads.LinearPolicyHead"},
                "value_head": {"num_cells": [32]},
            }
        }
    )


@pytest.fixture
def flat_obs_spec() -> Composite:
    """
    Observation spec matching the flat encoder plus the action mask.

    :return: Composite spec with the nested flat ``observation`` and the
        ``action_mask``, matching what :class:`TCGEnv` emits with the flat
        encoder (the encoder output is nested under ``observation``).
    """
    return Composite(
        observation=Composite(observation=Unbounded(shape=(FLAT_DIM,), dtype=torch.float32)),
        action_mask=Binary(n=N_ACTIONS, dtype=torch.bool),
    )


@pytest.fixture
def action_spec() -> Categorical:
    """
    Discrete action spec over the option slots plus the stop action.

    :return: Categorical spec of size ``N_ACTIONS``.
    """
    return Categorical(N_ACTIONS, dtype=torch.int64)


def flat_env_cfg(num_workers: int = 2) -> OmegaConf:
    """
    Build a config for flat-observation environment factories.

    :param num_workers: Number of environment workers.
    :return: OmegaConf config with ``seed`` and an ``env`` section.
    """
    return OmegaConf.create(
        {
            "seed": 0,
            "env": {
                "deck0": DECK_PATH,
                "deck1": DECK_PATH,
                "max_options": MAX_OPTIONS,
                "num_workers": num_workers,
                "parallel": False,
                "encoder": "flat",
            },
        }
    )
