from pathlib import Path

import pytest
import torch
from omegaconf import DictConfig, OmegaConf
from torchrl.data import Binary, Categorical, Composite

from src.env.structured_observation_encoder import StructuredObservationEncoder
from src.training.ppo_trainer import PPOTrainer

DECK_PATH = str(Path(__file__).parents[1] / "decks" / "example.csv")
MAX_OPTIONS = 96
N_ACTIONS = MAX_OPTIONS + 1


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

    def count_optimizer_steps_for_update(self, data) -> int:
        """
        Run one update, counting how many optimizer steps were applied.

        Used to observe ``target_kl`` early stopping (fewer steps than
        ``num_epochs * minibatches``) without reaching into trainer internals
        from the test body.

        :param data: Collected batch from a TorchRL collector.
        :return: Number of ``optimizer.step()`` calls during the update.
        """
        calls = 0
        original_step = self._optim.step

        def counting_step(*args, **kwargs):
            nonlocal calls
            calls += 1
            return original_step(*args, **kwargs)

        self._optim.step = counting_step
        try:
            self._update(data)
        finally:
            self._optim.step = original_step
        return calls

    @property
    def current_lr_for_test(self) -> float:
        """
        Current optimizer learning rate (for annealing assertions).

        :return: LR of the first optimizer parameter group.
        """
        return float(self._optim.param_groups[0]["lr"])

    @property
    def current_entropy_coeff_for_test(self) -> float:
        """
        Current loss-module entropy coefficient (for annealing assertions).

        :return: The entropy-bonus weight as a float.
        """
        return float(self._loss.entropy_coeff)

    def prepare_restart_for_test(self, restart_index: int) -> None:
        """
        Run the pre-restart hook the trainer calls before rebuilding a pool.

        :param restart_index: 1-based index of the restart being simulated.
        """
        self._prepare_restart(restart_index)


@pytest.fixture
def structured_obs_spec() -> Composite:
    """
    Observation spec matching the (default) structured encoder plus the
    action mask.

    :return: Composite spec with the nested structured observation groups and
        the ``action_mask``, matching what :class:`TCGEnv` emits with the
        default structured encoder.
    """
    return Composite(
        observation=StructuredObservationEncoder(max_options=MAX_OPTIONS).spec(),
        action_mask=Binary(n=N_ACTIONS, dtype=torch.bool),
    )


@pytest.fixture
def structured_model_cfg() -> DictConfig:
    """
    Minimal ``model`` config selecting the MLP backbone over the structured
    observation's top-level groups, with tiny layer widths so unit tests
    stay fast.

    :return: OmegaConf config with a ``model`` section.
    """
    return OmegaConf.create(
        {
            "model": {
                "embed_dim": 32,
                "backbone": {
                    "_target_": "src.models.mlp.MLPBackbone",
                    "num_cells": [32],
                    "activation": "tanh",
                    "in_keys": [
                        ["observation", "globals"],
                        ["observation", "select_cats"],
                        ["observation", "context_card_ids"],
                        ["observation", "stadium_id"],
                        ["observation", "options"],
                        ["observation", "pokemon"],
                        ["observation", "my"],
                        ["observation", "opp"],
                        ["observation", "select_deck"],
                        ["observation", "looking"],
                    ],
                },
                "head": {"_target_": "src.models.heads.LinearPolicyHead"},
                "value_head": {"num_cells": [32]},
            }
        }
    )


@pytest.fixture
def transformer_model_cfg() -> DictConfig:
    """
    Minimal ``model`` config selecting the transformer backbone over the
    structured observation's top-level groups, with a tiny embed dim so unit
    tests stay fast.

    :return: OmegaConf config with a ``model`` section.
    """
    return OmegaConf.create(
        {
            "model": {
                "embed_dim": 32,
                "backbone": {
                    "_target_": "src.models.transformer.TransformerBackbone",
                    "num_heads": 4,
                    "num_layers": 1,
                    "ff_dim": 32,
                    "activation": "gelu",
                    "in_keys": [
                        ["observation", "globals"],
                        ["observation", "select_cats"],
                        ["observation", "context_card_ids"],
                        ["observation", "stadium_id"],
                        ["observation", "options"],
                        ["observation", "pokemon"],
                        ["observation", "my"],
                        ["observation", "opp"],
                        ["observation", "select_deck"],
                        ["observation", "looking"],
                    ],
                },
                "head": {"_target_": "src.models.heads.LinearPolicyHead"},
                "value_head": {"num_cells": [32]},
            }
        }
    )


@pytest.fixture
def action_spec() -> Categorical:
    """
    Discrete action spec over the option slots plus the stop action.

    :return: Categorical spec of size ``N_ACTIONS``.
    """
    return Categorical(N_ACTIONS, dtype=torch.int64)


def structured_env_cfg(num_workers: int = 2) -> DictConfig:
    """
    Build a config for structured-observation environment factories.

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
                "encoder": "structured",
            },
        }
    )
