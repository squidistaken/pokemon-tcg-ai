"""Shared checkpoint loader for the head-to-head tool."""

import torch
from omegaconf import OmegaConf
from torchrl.data import Binary, Categorical, Composite, Unbounded

from src.env.observation.structured_observation_encoder import (
    StructuredObservationEncoder,
)
from src.policies.ppo_actor import build_actor_critic

MAX_OPTIONS = 128


def build_network(path: str):
    """
    Rebuild a checkpoint's actor-critic and its encoder.

    :param path: Snapshot path.
    :return: The network in eval mode and the matching encoder.
    """
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = OmegaConf.create(checkpoint["config"])
    encoder = StructuredObservationEncoder(max_options=MAX_OPTIONS)
    n_actions = MAX_OPTIONS + 1
    obs_spec = Composite(
        observation=encoder.spec(),
        action_mask=Binary(n=n_actions, dtype=torch.bool),
        level_id=Unbounded(shape=(1,), dtype=torch.int64),
        opponent_is_anchor=Binary(1, dtype=torch.bool),
    )
    network = build_actor_critic(
        config, obs_spec, Categorical(n_actions, dtype=torch.int64)
    )
    network.load_state_dict(checkpoint["state_dict"], strict=True)
    return network.eval(), encoder
