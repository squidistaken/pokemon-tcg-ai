from pathlib import Path

import torch
from omegaconf import OmegaConf
from torchrl.data import Categorical, Composite

from src.env.observation_encoder import ObservationEncoder
from src.policies.greedy_policy_opponent import GreedyPolicyOpponent
from src.policies.ppo_actor import build_actor_critic
from src.training.env_factory import make_encoder


def build_inference_specs(
        max_options: int,
        encoder_name: str = "structured",
) -> tuple[Composite, ObservationEncoder, Categorical]:
    """
    Build the observation/action specs :func:`~src.policies.ppo_actor.
    build_actor_critic` needs, without instantiating a live environment.

    :class:`~src.env.tcg_env.TCGEnv` derives both purely from ``max_options``
    and the encoder (``Composite(observation=encoder.spec(), ...)`` and
    ``Categorical(max_options + 1)``); reproducing that here lets inference
    rebuild the architecture without the engine, decks, or a battle handle.

    :param max_options: Padded size of the option space (stop action
        excluded) the checkpoint was trained with.
    :param encoder_name: Observation encoder name (see
        :func:`~src.training.env_factory.make_encoder`).
    :return: The observation composite spec, the encoder instance used to
        build it, and the action spec.
    """
    encoder = make_encoder(encoder_name, max_options)
    obs_spec = Composite(observation=encoder.spec())
    action_spec = Categorical(max_options + 1, dtype=torch.int64)
    return obs_spec, encoder, action_spec


def load_inference_opponent(
        checkpoint_path: str | Path,
        model_config_path: str | Path,
        device: torch.device | str = "cpu",
) -> GreedyPolicyOpponent:
    """
    Rebuild a greedy opponent from a self-contained checkpoint + config pair.

    Unlike :func:`~src.policies.greedy_policy_opponent.load_greedy_opponent`
    (which takes specs from a live training environment), this reads
    ``max_options``/``encoder`` from ``model_config_path`` itself and derives
    the specs via :func:`build_inference_specs`, so it needs nothing but the
    two files on disk. This is the loader used by ``main.py`` and by
    ``scripts/export_submission_checkpoint.py``.

    :param checkpoint_path: Path to a :func:`~src.policies.
        greedy_policy_opponent.save_actor_critic` state_dict.
    :param model_config_path: Path to the sidecar YAML written alongside the
        checkpoint, holding the resolved ``model`` config plus
        ``max_options``/``encoder``.
    :param device: Device for inference.
    :return: A greedy opponent playing the checkpoint's policy.
    """
    model_config = OmegaConf.load(model_config_path)
    obs_spec, encoder, action_spec = build_inference_specs(
        max_options=int(model_config.max_options),
        encoder_name=str(model_config.get("encoder", "structured")),
    )
    actor_critic = build_actor_critic(model_config, obs_spec, action_spec)
    state_dict = torch.load(Path(checkpoint_path), map_location=device, weights_only=True)
    actor_critic.load_state_dict(state_dict)
    return GreedyPolicyOpponent(actor_critic, encoder, device=device)
