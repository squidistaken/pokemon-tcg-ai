from pathlib import Path

import torch
from omegaconf import OmegaConf
from tensordict import TensorDict
from torchrl.data import Categorical, Composite

from cg.api import Observation
from src.env.observation_encoder import ObservationEncoder
from src.models.actor_critic import ActorCritic
from src.policies.ppo_actor import build_actor_critic
from src.training.env_factory import make_encoder


def build_inference_specs(
        max_options: int,
        encoder_name: str = "structured",
) -> tuple[Composite, ObservationEncoder, Categorical]:
    """
    Build the observation/action specs :func:`~src.policies.ppo_actor.
    build_actor_critic` needs, without instantiating a live environment.

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


class SamplingPolicyAgent:
    """
    Our Kaggle submission agent: wraps a trained actor-critic and samples
    actions from its learned distribution.

    Answers a whole engine selection in one call: encodes the observation
    from the acting seat, scores the option slots with the policy head, and
    samples picks within ``[minCount, maxCount]`` using the learned **stop**
    logit to decide how many to take.
    """

    def __init__(
            self,
            actor_critic: ActorCritic,
            encoder: ObservationEncoder,
            device: torch.device | str = "cpu",
            generator: torch.Generator | None = None,
    ) -> None:
        """
        :param actor_critic: Trained actor-critic to act with; put into eval
            mode and never updated here.
        :param encoder: Observation encoder matching the one used in training.
        :param device: Device for inference.
        :param generator: Optional RNG for reproducible sampling (e.g. in
            tests).
        """
        self._device = torch.device(device)
        self._actor_critic = actor_critic.to(self._device).eval()
        self._encoder = encoder
        self._generator = generator

    @torch.inference_mode()
    def __call__(self, observation: Observation) -> list[int]:
        """
        Sample a legal selection for the acting seat.

        :param observation: Current engine observation with a non-None select;
            ``current.yourIndex`` gives the acting seat used for encoding.
        :return: Option indices to submit, between ``minCount`` and
            ``maxCount`` entries with no duplicates.
        """
        select = observation.select
        state = observation.current
        if select is None or state is None:
            raise ValueError("SamplingPolicyAgent requires an observation with current state and select.")
        seat = state.yourIndex
        # Match TCGEnv, which nests the encoder output under "observation".
        encoded = TensorDict(
            {"observation": self._encoder.encode(observation, seat, 0)},
            batch_size=torch.Size(()),
        )
        # Guarded: on CPU .to() would still walk and copy every leaf for
        # nothing, on every move.
        if self._device.type != "cpu":
            encoded = encoded.to(self._device)
        logits = self._actor_critic.policy_logits(encoded)
        return self.sample_select(
            logits,
            n_options=len(select.option),
            min_count=select.minCount,
            max_count=select.maxCount,
            generator=self._generator,
        )

    @staticmethod
    def sample_select(
            logits: torch.Tensor,
            n_options: int,
            min_count: int,
            max_count: int,
            generator: torch.Generator | None = None,
    ) -> list[int]:
        """
        Sample option indices from action logits, one pick at a time.

        Options ``0..n_options-1`` map to logits ``0..n_options-1``; the final
        logit is the synthetic **stop**. At each step, samples from the
        softmax over the not-yet-picked options (plus stop, once ``minCount``
        picks have been made), stopping when stop is drawn or ``maxCount`` is
        reached.

        :param logits: Action logits of shape ``(n_actions,)`` where
            ``n_actions = max_options + 1``.
        :param n_options: Number of real options offered by the selection.
        :param min_count: Minimum number of options to pick.
        :param max_count: Maximum number of options to pick.
        :param generator: Optional RNG for reproducible sampling.
        :return: Chosen option indices (a subset of ``range(n_options)``).
        """
        capacity = logits.shape[-1] - 1
        n_options = min(n_options, capacity)
        max_count = min(max_count, n_options)
        stop_index = capacity
        remaining = list(range(n_options))
        picks: list[int] = []
        while len(picks) < max_count:
            candidates = remaining + ([stop_index] if len(picks) >= min_count else [])
            probs = torch.softmax(logits[candidates], dim=-1)
            choice = int(torch.multinomial(probs, 1, generator=generator).item())
            chosen = candidates[choice]
            if chosen == stop_index:
                break
            picks.append(chosen)
            remaining.remove(chosen)
        return picks


def load_inference_agent(
        checkpoint_path: str | Path,
        model_config_path: str | Path,
        device: torch.device | str = "cpu",
        generator: torch.Generator | None = None,
) -> SamplingPolicyAgent:
    """
    Rebuild the submission agent from a self-contained checkpoint + config
    pair.

    Reads ``max_options``/``encoder`` from ``model_config_path`` and derives
    the specs via :func:`build_inference_specs`, so it needs nothing but the
    two files on disk.

    :param checkpoint_path: Path to a :func:`~src.policies.
        greedy_policy_opponent.save_actor_critic` state_dict.
    :param model_config_path: Path to the sidecar YAML written alongside the
        checkpoint, holding the resolved ``model`` config plus
        ``max_options``/``encoder``.
    :param device: Device for inference.
    :param generator: Optional RNG for reproducible sampling.
    :return: The checkpoint's agent, ready to act.
    """
    model_config = OmegaConf.load(model_config_path)
    obs_spec, encoder, action_spec = build_inference_specs(
        max_options=int(model_config.max_options),
        encoder_name=str(model_config.get("encoder", "structured")),
    )
    actor_critic = build_actor_critic(model_config, obs_spec, action_spec)
    state_dict = torch.load(Path(checkpoint_path), map_location=device, weights_only=True)
    actor_critic.load_state_dict(state_dict)
    return SamplingPolicyAgent(actor_critic, encoder, device=device, generator=generator)
