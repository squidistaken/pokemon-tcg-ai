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


class InferenceAgent:
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
        max_options: int,
        device: torch.device | str = "cpu",
    ) -> None:
        """
        :param actor_critic: Trained actor-critic to act with; put into eval
            mode and never updated here.
        :param encoder: Observation encoder matching the one used in training.
        :param max_options: Option capacity the actor was trained with, not
            including the synthetic stop action.
        :param device: Device for inference.
        """
        if max_options < 0:
            raise ValueError("max_options must be non-negative.")
        self._device = torch.device(device)
        self._actor_critic = actor_critic.to(self._device).eval()
        self._encoder = encoder
        self._max_options = max_options

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
            raise ValueError(
                "SamplingPolicyAgent requires an observation with current state and select."
            )
        n_options = len(select.option)
        min_count = select.minCount
        max_count = select.maxCount
        if not 0 <= min_count <= max_count <= n_options:
            raise ValueError(
                "Invalid selection bounds: expected "
                f"0 <= minCount ({min_count}) <= maxCount ({max_count}) "
                f"<= option count ({n_options})."
            )
        if n_options > self._max_options:
            raise ValueError(
                f"Selection offers {n_options} options but the checkpoint supports "
                f"only {self._max_options}; export a checkpoint with a larger "
                "max_options."
            )
        if max_count == 0:
            return []

        seat = state.yourIndex
        encoded = self._encode_observation(observation, seat, 0)
        picks: list[int] = []
        stop_index = self._max_options
        mask = torch.zeros(
            self._max_options + 1,
            dtype=torch.bool,
            device=self._device,
        )
        mask[:n_options] = True
        while len(picks) < max_count:
            if len(picks) >= min_count:
                mask[stop_index] = True
            chosen = self._sample_action(encoded, mask)
            if chosen == stop_index:
                break
            picks.append(chosen)
            mask[chosen] = False
            if len(picks) < max_count:
                # TCGEnv presents every partial pick as a new policy step. The
                # policy must therefore see the new count before scoring the
                # next action. Structured encoding can update its sole
                # count-dependent field in place; unknown encoders fall back
                # to a full re-encode.
                encoded_observation = encoded.get("observation")
                update_count = getattr(
                    self._encoder, "update_already_chosen_option_count", None
                )
                if update_count is None or not update_count(
                    encoded_observation, len(picks)
                ):
                    encoded = self._encode_observation(
                        observation, seat, len(picks)
                    )
        return picks

    def _encode_observation(
        self,
        observation: Observation,
        seat: int,
        already_chosen_option_count: int,
    ) -> TensorDict:
        """Encode and transfer one policy observation."""
        encoded = TensorDict(
            {
                "observation": self._encoder.encode(
                    observation, seat, already_chosen_option_count
                )
            },
            batch_size=torch.Size(()),
        )
        # On CPU, .to() would still walk every leaf on every engine decision.
        if self._device.type != "cpu":
            encoded = encoded.to(self._device, non_blocking=True)
        return encoded

    def _sample_action(
        self,
        encoded_observation: TensorDict,
        mask: torch.Tensor,
    ) -> int:
        """
        Run the policy and sample one legal action for an encoded observation.

        Uses PyTorch directly instead of constructing a TorchRL distribution
        object for every partial pick. ``mask`` is True for legal actions.

        :param encoded_observation: Model-ready observation TensorDict.
        :param mask: Bool tensor with the same shape, marking legal actions.
        :return: Sampled action index.
        """
        logits = self._actor_critic.policy_logits(encoded_observation)
        if logits.ndim != 1 or mask.shape != logits.shape:
            raise ValueError(
                f"Expected matching one-dimensional logits/mask, got "
                f"{tuple(logits.shape)} and {tuple(mask.shape)}."
            )
        if not bool(mask.any()):
            raise ValueError("Cannot sample an action from an empty mask.")
        probabilities = torch.softmax(logits.masked_fill(~mask, -torch.inf), dim=0)
        return int(torch.multinomial(probabilities, 1).item())


def load_inference_agent(
    checkpoint_path: str | Path,
    model_config_path: str | Path,
    device: torch.device | str = "cpu",
) -> InferenceAgent:
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
    :return: The checkpoint's agent, ready to act.
    """
    model_config = OmegaConf.load(model_config_path)
    max_options = int(model_config.max_options)
    obs_spec, encoder, action_spec = build_inference_specs(
        max_options=max_options,
        encoder_name=str(model_config.get("encoder", "structured")),
    )
    actor_critic = build_actor_critic(model_config, obs_spec, action_spec)
    target_device = torch.device(device)
    state_dict = torch.load(
        Path(checkpoint_path),
        map_location=target_device,
        weights_only=True,
        mmap=target_device.type == "cpu",
    )
    # No optimizer exists in serving, so assigning checkpoint tensors avoids
    # copying every parameter into the freshly constructed module.
    actor_critic.load_state_dict(state_dict, assign=True)
    return InferenceAgent(
        actor_critic,
        encoder,
        max_options=max_options,
        device=device,
    )
