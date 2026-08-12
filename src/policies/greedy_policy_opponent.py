from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import torch
from omegaconf import DictConfig, OmegaConf
from tensordict import TensorDict
from torchrl.data import Composite, TensorSpec

from cg.api import Observation
from src.env.observation.observation_encoder import ObservationEncoder
from src.models.actor_critic import ActorCritic
from src.policies.ppo_actor import build_actor_critic

CHECKPOINT_FORMAT_VERSION = 1


def save_actor_critic(
    actor_critic: ActorCritic,
    path: str | Path,
    *,
    config: Mapping[str, Any] | None = None,
    frames: int | None = None,
) -> Path:
    """
    Snapshot an actor-critic's parameters to disk.

    With ``config=None`` this saves the legacy bare ``state_dict()`` format.
    Training passes the small inference-time config, producing a versioned
    checkpoint that can rebuild itself for Kaggle packaging without relying on
    the Hydra output directory still being present.

    :param actor_critic: Actor-critic to snapshot.
    :param path: Destination file path.
    :param config: Resolved inference-time model/environment config to embed.
    :param frames: Collected-frame counter for this checkpoint.
    :return: The path written to.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state_dict = actor_critic.state_dict()
    payload: object = state_dict
    if config is not None:
        payload = {
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "state_dict": state_dict,
            "config": dict(config),
            "frames": frames,
        }
    torch.save(payload, path)
    return path


def checkpoint_state_dict(payload: object) -> Mapping[str, torch.Tensor]:
    """
    Return model parameters from either the versioned or legacy checkpoint.

    :param payload: Object returned by :func:`torch.load`.
    :return: Actor-critic state dict.
    :raises ValueError: If a versioned checkpoint has no mapping state dict.
    """
    if isinstance(payload, Mapping) and "state_dict" in payload:
        state_dict = payload["state_dict"]
        if not isinstance(state_dict, Mapping):
            raise ValueError("Checkpoint 'state_dict' must be a mapping.")
        return cast(Mapping[str, torch.Tensor], state_dict)
    if not isinstance(payload, Mapping):
        raise TypeError("Checkpoint must contain a state-dict mapping.")
    return cast(Mapping[str, torch.Tensor], payload)


def checkpoint_model_config(payload: object) -> DictConfig | None:
    """
    Return the ``model`` config a versioned checkpoint was trained with.

    :param payload: Object returned by :func:`torch.load`.
    :return: The embedded ``model`` section, or None for a legacy checkpoint
        that carries a bare state dict and so has no config to rebuild from.
    """
    if not isinstance(payload, Mapping):
        return None
    config = payload.get("config")
    if not isinstance(config, Mapping) or "model" not in config:
        return None
    return cast(DictConfig, OmegaConf.create({"model": dict(config["model"])}))


class GreedyPolicyOpponent:
    """
    Deterministic policy opponent wrapping a trained actor-critic.

    This is the self-play snapshot: a frozen network dropped into the opponent
    seat via the environment's ``opponent`` callable / an
    :class:`~src.env.opponents.opponent_pool.OpponentPool`. Like
    :class:`~src.env.opponents.random_opponent.RandomOpponent`, it answers a whole engine
    selection in one call (the environment does not decompose the opponent's
    multi-select), so it mirrors the Kaggle ``main.py`` inference path: encode
    the observation from the acting seat, score the option slots with the
    policy head, and take the highest-scoring legal option, using the learned
    **stop** logit to decide when to stop within ``[minCount, maxCount]``.

    The wrapped :class:`~src.models.actor_critic.ActorCritic` and the encoder
    are used exactly as in training, keeping train/serve behavior aligned.
    """

    def __init__(
        self,
        actor_critic: ActorCritic,
        encoder: ObservationEncoder,
        device: torch.device | str = "cpu",
    ) -> None:
        """
        :param actor_critic: Trained actor-critic to act greedily with; put
            into eval mode and never updated here.
        :param encoder: Observation encoder matching the one used in training
            (the flat encoder for the Phase-1 baseline).
        :param device: Device for inference.
        """
        self._device = torch.device(device)
        self._actor_critic = actor_critic.to(self._device).eval()
        self._encoder = encoder

    @torch.inference_mode()
    def __call__(self, observation: Observation) -> list[int]:
        """
        Choose a greedy legal selection for the acting seat.

        :param observation: Current engine observation with a non-None select;
            ``current.yourIndex`` gives the acting seat used for encoding.
        :return: Option indices to submit, between ``minCount`` and
            ``maxCount`` entries with no duplicates.
        """
        select = observation.select
        state = observation.current
        if select is None or state is None:
            raise ValueError(
                "GreedyPolicyOpponent requires an observation with current state and select."
            )
        seat = state.yourIndex
        # Match TCGEnv, which nests the encoder output under "observation".
        encoded = TensorDict(
            {"observation": self._encoder.encode(observation, seat, 0)},
            batch_size=torch.Size(()),
        )
        # Guarded: on CPU .to() would still walk and copy
        # every leaf for nothing, on every opponent move.
        if self._device.type != "cpu":
            encoded = encoded.to(self._device)

        n_options = len(select.option)
        min_count = select.minCount
        max_count = min(select.maxCount, n_options)
        picks: list[int] = []
        while len(picks) < max_count:
            logits = self._actor_critic.policy_logits(encoded)
            chosen = self.greedy_pick(logits, n_options, picks, len(picks) >= min_count)
            if chosen is None:
                break
            picks.append(chosen)
            if len(picks) < max_count:
                self._reencode(encoded, observation, seat, len(picks))
        return picks

    def _reencode(
        self,
        encoded: TensorDict,
        observation: Observation,
        seat: int,
        already_chosen_option_count: int,
    ) -> None:
        """
        Refresh the encoding for the next partial pick, in place where possible.

        :param encoded: Policy-ready tensordict to update.
        :param observation: Engine observation being re-encoded.
        :param seat: Acting seat.
        :param already_chosen_option_count: Picks accumulated so far.
        """
        # Structured encoding can update its sole count-dependent field in
        # place; unknown encoders fall back to a full re-encode.
        update_count = getattr(
            self._encoder, "update_already_chosen_option_count", None
        )
        if update_count is not None and update_count(
            encoded.get("observation"), already_chosen_option_count
        ):
            return
        refreshed = self._encoder.encode(observation, seat, already_chosen_option_count)
        encoded.set("observation", refreshed.to(self._device))

    @staticmethod
    def greedy_pick(
        logits: torch.Tensor,
        n_options: int,
        already_chosen: list[int],
        stop_allowed: bool,
    ) -> int | None:
        """
        Take the best legal option, or stop.

        :param logits: Action logits of shape ``(n_actions,)``; the last entry
            is the synthetic **stop**.
        :param n_options: Number of real options offered by the selection.
        :param already_chosen: Option indices picked so far, which the engine
            rejects as duplicates and which are therefore excluded here.
        :param stop_allowed: Whether ``minCount`` has been met, making stop legal.
        :return: The chosen option index, or None to stop.
        """
        capacity = logits.shape[-1] - 1
        n_options = min(n_options, capacity)
        if n_options <= 0:
            # A selection offering np options can only be answered with an empty submission.
            return None
        scores = logits[:n_options].clone()
        if already_chosen:
            scores[already_chosen] = -torch.inf
        best = int(torch.argmax(scores).item())
        if not torch.isfinite(scores[best]):
            return None
        if stop_allowed and scores[best] <= logits[capacity]:
            return None
        return best

    @staticmethod
    def greedy_select(
        logits: torch.Tensor,
        n_options: int,
        min_count: int,
        max_count: int,
    ) -> list[int]:
        """
        Resolve a whole selection from one fixed set of action logits.

        Options ``0..n_options-1`` map to logits ``0..n_options-1``; the final
        logit is the synthetic **stop**. Options are taken in descending logit
        order until ``max_count`` is reached, or until ``min_count`` is met and
        the next-best option scores no higher than stop.

        :param logits: Action logits of shape ``(n_actions,)`` where
            ``n_actions = max_options + 1``.
        :param n_options: Number of real options offered by the selection.
        :param min_count: Minimum number of options to pick.
        :param max_count: Maximum number of options to pick.
        :return: Chosen option indices (a subset of ``range(n_options)``).
        """
        max_count = min(max_count, min(n_options, logits.shape[-1] - 1))
        picks: list[int] = []
        while len(picks) < max_count:
            chosen = GreedyPolicyOpponent.greedy_pick(
                logits, n_options, picks, len(picks) >= min_count
            )
            if chosen is None:
                break
            picks.append(chosen)
        return picks


def load_actor_critic(
    checkpoint_path: str | Path,
    cfg: DictConfig,
    obs_spec: Composite,
    action_spec: TensorSpec,
    device: torch.device | str = "cpu",
) -> ActorCritic:
    """
    Rebuild an actor-critic from a snapshot and load its weights.

    The architecture comes from the config the checkpoint itself embeds, so a
    frozen reference keeps loading after the *current* run's architecture has
    moved on — which is what makes "am I better than the agent we already
    submitted" answerable at all. ``cfg`` is the fallback for a legacy
    checkpoint saved as a bare state dict, which carries no config of its own.
    Without this, evaluating against any earlier-architecture checkpoint fails
    on a size mismatch at the first evaluation round, minutes into a run.

    :param checkpoint_path: Path to a :func:`save_actor_critic` snapshot.
    :param cfg: Hydra config used to build the architecture when the
        checkpoint embeds none.
    :param obs_spec: Environment observation composite spec.
    :param action_spec: Environment action spec.
    :param device: Device to load the weights onto.
    :return: The reconstructed actor-critic with the checkpoint's weights.
    """
    payload = torch.load(Path(checkpoint_path), map_location=device, weights_only=True)
    embedded = checkpoint_model_config(payload)
    actor_critic = build_actor_critic(embedded or cfg, obs_spec, action_spec)
    actor_critic.load_state_dict(checkpoint_state_dict(payload))
    return actor_critic


def load_greedy_opponent(
    checkpoint_path: str | Path,
    cfg: DictConfig,
    obs_spec: Composite,
    action_spec: TensorSpec,
    encoder: ObservationEncoder,
    device: torch.device | str = "cpu",
) -> GreedyPolicyOpponent:
    """
    Load a snapshot and wrap it as a greedy opponent.

    :param checkpoint_path: Path to a :func:`save_actor_critic` snapshot.
    :param cfg: Hydra config used to build the matching architecture.
    :param obs_spec: Environment observation composite spec.
    :param action_spec: Environment action spec.
    :param encoder: Observation encoder matching training.
    :param device: Device for inference.
    :return: A greedy opponent playing the snapshot.
    """
    actor_critic = load_actor_critic(
        checkpoint_path, cfg, obs_spec, action_spec, device
    )
    return GreedyPolicyOpponent(actor_critic, encoder, device=device)
