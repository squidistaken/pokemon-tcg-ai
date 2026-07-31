from pathlib import Path

import torch
from omegaconf import DictConfig
from tensordict import TensorDict
from torchrl.data import Composite, TensorSpec

from cg.api import Observation
from src.env.observation_encoder import ObservationEncoder
from src.models.actor_critic import ActorCritic
from src.policies.ppo_actor import build_actor_critic


def save_actor_critic(actor_critic: ActorCritic, path: str | Path) -> Path:
    """
    Snapshot an actor-critic's parameters to disk.

    Saves ``state_dict()`` (stable ``backbone.* / policy_head.* /
    value_head.*`` keys) so it reloads into a fresh
    :func:`~src.policies.ppo_actor.build_actor_critic` regardless of the
    torchrl operator wiring used for training.

    :param actor_critic: Actor-critic to snapshot.
    :param path: Destination file path.
    :return: The path written to.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(actor_critic.state_dict(), path)
    return path


class GreedyPolicyOpponent:
    """
    Deterministic policy opponent wrapping a trained actor-critic.

    This is the self-play snapshot: a frozen network dropped into the opponent
    seat via the environment's ``opponent`` callable / an
    :class:`~src.env.opponent_pool.OpponentPool`. Like
    :class:`~src.env.random_opponent.RandomOpponent`, it answers a whole engine
    selection in one call (the environment does not decompose the opponent's
    multi-select), so it mirrors the Kaggle ``main.py`` inference path: encode
    the observation from the acting seat, score the option slots with the
    policy head, and greedily take the highest-scoring legal options, using the
    learned **stop** logit to decide how many to take within
    ``[minCount, maxCount]``.

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
            raise ValueError("GreedyPolicyOpponent requires an observation with current state and select.")
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
        logits = self._actor_critic.policy_logits(encoded)
        return self.greedy_select(
            logits,
            n_options=len(select.option),
            min_count=select.minCount,
            max_count=select.maxCount,
        )

    @staticmethod
    def greedy_select(
            logits: torch.Tensor,
            n_options: int,
            min_count: int,
            max_count: int,
    ) -> list[int]:
        """
        Greedily pick option indices from action logits.

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
        capacity = logits.shape[-1] - 1
        n_options = min(n_options, capacity)
        max_count = min(max_count, n_options)
        stop_logit = logits[capacity]
        order = torch.argsort(logits[:n_options], descending=True).tolist()
        picks: list[int] = []
        for index in order:
            if len(picks) >= max_count:
                break
            if len(picks) >= min_count and logits[index] <= stop_logit:
                break
            picks.append(int(index))
        return picks


def load_greedy_opponent(
        checkpoint_path: str | Path,
        cfg: DictConfig,
        obs_spec: Composite,
        action_spec: TensorSpec,
        encoder: ObservationEncoder,
        device: torch.device | str = "cpu",
) -> GreedyPolicyOpponent:
    """
    Rebuild an actor-critic from config and load a snapshot as an opponent.

    Reconstructs the network with :func:`~src.policies.ppo_actor.
    build_actor_critic` (so it matches the trained architecture), loads the
    checkpoint written by :func:`save_actor_critic`, and wraps it as a
    :class:`GreedyPolicyOpponent`. Per the ``ParallelEnv`` worker-isolation
    caveat, opponents load the network from disk in each worker rather than
    sharing a Python object across processes.

    :param checkpoint_path: Path to a :func:`save_actor_critic` snapshot.
    :param cfg: Hydra config used to build the matching architecture.
    :param obs_spec: Environment observation composite spec.
    :param action_spec: Environment action spec.
    :param encoder: Observation encoder matching training.
    :param device: Device for inference.
    :return: A greedy opponent playing the snapshot.
    """
    actor_critic = build_actor_critic(cfg, obs_spec, action_spec)
    state_dict = torch.load(Path(checkpoint_path), map_location=device, weights_only=True)
    actor_critic.load_state_dict(state_dict)
    return GreedyPolicyOpponent(actor_critic, encoder, device=device)
