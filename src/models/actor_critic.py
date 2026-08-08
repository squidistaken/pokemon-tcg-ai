from tensordict import TensorDictBase
from torch import Tensor, nn

from .backbone import Backbone


class ActorCritic(nn.Module):
    """
    Shared-trunk actor-critic: one backbone feeding a policy and a value head.

    A single forward pass runs the observation through the shared
    :class:`~src.models.backbone.Backbone`, then fans the resulting latent out
    to both heads (weight sharing -> sample efficiency, the standard PPO /
    ByteRL pattern). Writes ``logits`` (``(..., n_actions)``) and
    ``state_value`` (``(..., 1)``) into the tensordict.

    This module is the standalone, tensordict-in/tensordict-out core used for
    unit testing and for greedy inference (see
    :class:`~src.policies.greedy_policy_opponent.GreedyPolicyOpponent`). The
    training path wraps the *same* backbone and head instances in a torchrl
    :class:`~torchrl.modules.ActorValueOperator` via
    :func:`src.policies.ppo_actor.build_ppo_actor_critic`, so parameters are
    shared between the two views.
    """

    def __init__(
        self,
        backbone: Backbone,
        policy_head: nn.Module,
        value_head: nn.Module,
    ) -> None:
        """
        :param backbone: Shared trunk producing ``state_repr`` (and optionally
            ``option_repr``) from the observation keys in ``backbone.in_keys``.
        :param policy_head: Head mapping the latent(s) to action logits.
        :param value_head: Head mapping ``state_repr`` to a scalar value.
        """
        super().__init__()
        self.backbone = backbone
        self.policy_head = policy_head
        self.value_head = value_head

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        """
        Encode the observation and write policy logits and the state value.

        :param tensordict: Input tensordict carrying the backbone's
            ``in_keys``; the observation keys the flat baseline reads are the
            single ``"observation"`` vector.
        :return: The same tensordict with ``"logits"`` and ``"state_value"``
            written in.
        """
        inputs = [tensordict.get(key) for key in self.backbone.in_keys]
        encoded = self.backbone(*inputs)
        if self.backbone.produces_option_repr:
            state_repr, option_repr = encoded
        else:
            state_repr, option_repr = encoded, None
        tensordict.set("logits", self.policy_head(state_repr, option_repr))
        tensordict.set("state_value", self.value_head(state_repr))
        return tensordict

    def policy_logits(self, tensordict: TensorDictBase) -> Tensor:
        """
        Compute action logits only, skipping the value head.

        Used by inference-time consumers that never read the critic, most
        notably :class:`~src.policies.greedy_policy_opponent.GreedyPolicyOpponent`
        running inside the environment workers, where the discarded value pass
        is pure overhead on every opponent move.

        :param tensordict: Input tensordict carrying the backbone's ``in_keys``.
        :return: Action logits of shape ``(..., n_actions)``.
        """
        inputs = [tensordict.get(key) for key in self.backbone.in_keys]
        encoded = self.backbone(*inputs)
        if self.backbone.produces_option_repr:
            state_repr, option_repr = encoded
        else:
            state_repr, option_repr = encoded, None
        return self.policy_head(state_repr, option_repr)
