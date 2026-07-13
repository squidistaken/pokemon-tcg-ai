import torch
from torch import nn
from torchrl.modules import MLP

from .backbone import activation_class


class LinearPolicyHead(nn.Module):
    """
    Flat policy head: a single linear map from ``state_repr`` to action logits.

    Produces one logit per action slot (options ``0..max_options-1`` plus the
    synthetic stop at ``max_options``), i.e. ``(..., n_actions)``. This is the
    MLP baseline's head — it reads only the global ``state_repr`` and ignores
    per-option tokens, so it pairs with any backbone. Illegal actions are
    zeroed downstream by ``MaskedCategorical`` against the ``action_mask``; the
    head itself is mask-agnostic.
    """

    #: This head reads only ``state_repr`` and needs no per-option tokens.
    requires_option_repr = False

    def __init__(self, in_features: int, n_actions: int) -> None:
        """
        :param in_features: Width of the incoming ``state_repr``.
        :param n_actions: Size of the action space (``max_options + 1``).
        """
        super().__init__()
        self.linear = nn.Linear(in_features, n_actions)

    def forward(
            self,
            state_repr: torch.Tensor,
            _option_repr: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Map the latent state to per-action logits.

        :param state_repr: Latent state of shape ``(..., in_features)``.
        :param _option_repr: Unused; accepted so the head shares the pointer
            head's signature and can be swapped without touching the trunk.
        :return: Logits of shape ``(..., n_actions)``.
        """
        return self.linear(state_repr)


class ValueHead(nn.Module):
    """
    State-value critic: an MLP mapping ``state_repr`` to a scalar value.

    Emits ``state_value`` of shape ``(..., 1)`` for PPO's advantage estimation.
    """

    def __init__(
            self,
            in_features: int,
            num_cells: list[int],
            activation: str = "tanh",
    ) -> None:
        """
        :param in_features: Width of the incoming ``state_repr``.
        :param num_cells: Hidden layer widths of the value MLP.
        :param activation: Hidden activation name (see
            :func:`src.models.backbone.activation_class`).
        """
        super().__init__()
        self.mlp = MLP(
            in_features=in_features,
            out_features=1,
            num_cells=list(num_cells),
            activation_class=activation_class(activation),
        )

    def forward(self, state_repr: torch.Tensor) -> torch.Tensor:
        """
        Estimate the state value from the latent state.

        :param state_repr: Latent state of shape ``(..., in_features)``.
        :return: State value of shape ``(..., 1)``.
        """
        return self.mlp(state_repr)
