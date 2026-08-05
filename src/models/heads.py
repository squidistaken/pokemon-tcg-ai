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
            option_repr: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Map the latent state to per-action logits.

        :param state_repr: Latent state of shape ``(..., in_features)``.
        :param option_repr: Unused; accepted so the head shares the pointer
            head's signature and can be swapped without touching the trunk.
        :return: Logits of shape ``(..., n_actions)``.
        """
        del option_repr
        return self.linear(state_repr)


class PointerPolicyHead(nn.Module):
    """
    Pointer head: score each per-option token against a query from the state.

    The flat :class:`LinearPolicyHead` maps a pooled state to one logit per
    action *slot*, so it can only learn positional preferences — "pick slot
    3" — because the identity of the option occupying that slot never reaches
    it. This head instead scores option ``i``'s own representation, so the
    policy can learn *what* an option does rather than where it sits in the
    table. That is the standard pointer-network formulation, and the
    literature's answer to a variable-length, permutation-arbitrary action
    set (see ``docs/architecture/ppo-transformer-actor-critic.md`` §3).

    Cost is linear in the option count — one dot product per option — unlike
    putting the option tokens through the trunk's self-attention, which is
    quadratic. Requires a backbone that emits ``option_repr``
    (:attr:`~src.models.backbone.Backbone.produces_option_repr`).

    The option table has one row per action slot including the synthetic stop
    at index ``max_options``, so scoring every row yields exactly the
    ``n_actions`` logits the action spec expects; no separate stop logit is
    needed. Illegal actions are zeroed downstream by ``MaskedCategorical``
    against the ``action_mask``; the head itself is mask-agnostic.
    """

    #: This head cannot run on ``state_repr`` alone.
    requires_option_repr = True

    def __init__(self, in_features: int, n_actions: int) -> None:
        """
        :param in_features: Width of the incoming ``state_repr``, which is
            also the width of each option token.
        :param n_actions: Size of the action space (``max_options + 1``);
            must match the option table's slot count.
        """
        super().__init__()
        self.n_actions = n_actions
        self.query = nn.Linear(in_features, in_features)
        # Scaled dot product, as in attention: without it the logits' scale
        # grows with in_features and the initial policy is near-deterministic.
        self._scale = float(in_features) ** 0.5

    def forward(
            self,
            state_repr: torch.Tensor,
            option_repr: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Score every option token against a state-derived query.

        :param state_repr: Latent state of shape ``(..., in_features)``.
        :param option_repr: Per-option tokens of shape
            ``(..., n_actions, in_features)``.
        :return: Logits of shape ``(..., n_actions)``.
        :raises ValueError: If ``option_repr`` is missing, or its slot count
            disagrees with the action space.
        """
        if option_repr is None:
            raise ValueError(
                "PointerPolicyHead needs per-option tokens; pair it with a backbone "
                "whose produces_option_repr is True."
            )
        if option_repr.shape[-2] != self.n_actions:
            raise ValueError(
                f"option_repr has {option_repr.shape[-2]} slots but the action space has "
                f"{self.n_actions}; the option table must carry one row per action."
            )
        query = self.query(state_repr).unsqueeze(-2)
        return (option_repr * query).sum(dim=-1) / self._scale


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
