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


class PointerHead(nn.Module):
    """
    Per-option scoring head: one shared scorer applied to every action slot.

    The flat :class:`LinearPolicyHead` produces logit ``i`` from the pooled
    state alone, so it can only ever learn a prior over slot *indices* — the
    option table reaches it as a permutation-invariant mean, and shuffling the
    options leaves its output bit-identical while the correct action moves.
    This head instead scores slot ``i`` from ``[state_repr, option_repr_i]``
    with weights shared across slots, making it permutation-**equivariant**:
    reorder the options and the logits follow. That is the property a policy
    needs in order to pick an action for what it does rather than where it sits.

    Sharing one scorer across slots also means every option seen anywhere in
    the table trains the same parameters, instead of each slot index having to
    learn its own mapping from its own visits.

    The synthetic **stop** action has no option row to score (its slot is
    padding), so its logit comes from a separate state-only branch.
    """

    #: This head scores per-option tokens and cannot run without them.
    requires_option_repr = True

    def __init__(
            self,
            in_features: int,
            n_actions: int,
            option_dim: int,
            num_cells: list[int] | None = None,
            activation: str = "tanh",
    ) -> None:
        """
        :param in_features: Width of the incoming ``state_repr``.
        :param n_actions: Size of the action space (``max_options + 1``); the
            last index is the synthetic stop action.
        :param option_dim: Width of one per-option token.
        :param num_cells: Hidden widths of the shared scorer; defaults to
            ``[in_features]``.
        :param activation: Hidden activation name (see
            :func:`src.models.backbone.activation_class`).
        """
        super().__init__()
        self.n_actions = n_actions
        self.n_option_slots = n_actions - 1
        hidden = list(num_cells) if num_cells else [in_features]
        self.scorer = MLP(
            in_features=in_features + option_dim,
            out_features=1,
            num_cells=hidden,
            activation_class=activation_class(activation),
        )
        self.stop_scorer = MLP(
            in_features=in_features,
            out_features=1,
            num_cells=hidden,
            activation_class=activation_class(activation),
        )

    def forward(
            self,
            state_repr: torch.Tensor,
            option_repr: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Score every option slot against the state, plus the stop action.

        :param state_repr: Latent state of shape ``(..., in_features)``.
        :param option_repr: Per-option tokens of shape
            ``(..., n_slots, option_dim)``; ``n_slots`` must cover the
            ``n_actions - 1`` real option slots.
        :return: Logits of shape ``(..., n_actions)``.
        :raises ValueError: If the option tokens are missing or too few.
        """
        if option_repr is None:
            raise ValueError("PointerHead requires per-option tokens; got None.")
        if option_repr.shape[-2] < self.n_option_slots:
            raise ValueError(
                f"PointerHead needs at least {self.n_option_slots} option slots, "
                f"got {option_repr.shape[-2]}."
            )
        options = option_repr[..., : self.n_option_slots, :]
        broadcast_state = state_repr.unsqueeze(-2).expand(
            *state_repr.shape[:-1], self.n_option_slots, state_repr.shape[-1]
        )
        option_logits = self.scorer(
            torch.cat([broadcast_state, options], dim=-1)
        ).squeeze(-1)
        return torch.cat([option_logits, self.stop_scorer(state_repr)], dim=-1)


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
