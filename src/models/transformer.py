from collections.abc import Sequence
from typing import cast

import torch
from tensordict import TensorDictBase
from torch import nn

from src.models.backbone import Backbone

#: Activation names :class:`nn.TransformerEncoderLayer` accepts natively.
_ENCODER_LAYER_ACTIVATIONS = {"relu", "gelu"}


class TransformerBackbone(Backbone):
    """
    Self-attention trunk: one token per observation group, attending across
    groups before pooling to a single ``state_repr``.

    Where :class:`~src.models.mlp.MLPBackbone` flattens and concatenates every
    group into one vector, this backbone keeps each group (``globals``,
    ``options``, ``pokemon``, the zone tables, ...) as its own token, so a few
    self-attention layers can learn cross-group interactions (e.g. weighting
    ``options`` against the current ``pokemon`` board state) that a single
    linear layer over the concatenation cannot represent directly. Per Issue
    #45 this attends over the *feature groups within one observation*, not
    over time — a temporal/history transformer is a separate, later backbone.

    Deliberately shallow (the literature's baseline here is the flat MLP, and
    a single-observation feature sequence is short — around ten groups): the
    defaults are one encoder layer and four heads, not a deep stack. It emits
    no per-option tokens (:attr:`produces_option_repr` is False) and pairs
    with the same :class:`~src.models.heads.LinearPolicyHead` /
    :class:`~src.models.heads.ValueHead` as the MLP baseline.

    Requires a :class:`~src.models.structured_obs_adapter.StructuredObsAdapter`
    (built by :func:`~src.policies.ppo_actor.build_actor_critic` whenever the
    backbone's ``in_keys`` name structured groups): its
    :attr:`~src.models.structured_obs_adapter.StructuredObsAdapter.group_feature_widths`
    is what sizes the per-group input projections below, and its
    :meth:`~src.models.structured_obs_adapter.StructuredObsAdapter.encode_groups`
    supplies one pre-pooled vector per group per step, rather than the single
    flat vector :meth:`~src.models.structured_obs_adapter.StructuredObsAdapter.forward`
    returns.
    """

    produces_option_repr = False

    def __init__(
            self,
            input_dim: int,
            out_features: int,
            adapter: nn.Module | None = None,
            num_heads: int = 4,
            num_layers: int = 1,
            ff_dim: int = 256,
            dropout: float = 0.0,
            activation: str = "gelu",
            in_keys: list[str] | None = None,
    ) -> None:
        """
        :param input_dim: Summed width of the adapter's per-group vectors;
            must equal ``adapter.out_features`` (sanity check against
            ``in_keys`` mismatches, mirroring :class:`~src.models.mlp.MLPBackbone`).
        :param out_features: Width of the produced ``state_repr``; also the
            attention ``d_model``, so it must be divisible by ``num_heads``.
        :param adapter: :class:`~src.models.structured_obs_adapter.
            StructuredObsAdapter` handling the structured groups; required
            because its per-group widths size the token projections and its
            :meth:`~src.models.structured_obs_adapter.StructuredObsAdapter.encode_groups`
            supplies the per-group tokens; there is no naive-flatten fallback.
        :param num_heads: Attention heads per encoder layer.
        :param num_layers: Stacked :class:`nn.TransformerEncoderLayer` count.
        :param ff_dim: Width of each layer's feed-forward sublayer.
        :param dropout: Dropout used in attention and the feed-forward block.
        :param activation: Feed-forward activation; ``"relu"`` or ``"gelu"``
            (the two :class:`nn.TransformerEncoderLayer` supports natively).
        :param in_keys: Observation keys to consume; one token per key, in
            the same order as ``adapter.group_feature_widths``.
        :raises ValueError: If ``adapter`` is missing, its width disagrees
            with ``input_dim``, ``out_features`` isn't divisible by
            ``num_heads``, or ``activation`` is unsupported.
        """
        super().__init__(in_keys=in_keys or ["observation"], out_features=out_features)
        if adapter is None:
            raise ValueError(
                "TransformerBackbone requires a StructuredObsAdapter to derive "
                "per-group token widths and encodings; use MLPBackbone for the "
                "no-adapter (single flat observation) case."
            )
        if adapter.out_features != input_dim:
            raise ValueError(
                f"Adapter produces {adapter.out_features} features but the transformer "
                f"expects input_dim={input_dim}."
            )
        if out_features % num_heads != 0:
            raise ValueError(
                f"out_features={out_features} must be divisible by num_heads={num_heads}."
            )
        if activation not in _ENCODER_LAYER_ACTIVATIONS:
            raise ValueError(
                f"Unknown activation '{activation}'; expected one of {sorted(_ENCODER_LAYER_ACTIVATIONS)}."
            )
        self.adapter = adapter
        token_widths = cast(Sequence[int], adapter.group_feature_widths)
        self.token_projections = nn.ModuleList(
            [nn.Linear(width, out_features) for width in token_widths]
        )
        # Learned per-group identity: attention over these tokens would
        # otherwise be permutation-invariant to which group is which.
        self.token_type_embedding = nn.Parameter(torch.zeros(len(token_widths), out_features))
        nn.init.normal_(self.token_type_embedding, std=0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=out_features,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation=activation,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, *inputs: torch.Tensor | TensorDictBase) -> torch.Tensor:
        """
        Encode each group to a token, attend across them, and mean-pool.

        :param inputs: One entry per :attr:`in_keys`, handed to the adapter.
        :return: ``state_repr`` of shape ``(*batch, out_features)``.
        """
        if len(inputs) != len(self.in_keys):
            raise ValueError(
                f"TransformerBackbone expected {len(self.in_keys)} inputs for keys "
                f"{self.in_keys}, got {len(inputs)}."
            )
        group_vectors = self.adapter.encode_groups(*inputs)
        tokens = torch.stack(
            [proj(vector) for proj, vector in zip(self.token_projections, group_vectors, strict=True)],
            dim=-2,
        )
        tokens = tokens + self.token_type_embedding

        # nn.TransformerEncoder only accepts a single leading batch dim;
        # flatten any extra ones (torchrl rollouts are often (*batch, feature)
        # with batch itself multi-dimensional, e.g. (time, env)) and restore
        # them after pooling out the token dimension.
        batch_shape = tokens.shape[:-2]
        flat_tokens = tokens.reshape(-1, tokens.shape[-2], tokens.shape[-1])
        encoded = self.encoder(flat_tokens)
        # Mean pooling keeps the readout simple and permutation-invariant
        # over groups, matching the deliberately shallow scope of this trunk.
        pooled = encoded.mean(dim=-2)
        return pooled.reshape(*batch_shape, self.out_features)
