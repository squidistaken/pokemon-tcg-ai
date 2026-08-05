from collections.abc import Sequence
from typing import cast

import torch
from tensordict import TensorDictBase
from torch import nn

from src.models.backbone import Backbone

#: Activation names :class:`nn.TransformerEncoderLayer` accepts natively.
_ENCODER_LAYER_ACTIVATIONS = {"relu", "gelu"}
#: Readout strategies :class:`TransformerBackbone` supports.
_POOLING_MODES = {"mean", "cls", "attention"}


class TransformerBackbone(Backbone):
    """
    Self-attention trunk over the structured observation.

    Where :class:`~src.models.mlp.MLPBackbone` flattens and concatenates every
    group into one vector, this backbone keeps each group (``globals``,
    ``options``, ``pokemon``, the zone tables, ...) as its own token, so
    self-attention can learn cross-group interactions (e.g. weighting
    ``options`` against the current ``pokemon`` board state) that a single
    linear layer over the concatenation cannot represent directly. Per Issue
    #45 this attends over the *features within one observation*, not over
    time — a temporal/history transformer is a separate, later backbone.

    **Two token granularities.** By default every group contributes exactly
    one token, built from the adapter's masked-mean-pooled group vector. That
    is cheap but means the trunk never sees individual entities: the pooling
    has already averaged the option rows together before attention runs. Set
    ``token_groups`` to additionally expand named groups into *per-entity*
    tokens via
    :meth:`~src.models.structured_obs_adapter.StructuredObsAdapter.encode_entity_tokens`,
    so attention relates one Pokemon (or one option) to another. This is
    deliberately opt-in and per-group: the padded slot counts are large
    (``options`` alone is ``max_options + 1``), attention is quadratic in the
    token count, and the league opponent forward runs on CPU inside every
    environment worker — see ``docs/training-performance.md``, where that
    forward is already 46% of throughput.

    **Feeding a pointer head.** ``option_tokens=True`` additionally emits the
    per-option tokens as ``option_repr``, which
    :class:`~src.models.heads.PointerPolicyHead` scores individually. That
    costs one projection per option rather than quadratic attention, so it is
    much cheaper than putting ``options`` in ``token_groups`` — and it is what
    lets the policy distinguish *which* option is which, rather than choosing
    among action slots from a pooled state alone.

    Requires a :class:`~src.models.structured_obs_adapter.StructuredObsAdapter`
    (built by :func:`~src.policies.ppo_actor.build_actor_critic` whenever the
    backbone's ``in_keys`` name structured groups): its
    :attr:`~src.models.structured_obs_adapter.StructuredObsAdapter.group_feature_widths`
    sizes the per-group input projections below.
    """

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
            norm_first: bool = False,
            final_norm: bool = False,
            pooling: str = "mean",
            token_groups: Sequence[str] = (),
            option_tokens: bool = False,
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
            because its per-group widths size the token projections; there is
            no naive-flatten fallback.
        :param num_heads: Attention heads per encoder layer.
        :param num_layers: Stacked :class:`nn.TransformerEncoderLayer` count.
        :param ff_dim: Width of each layer's feed-forward sublayer. The
            conventional ratio is ``4 * out_features``.
        :param dropout: Dropout used in attention and the feed-forward block.
        :param activation: Feed-forward activation; ``"relu"`` or ``"gelu"``
            (the two :class:`nn.TransformerEncoderLayer` supports natively).
        :param norm_first: Pre-LN (True) rather than post-LN (False). Post-LN
            is torch's default but is the variant that needs learning-rate
            warmup to train stably; pre-LN trains without it, which matters
            here because the PPO config has no warmup schedule.
        :param final_norm: Apply a :class:`nn.LayerNorm` to the encoder
            output. Conventional with pre-LN, where the last sublayer's
            residual branch is otherwise unnormalized.
        :param pooling: Readout over the token sequence. ``"mean"`` is a
            masked mean; ``"cls"`` prepends a learned token and reads it back;
            ``"attention"`` scores tokens against a learned query. Mean is
            permutation-invariant over groups and cannot preferentially read
            one group out, which is a real limitation once tokens carry
            heterogeneous content.
        :param token_groups: Group names to additionally expand into
            per-entity tokens (e.g. ``["pokemon"]``). Costs
            ``adapter.group_slot_counts[name]`` extra tokens each.
        :param option_tokens: Emit per-option tokens as ``option_repr`` for a
            pointer head. Sets :attr:`produces_option_repr`.
        :param in_keys: Observation keys to consume; one token per key, in
            the same order as ``adapter.group_feature_widths``.
        :raises ValueError: If ``adapter`` is missing, its width disagrees
            with ``input_dim``, ``out_features`` isn't divisible by
            ``num_heads``, or ``activation``/``pooling`` is unsupported.
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
        if pooling not in _POOLING_MODES:
            raise ValueError(
                f"Unknown pooling '{pooling}'; expected one of {sorted(_POOLING_MODES)}."
            )
        self.adapter = adapter
        self.pooling = pooling
        self.token_groups = list(token_groups)
        self.option_tokens = bool(option_tokens)
        self.produces_option_repr = self.option_tokens

        token_widths = cast(Sequence[int], adapter.group_feature_widths)
        self.token_projections = nn.ModuleList(
            [nn.Linear(width, out_features) for width in token_widths]
        )
        # Learned per-group identity: attention over these tokens would
        # otherwise be permutation-invariant to which group is which.
        self.token_type_embedding = nn.Parameter(torch.zeros(len(token_widths), out_features))
        nn.init.normal_(self.token_type_embedding, std=0.02)

        # Per-entity token path. Entities arrive at the adapter's entity_dim
        # regardless of group, so one projection per group suffices; the
        # accompanying type embedding keeps "an option" distinguishable from
        # "a Pokemon" after they are concatenated into one sequence.
        entity_dim = getattr(adapter, "_entity_dim", None)
        self._needs_entity_tokens = sorted(set(self.token_groups) | ({"options"} if self.option_tokens else set()))
        if self._needs_entity_tokens and entity_dim is None:
            raise ValueError(
                "token_groups/option_tokens need per-entity encodings, which require the "
                "adapter to be built with entity_dim set (the default)."
            )
        self.entity_projections = nn.ModuleDict(
            {name: nn.Linear(int(cast(int, entity_dim)), out_features) for name in self._needs_entity_tokens}
        )
        self.entity_type_embedding = nn.ParameterDict(
            {name: nn.Parameter(torch.zeros(out_features)) for name in self._needs_entity_tokens}
        )
        for parameter in self.entity_type_embedding.values():
            nn.init.normal_(parameter, std=0.02)

        if pooling == "cls":
            self.cls_token = nn.Parameter(torch.zeros(out_features))
            nn.init.normal_(self.cls_token, std=0.02)
        elif pooling == "attention":
            self.pool_query = nn.Parameter(torch.zeros(out_features))
            nn.init.normal_(self.pool_query, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=out_features,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation=activation,
            norm_first=norm_first,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(out_features) if final_norm else None,
            # The nested-tensor fast path silently ignores some mask/norm
            # combinations; the token sequence here is short enough that its
            # padding-skip is not worth the divergence in behaviour.
            enable_nested_tensor=False,
        )

    def _group_tokens(self, inputs: tuple) -> tuple[torch.Tensor, torch.Tensor]:
        """
        One projected token per observation group, all of them valid.

        :param inputs: The backbone's forward inputs.
        :return: ``(tokens, valid)`` shaped ``(*batch, n_groups, d_model)``
            and ``(*batch, n_groups)``.
        """
        group_vectors = self.adapter.encode_groups(*inputs)
        tokens = torch.stack(
            [proj(vector) for proj, vector in zip(self.token_projections, group_vectors, strict=True)],
            dim=-2,
        )
        tokens = tokens + self.token_type_embedding
        valid = tokens.new_ones(tokens.shape[:-1], dtype=torch.bool)
        return tokens, valid

    def forward(self, *inputs: torch.Tensor | TensorDictBase):
        """
        Encode the observation into ``state_repr`` (and optionally per-option
        tokens).

        :param inputs: One entry per :attr:`in_keys`, handed to the adapter.
        :return: ``state_repr`` of shape ``(*batch, out_features)``, or a
            ``(state_repr, option_repr)`` pair when :attr:`option_tokens` is
            set, with ``option_repr`` shaped ``(*batch, n_slots, out_features)``.
        """
        if len(inputs) != len(self.in_keys):
            raise ValueError(
                f"TransformerBackbone expected {len(self.in_keys)} inputs for keys "
                f"{self.in_keys}, got {len(inputs)}."
            )
        tokens, valid = self._group_tokens(inputs)

        entity_tokens: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        if self._needs_entity_tokens:
            entity_tokens = self.adapter.encode_entity_tokens(
                *inputs, groups=self._needs_entity_tokens
            )
        for name in self.token_groups:
            group_tokens, group_valid = entity_tokens[name]
            projected = self.entity_projections[name](group_tokens) + self.entity_type_embedding[name]
            tokens = torch.cat([tokens, projected], dim=-2)
            valid = torch.cat([valid, group_valid], dim=-1)

        if self.pooling == "cls":
            cls = self.cls_token.expand(*tokens.shape[:-2], 1, tokens.shape[-1])
            tokens = torch.cat([cls, tokens], dim=-2)
            valid = torch.cat([valid.new_ones(*valid.shape[:-1], 1), valid], dim=-1)

        # nn.TransformerEncoder only accepts a single leading batch dim;
        # flatten any extra ones (torchrl rollouts are often (*batch, feature)
        # with batch itself multi-dimensional, e.g. (time, env)) and restore
        # them after pooling out the token dimension.
        batch_shape = tokens.shape[:-2]
        flat_tokens = tokens.reshape(-1, tokens.shape[-2], tokens.shape[-1])
        flat_valid = valid.reshape(-1, valid.shape[-1])
        # Only the per-entity path introduces padding; without it every token
        # is real, and passing an all-False mask would push the encoder down
        # the masked attention path for nothing. Decided from config rather
        # than from `flat_valid.all()`, which would force a device sync.
        # Where the mask does apply, the per-group tokens are always valid, so
        # no row is fully padded and the attention softmax cannot yield NaN.
        padding_mask = ~flat_valid if self.token_groups else None
        encoded = self.encoder(flat_tokens, src_key_padding_mask=padding_mask)

        if self.pooling == "cls":
            pooled = encoded[:, 0, :]
        elif self.pooling == "attention":
            scores = (encoded @ self.pool_query) / (self.out_features ** 0.5)
            scores = scores.masked_fill(~flat_valid, float("-inf"))
            pooled = (scores.softmax(dim=-1).unsqueeze(-1) * encoded).sum(dim=-2)
        else:
            weights = flat_valid.to(encoded.dtype).unsqueeze(-1)
            pooled = (encoded * weights).sum(dim=-2) / weights.sum(dim=-2).clamp(min=1.0)
        state_repr = pooled.reshape(*batch_shape, self.out_features)

        if not self.option_tokens:
            return state_repr
        option_rows, _ = entity_tokens["options"]
        option_repr = self.entity_projections["options"](option_rows)
        return state_repr, option_repr
