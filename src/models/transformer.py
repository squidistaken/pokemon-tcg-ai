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
#: Groups with no entity axis: already exactly one token via
#: :meth:`~src.models.structured_obs_adapter.StructuredObsAdapter.encode_groups`,
#: so they cannot be expanded into per-entity tokens.
_SCALAR_GROUPS = {"globals", "select_cats"}


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
    :class:`~src.models.heads.PointerPolicyHead` scores individually. By
    default (``encoded_option_repr=False``) that costs one projection per
    option rather than quadratic attention, so it is much cheaper than
    putting ``options`` in ``token_groups`` — and it is what lets the policy
    distinguish *which* option is which, rather than choosing among action
    slots from a pooled state alone. Set ``encoded_option_repr=True`` (with
    ``options`` also in ``token_groups``) to score the *attended* option rows
    instead, once the extra attention cost is one you want to pay for.

    **Entity identity.** Every per-entity token also gets a learned
    embedding for its
    :attr:`~src.models.structured_obs_adapter.StructuredObsAdapter.group_segment_ids`
    — which seat a Pokemon belongs to, which zone a card sits in, whether an
    option slot is the synthetic stop action — added on top of the group's
    type embedding. Without it, attention and pooling over per-entity tokens
    are permutation-*equivariant*/-*invariant* to that identity, so e.g.
    swapping the two players' boards would leave ``state_repr`` unchanged.

    **Replacing pooled tokens.** ``replace_pooled=True`` drops, for every
    name in ``token_groups``, that group's single pooled ``encode_groups``
    token (and its ``token_projections`` entry / ``token_type_embedding``
    row) from the sequence, so the per-entity tokens are the *only* route by
    which that group reaches the trunk rather than an addition to a pooled
    summary that already saw it.

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
        replace_pooled: bool = False,
        encoded_option_repr: bool = False,
        in_keys: list[str] | None = None,
    ) -> None:
        """
        :param input_dim: Summed width of the adapter's per-group vectors;
            must equal ``adapter.out_features``. Unused beyond that check,
            which only bites on direct construction:
            :func:`~src.policies.ppo_actor.build_actor_critic` derives the
            argument from the adapter it just built, so the two agree by
            construction on the configured path. Kept because a hand-built
            backbone paired with the wrong adapter is worth catching here
            rather than as a shape error inside the first forward.
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
            ``adapter.group_slot_counts[name]`` extra tokens each. Must be
            names the adapter registered with an entity axis (not
            ``globals``/``select_cats``).
        :param option_tokens: Emit per-option tokens as ``option_repr`` for a
            pointer head. Sets :attr:`produces_option_repr`.
        :param replace_pooled: If True, drop the pooled ``encode_groups``
            token (and its ``token_projections``/``token_type_embedding``
            entry) for every name in ``token_groups``, so the per-entity
            tokens replace rather than duplicate that group's pooled summary.
            False (the default) keeps every existing checkpoint's
            ``token_projections``/``token_type_embedding`` indices unchanged.
        :param encoded_option_repr: If True, read ``option_repr`` from the
            encoder's *output* rows at the ``options`` tokens' offsets
            instead of the cheap pre-attention projection. Requires
            ``option_tokens=True`` and ``"options" in token_groups`` — the
            offsets only exist once ``options`` tokens actually enter the
            sequence.
        :param in_keys: Observation keys to consume; one token per key, in
            the same order as ``adapter.group_feature_widths``.
        :raises ValueError: If ``adapter`` is missing, its width disagrees
            with ``input_dim``, ``out_features`` isn't divisible by
            ``num_heads``, ``activation``/``pooling`` is unsupported,
            ``token_groups`` names an unregistered or scalar group, the
            ``options``/``option_tokens``/``encoded_option_repr`` combination
            wastes its own attention pass, or the resulting configuration
            could leave a row with zero valid tokens (fully-padded rows make
            :class:`nn.TransformerEncoder` emit NaN).
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
        self.encoded_option_repr = bool(encoded_option_repr)
        self.replace_pooled = bool(replace_pooled)

        group_names = cast(list[str], adapter.group_names)
        group_widths = cast(Sequence[int], adapter.group_feature_widths)
        group_slot_counts = cast(dict[str, int], adapter.group_slot_counts)

        # Eager (construction-time, not first-forward) validation: a typo or
        # an incoherent flag combination should fail before env spin-up and
        # W&B run creation, not on the first batch (finding 4).
        for name in self.token_groups:
            if name not in group_slot_counts:
                raise ValueError(
                    f"'{name}' is not among the adapter's registered groups "
                    f"{sorted(group_slot_counts)}; check token_groups for a typo."
                )
            if name in _SCALAR_GROUPS:
                raise ValueError(
                    f"'{name}' has no entity axis -- it is already exactly one token via "
                    "encode_groups(); remove it from token_groups."
                )
        if len(set(self.token_groups)) != len(self.token_groups):
            # A repeat would concatenate that group's entity block into the
            # sequence twice: the readout would double-count it, and
            # `entity_offsets` (which `encoded_option_repr` slices at) records
            # only the last occurrence. Both are silent, so reject the config.
            duplicates = sorted(
                {
                    name
                    for name in self.token_groups
                    if self.token_groups.count(name) > 1
                }
            )
            raise ValueError(
                f"token_groups repeats {duplicates}; each group can only be expanded once. "
                "A repeat would put that group's entity tokens into the sequence twice and "
                "double its weight in the readout."
            )
        if self.option_tokens and "options" not in group_slot_counts:
            raise ValueError(
                "option_tokens=True needs an 'options' group registered on the adapter to "
                "encode per-option tokens from; this adapter registered only "
                f"{sorted(group_slot_counts)}."
            )
        if (
            "options" in self.token_groups
            and self.option_tokens
            and not self.encoded_option_repr
        ):
            raise ValueError(
                "token_groups=[..., 'options', ...] with option_tokens=True routes every "
                "option row through the encoder's quadratic attention and then discards the "
                "result, because option_repr would still read the pre-attention projection. "
                "Set encoded_option_repr=True to use the attended rows instead, or drop "
                "'options' from token_groups to keep the cheap projection."
            )
        if self.encoded_option_repr and not (
            self.option_tokens and "options" in self.token_groups
        ):
            raise ValueError(
                "encoded_option_repr=True slices option_repr out of the encoder's output at "
                "the 'options' tokens' offset, so it requires option_tokens=True and "
                "'options' in token_groups to put those tokens there."
            )

        # `replace_pooled` drops, for each name in token_groups, the pooled
        # encode_groups() token (and its token_projections entry /
        # token_type_embedding row) from the sequence -- see the class
        # docstring. With replace_pooled=False (the default) dropped_pooled is
        # empty, so pooled_group_names/token_projections/token_type_embedding
        # are byte-identical to before this flag existed: an existing
        # checkpoint's backbone.token_projections.0..9 indices keep meaning
        # what they meant, because only the *shorter*-list arms are new.
        dropped_pooled = (
            frozenset(self.token_groups) if self.replace_pooled else frozenset()
        )
        self._dropped_pooled_groups = dropped_pooled
        #: Names of the groups still contributing one pooled ``encode_groups``
        #: token, in :attr:`token_projections`/:attr:`token_type_embedding`
        #: order. Equal to every registered group when :attr:`replace_pooled`
        #: is False.
        self.pooled_group_names: list[str] = [
            name for name in group_names if name not in dropped_pooled
        ]
        self.token_projections = nn.ModuleList(
            [
                nn.Linear(width, out_features)
                for name, width in zip(group_names, group_widths, strict=True)
                if name not in dropped_pooled
            ]
        )
        # Learned per-group identity: attention over these tokens would
        # otherwise be permutation-invariant to which group is which.
        self.token_type_embedding = nn.Parameter(
            torch.zeros(len(self.pooled_group_names), out_features)
        )
        nn.init.normal_(self.token_type_embedding, std=0.02)

        # A row with zero valid tokens makes nn.TransformerEncoder itself
        # emit NaN (verified directly against torch): mean pooling's
        # clamp(min=1.0) cannot rescue it because the encoder's output is
        # already NaN by then, and attention pooling's softmax over all -inf
        # NaNs too. Without replace_pooled this cannot happen, because every
        # pooled token is unconditionally valid (see _group_tokens). Once
        # replace_pooled can drop every pooled token, the only remaining
        # backstop is the 'options' entity group's always-valid stop slot
        # (StructuredObsAdapter._option_validity) -- if neither is present,
        # reject the configuration now rather than NaN on some future batch
        # that happens to pad out every requested entity group at once.
        if not self.pooled_group_names and "options" not in self.token_groups:
            raise ValueError(
                "This configuration can leave a row with zero valid tokens: replace_pooled "
                f"drops every pooled group's token ({', '.join(group_names)}) and 'options' "
                "is not in token_groups to backstop it with its always-valid stop slot. "
                "nn.TransformerEncoder emits NaN for a fully padded row. Keep at least one "
                "group out of token_groups, or include 'options' in token_groups."
            )

        # Per-entity token path. Entities arrive at the adapter's entity_dim
        # regardless of group, so one projection per group suffices; the
        # accompanying type embedding keeps "an option" distinguishable from
        # "a Pokemon" after they are concatenated into one sequence.
        entity_dim = adapter.entity_dim
        self._needs_entity_tokens = sorted(
            set(self.token_groups) | ({"options"} if self.option_tokens else set())
        )
        if self._needs_entity_tokens and entity_dim is None:
            raise ValueError(
                "token_groups/option_tokens need per-entity encodings, which require the "
                "adapter to be built with entity_dim set (the default)."
            )
        self.entity_projections = nn.ModuleDict(
            {
                name: nn.Linear(int(cast(int, entity_dim)), out_features)
                for name in self._needs_entity_tokens
            }
        )
        self.entity_type_embedding = nn.ParameterDict(
            {
                name: nn.Parameter(torch.zeros(out_features))
                for name in self._needs_entity_tokens
            }
        )
        for parameter in self.entity_type_embedding.values():
            nn.init.normal_(parameter, std=0.02)
        # One learned vector per segment id (StructuredObsAdapter.
        # group_segment_ids), added to that slot's projected token: the stop
        # slot's own vector (finding 2b) and the seat/zone identity the
        # pooled and (pre-this-change) entity paths could not express
        # (finding 1), without reintroducing a per-slot-index preference,
        # since every *real* slot within a group still shares one vector.
        adapter_segment_ids = cast(dict[str, torch.Tensor], adapter.group_segment_ids)
        self.entity_segment_embedding = nn.ParameterDict(
            {
                name: nn.Parameter(
                    torch.zeros(
                        int(adapter_segment_ids[name].max().item()) + 1, out_features
                    )
                )
                for name in self._needs_entity_tokens
            }
        )
        for parameter in self.entity_segment_embedding.values():
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
        One projected token per surviving pooled observation group, all of
        them unconditionally valid.

        ``encode_groups`` always computes every registered group's pooled
        vector in one call; :attr:`replace_pooled` only changes which of
        those vectors get projected and concatenated here, not whether the
        adapter computes them.

        :param inputs: The backbone's forward inputs.
        :return: ``(tokens, valid)`` shaped ``(*batch, n_pooled_groups,
            d_model)`` and ``(*batch, n_pooled_groups)``.
        """
        group_names = cast(list[str], self.adapter.group_names)
        group_vectors = self.adapter.encode_groups(*inputs)
        kept_vectors = [
            vector
            for name, vector in zip(group_names, group_vectors, strict=True)
            if name not in self._dropped_pooled_groups
        ]
        tokens = torch.stack(
            [
                proj(vector)
                for proj, vector in zip(
                    self.token_projections, kept_vectors, strict=True
                )
            ],
            dim=-2,
        )
        tokens = tokens + self.token_type_embedding
        valid = tokens.new_ones(tokens.shape[:-1], dtype=torch.bool)
        return tokens, valid

    def _project_entity_tokens(self, name: str, tokens: torch.Tensor) -> torch.Tensor:
        """
        Project one group's raw per-entity encodings and add its identity.

        Shared by the tokens entering the encoder (:attr:`token_groups`) and
        the cheap ``option_repr`` path (:attr:`option_tokens` with
        ``encoded_option_repr=False``), so both see the same type and
        segment identity rather than the trunk seeing an enriched
        representation while the pointer head sees a bare projection.

        :param name: Entity group name; a key of :attr:`entity_projections`,
            :attr:`entity_type_embedding` and :attr:`entity_segment_embedding`.
        :param tokens: Raw ``(*batch, n_slots, entity_dim)`` per-entity
            encodings from
            :meth:`~src.models.structured_obs_adapter.StructuredObsAdapter.encode_entity_tokens`.
        :return: ``(*batch, n_slots, out_features)`` tokens carrying the
            group's learned type identity and each slot's segment identity.
        """
        segment_ids = cast(dict[str, torch.Tensor], self.adapter.group_segment_ids)[
            name
        ]
        return (
            self.entity_projections[name](tokens)
            + self.entity_type_embedding[name]
            + self.entity_segment_embedding[name][segment_ids]
        )

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
        # Offsets recorded while assembling the sequence, not recomputed from
        # a static formula afterward: `pooling="cls"` prepends its token
        # *after* this loop (shifting every later index by one, handled
        # below) and `replace_pooled` changes how many pooled tokens precede
        # the entity tokens, so a formula would have to duplicate this
        # bookkeeping and could silently drift from it.
        entity_offsets: dict[str, int] = {}
        for name in self.token_groups:
            group_tokens, group_valid = entity_tokens[name]
            entity_offsets[name] = tokens.shape[-2]
            projected = self._project_entity_tokens(name, group_tokens)
            tokens = torch.cat([tokens, projected], dim=-2)
            valid = torch.cat([valid, group_valid], dim=-1)

        if self.pooling == "cls":
            cls = self.cls_token.expand(*tokens.shape[:-2], 1, tokens.shape[-1])
            tokens = torch.cat([cls, tokens], dim=-2)
            valid = torch.cat([valid.new_ones(*valid.shape[:-1], 1), valid], dim=-1)
            entity_offsets = {
                name: offset + 1 for name, offset in entity_offsets.items()
            }

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
        # A fully padded row would make the encoder itself emit NaN (mean/
        # attention pooling cannot rescue it, since its input is already
        # NaN); __init__ asserts that some group's token is unconditionally
        # valid whenever the mask is in play (a surviving pooled token, or
        # 'options' with its always-valid stop slot), so that cannot happen
        # here regardless of what the current batch's entities look like.
        padding_mask = ~flat_valid if self.token_groups else None
        encoded = self.encoder(flat_tokens, src_key_padding_mask=padding_mask)

        if self.pooling == "cls":
            pooled = encoded[:, 0, :]
        elif self.pooling == "attention":
            scores = (encoded @ self.pool_query) / (self.out_features**0.5)
            scores = scores.masked_fill(~flat_valid, float("-inf"))
            pooled = (scores.softmax(dim=-1).unsqueeze(-1) * encoded).sum(dim=-2)
        else:
            weights = flat_valid.to(encoded.dtype).unsqueeze(-1)
            pooled = (encoded * weights).sum(dim=-2) / weights.sum(dim=-2).clamp(
                min=1.0
            )
        state_repr = pooled.reshape(*batch_shape, self.out_features)

        if not self.option_tokens:
            return state_repr
        if self.encoded_option_repr:
            n_options = entity_tokens["options"][0].shape[-2]
            start = entity_offsets["options"]
            option_repr = encoded[:, start : start + n_options, :].reshape(
                *batch_shape, n_options, self.out_features
            )
        else:
            option_rows, _ = entity_tokens["options"]
            option_repr = self._project_entity_tokens("options", option_rows)
        return state_repr, option_repr
