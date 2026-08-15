from collections.abc import Sequence

import torch
from tensordict import TensorDictBase
from torch import nn
from torchrl.data import Composite, TensorSpec

from cg.api import (
    AreaType,
    CardType,
    EnergyType,
    OptionType,
    SelectContext,
    SelectType,
    SpecialConditionType,
)
from src.env.observation.card_database import CardDatabase

_CATEGORY_FIELD_ENUMS = {
    "select_cats": (SelectType, SelectContext),
    "options.cats": (OptionType, AreaType, AreaType, SpecialConditionType),
    "card_cats": (CardType, EnergyType, EnergyType, EnergyType),
}


class StructuredObsAdapter(nn.Module):
    """
    Encodes the structured observation for an MLP or Transformer backbone.

    Card IDs, table rows (options, pokemon) and scalars each get their own
    per-entity encoder. Two paths consume the result:

    * :meth:`encode_entity_tokens` returns one token per entity plus a validity
      mask, for a trunk that runs attention over them.
    * ``pool=True`` collapses every group by masked set pooling into one flat
      vector, for an MLP trunk.

    Both read the same encoders, so the token path costs no extra parameters.
    ``pokemon_seat_split`` makes the pooled path seat-aware, and
    ``emit_option_tokens`` hands the per-slot option table up for a pointer
    head. Shapes, group tables and the reasoning behind each choice are in
    ``docs/architecture/structured-obs-adapter.md``.
    """

    #: Supported set-pooling modes for zone/board groups (see ``zone_pooling``).
    POOL_MODES = ("mean", "mean_max_sum")

    #: Rows reserved per categorical field in the shared category table.
    CATEGORY_VOCAB_SIZE = 64
    #: Field order in the shared category table: selection type, selection
    #: context, then the option table's four category columns, then the
    #: static per-card category columns.
    SELECT_CATEGORY_FIELD_COUNT = 2
    OPTION_CATEGORY_FIELD_COUNT = 4
    #: ``CardDatabase.card_cats`` columns: card type, energy type, weakness,
    #: resistance.
    CARD_CATEGORY_FIELD_COUNT = 4
    #: ``options.owner`` values: 0 = none, 1 = agent, 2 = opponent.
    OWNER_VALUE_COUNT = 3
    #: Scale for the option scalar columns (zone indices/counts, cap 60).
    OPTION_SCALAR_SCALE = 60.0

    #: Per-column scales for ``options.target_state`` (see
    #: ``StructuredObservationEncoder.OPTION_TARGET_FEATURE_COUNT``): resolved
    #: flag, hp, maxHp, hp fraction, energy count, tool count, is-active. Its
    #: own scales rather than ``OPTION_SCALAR_SCALE``, because HP runs to ~400
    #: while the flags are already 0/1 and dividing those by 60 would bury them.
    OPTION_TARGET_SCALES = (1.0, 400.0, 400.0, 1.0, 16.0, 2.0, 1.0)

    #: Per-index scales for the encoder's ``globals`` layout (see
    #: ``StructuredObservationEncoder._encode_globals``): 8 game entries,
    #: 9 selection entries, then 12 per player (agent first).
    GAME_SCALES = (50.0, 20.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
    SELECT_SCALES = (1.0, 6.0, 6.0, 96.0, 6.0, 30.0, 5.0, 1.0, 1.0)
    PLAYER_SCALES = (60.0, 15.0, 6.0, 8.0, 8.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)

    #: Per-column scales for ``pokemon.features``: hp, maxHp, hp fraction,
    #: appeared-this-turn, is-active, tool count, energy count, pre-evolution
    #: count, then the 12-type provided-energy histogram.
    POKEMON_FEATURE_SCALES = (400.0, 400.0, 1.0, 1.0, 1.0, 2.0, 16.0, 2.0) + (4.0,) * 12

    #: Declared types for the ``register_buffer`` attributes set in
    #: :meth:`__init__`; ``nn.Module``'s dynamic ``__getattr__`` otherwise
    #: types them as ``Tensor | Module``, which drops shape/operator info.
    _card_static: torch.Tensor
    _attack_static: torch.Tensor
    _card_categories: torch.Tensor
    _card_attack_ids: torch.Tensor
    _select_category_offsets: torch.Tensor
    _option_category_offsets: torch.Tensor
    _card_category_offsets: torch.Tensor
    _global_scales: torch.Tensor
    _pokemon_feature_scales: torch.Tensor
    _option_target_scales: torch.Tensor

    def __init__(
        self,
        obs_spec: Composite,
        in_keys: list[str | tuple[str, ...]],
        card_embed_dim: int = 8,
        attack_embed_dim: int = 8,
        category_embed_dim: int = 4,
        card_database: CardDatabase | None = None,
        entity_dim: int | None = 64,
        pool: bool = True,
        emit_option_tokens: bool = False,
        zone_pooling: str = "mean",
        pokemon_seat_split: bool = False,
        option_target_state: bool = False,
        card_effect_features: bool = False,
    ) -> None:
        """
        :param obs_spec: Full environment observation spec.
        :param in_keys: Observation keys the backbone consumes, in order.
        :param card_embed_dim: Width of the learned per-card embedding.
        :param attack_embed_dim: Width of the learned per-attack embedding.
        :param category_embed_dim: Width of the shared categorical embedding.
        :param card_database: Static card/attack lookup tables.
        :param entity_dim: If set, each option row, Pokémon row, and zone
            card is projected to this width before masked-mean pooling, so
            the output size is independent of the padded table widths.
            ``None`` reverts to the legacy flat-padded behaviour.
        :param pool: If True (the default), masked-mean pool the per-entity
            encodings so each group contributes one fixed vector. If False,
            revert to the legacy flat-padded encoding, where every padded slot
            consumes input width (13,297 features against 824) — this is *not*
            the transformer token path, which is :meth:`encode_entity_tokens`
            and works with pooling left on. Ignored when ``entity_dim`` is None.
        :param emit_option_tokens: If True, :meth:`forward` additionally
            returns the per-option token table, so a backbone that does not
            build tokens of its own can still feed a pointer head. This is the
            same table :meth:`encode_entity_tokens` returns for ``options``,
            handed up unprojected; without it the option group is only ever
            seen through its pooled mean, which is permutation-invariant and
            so carries no slot-to-option correspondence at all.
        :param zone_pooling: How unordered card sets (zones and the Pokémon
            board) are collapsed. ``"mean"`` is the centroid alone;
            ``"mean_max_sum"`` concatenates the mean, the element-wise max
            and a capacity-normalized sum, which lets the trunk answer "is
            card X present" and "how many" rather than only "what is the
            average card here".
        :param pokemon_seat_split: Pool the two seats' board rows separately
            and concatenate, instead of pooling all
            ``group_slot_counts["pokemon"]`` rows into one summary. The
            ``pokemon`` table is the only group holding both players' rows,
            so a single pool over it is seat-blind: swapping the two boards
            leaves the group vector — and every backbone's ``state_repr``
            built from it — bit-identical, measured at ``0.0``. That matters
            most for the critic, which sees only ``state_repr`` and is asked
            to predict win probability from a board summary that cannot say
            whose Pokémon is whose. Doubles this group's feature width
            (``2 × _pool_width(entity_dim)``) and costs nothing at inference
            beyond one extra pool. Defaults False so every checkpoint
            predating this argument keeps loading against the width it was
            trained with; the seat identity for the *token* path is separate
            and always on (see :attr:`group_segment_ids`).
        :param option_target_state: Consume ``options.target_state`` -- the live
            state (HP, attached energy/tool counts, is-active) of the in-play
            Pokemon each option acts on. ``target_id`` already resolves *which
            card* is targeted, but a card ID is shared by every copy, so
            without this two options over two copies of the same Pokemon differ
            only in a raw index scalar the network cannot dereference. Widens
            an option row by ``OPTION_TARGET_FEATURE_COUNT``.

            Defaults False because this width comes from the *observation
            spec*, not from a checkpoint's config: a snapshot trained before
            the ``target_state`` block exists must still rebuild at its
            original width to be usable as an evaluation opponent, and the
            deployed Kaggle agent must keep running. New runs opt in.
        :param card_effect_features: Build the default
            :class:`~src.env.observation.card_database.CardDatabase` with the
            parsed card-text columns, so a card reaches the network as what it
            does rather than only as a learned index. Ignored when
            ``card_database`` is supplied. Widens the card representation, so a
            checkpoint trained with it set cannot be rebuilt with it clear.
        :raises ValueError: If ``zone_pooling`` is not a supported mode, or
            ``emit_option_tokens`` is set without an ``options`` group.
        """
        super().__init__()
        if zone_pooling not in self.POOL_MODES:
            raise ValueError(
                f"Unknown zone_pooling '{zone_pooling}'; expected one of {list(self.POOL_MODES)}."
            )
        self._zone_pooling = zone_pooling
        self._check_category_vocab()
        self._emit_option_tokens = bool(emit_option_tokens)
        self._pokemon_seat_split = bool(pokemon_seat_split)
        self._option_target_state = bool(option_target_state)
        database = (
            card_database
            if card_database is not None
            else CardDatabase(effect_features=bool(card_effect_features))
        )
        card_static = database.card_features
        attack_static = database.attack_features
        card_categories = database.card_cats
        card_attack_ids = database.card_attack_ids
        self.register_buffer(
            "_card_static", card_static / card_static.abs().amax(dim=0).clamp(min=1.0)
        )
        self.register_buffer(
            "_attack_static",
            attack_static / attack_static.abs().amax(dim=0).clamp(min=1.0),
        )
        self.register_buffer("_card_categories", card_categories)
        self.register_buffer("_card_attack_ids", card_attack_ids)
        self._attack_repr_dim = attack_embed_dim + attack_static.shape[1]
        self._card_repr_dim = (
            card_embed_dim
            + card_static.shape[1]
            + self.CARD_CATEGORY_FIELD_COUNT * category_embed_dim
            + self._attack_repr_dim
        )
        self._category_embed_dim: int = category_embed_dim
        self._entity_dim: int | None = entity_dim
        self._pool: bool = pool and entity_dim is not None

        self._card_embedding = nn.Embedding(
            card_static.shape[0], card_embed_dim, padding_idx=0
        )
        self._attack_embedding = nn.Embedding(
            attack_static.shape[0], attack_embed_dim, padding_idx=0
        )
        field_count = (
            self.SELECT_CATEGORY_FIELD_COUNT
            + self.OPTION_CATEGORY_FIELD_COUNT
            + self.CARD_CATEGORY_FIELD_COUNT
        )
        self._category_embedding = nn.Embedding(
            field_count * self.CATEGORY_VOCAB_SIZE, category_embed_dim
        )
        self.register_buffer(
            "_select_category_offsets",
            torch.arange(self.SELECT_CATEGORY_FIELD_COUNT, dtype=torch.int64)
            * self.CATEGORY_VOCAB_SIZE,
        )
        self.register_buffer(
            "_option_category_offsets",
            (
                torch.arange(self.OPTION_CATEGORY_FIELD_COUNT, dtype=torch.int64)
                + self.SELECT_CATEGORY_FIELD_COUNT
            )
            * self.CATEGORY_VOCAB_SIZE,
        )
        self.register_buffer(
            "_card_category_offsets",
            (
                torch.arange(self.CARD_CATEGORY_FIELD_COUNT, dtype=torch.int64)
                + self.SELECT_CATEGORY_FIELD_COUNT
                + self.OPTION_CATEGORY_FIELD_COUNT
            )
            * self.CATEGORY_VOCAB_SIZE,
        )
        global_scales = self.GAME_SCALES + self.SELECT_SCALES + 2 * self.PLAYER_SCALES
        self.register_buffer(
            "_global_scales", torch.tensor(global_scales, dtype=torch.float32)
        )
        self.register_buffer(
            "_pokemon_feature_scales",
            torch.tensor(self.POKEMON_FEATURE_SCALES, dtype=torch.float32),
        )
        # Non-persistent: a constant, rederived identically on every
        # construction. A persistent buffer would appear as an unexpected key
        # when loading a checkpoint that predates it, and as a missing one when
        # this class loads such a checkpoint -- the same reasoning as
        # _register_segment_ids.
        self.register_buffer(
            "_option_target_scales",
            torch.tensor(self.OPTION_TARGET_SCALES, dtype=torch.float32),
            persistent=False,
        )

        # Per-entity projection layers.  Built in __init__ or _register_group.
        # Typed as nn.Module (not nn.Linear) because the non-pooling path
        # uses Identity placeholders that still satisfy the callable interface.
        if entity_dim is not None:
            self._card_proj: nn.Module = nn.Linear(self._card_repr_dim, entity_dim)
            self._option_encoder: nn.Module = nn.Identity()
            self._pokemon_encoder: nn.Module = nn.Identity()
        else:
            self._card_proj = nn.Identity()
            self._option_encoder = nn.Identity()
            self._pokemon_encoder = nn.Identity()

        self._group_names: list[str] = []
        self._zone_pairs: dict[str, list[tuple[str, str]]] = {}
        #: Names of the groups registered with a segment id buffer (every
        #: group except ``globals``/``select_cats``, which have no entity
        #: axis), in registration order. Backs the :attr:`group_segment_ids`
        #: property; not itself public because it is only ever read through
        #: that property (which re-fetches each buffer via ``getattr`` so
        #: moving the module with ``.to(device)`` is reflected immediately).
        self._segment_group_names: list[str] = []
        self.out_features = 0
        #: Feature width each group contributes, in :attr:`_group_names` order.
        #: Lets a token-per-group backbone (e.g. a transformer trunk) size its
        #: per-group input projections without re-deriving the arithmetic in
        #: :meth:`_register_group`.
        self.group_feature_widths: list[int] = []
        #: Padded per-entity slot count of each group, so a token-sequence
        #: backbone can size its embeddings and budget its attention cost
        #: without re-reading the observation spec. For the groups
        #: :meth:`encode_entity_tokens` accepts, this is how many tokens it
        #: emits. ``globals``/``select_cats`` are recorded as ``1``, their
        #: contribution to :meth:`encode_groups`, but have no entity axis:
        #: enumerate :attr:`group_segment_ids` instead to ask which groups can
        #: become per-entity tokens.
        self.group_slot_counts: dict[str, int] = {}
        for key in in_keys:
            name = key[-1] if isinstance(key, tuple) else key
            self._group_names.append(name)
            width = self._register_group(name, obs_spec[key])
            self.group_feature_widths.append(width)
            self.out_features += width
        if self._emit_option_tokens and "options" not in self._group_names:
            raise ValueError(
                "emit_option_tokens=True requires the 'options' group in in_keys; "
                f"got {self._group_names}."
            )

    @classmethod
    def _check_category_vocab(cls) -> None:
        """
        Fail loudly if an engine enum has outgrown its slice of the category table.

        :raises ValueError: If any field's largest encoded value does not fit.
        """
        for field_group, enums in _CATEGORY_FIELD_ENUMS.items():
            for enum in enums:
                needed = max(int(value) for value in enum) + 2
                if needed > cls.CATEGORY_VOCAB_SIZE:
                    raise ValueError(
                        f"{enum.__name__} (in {field_group}) needs {needed} category "
                        f"rows but CATEGORY_VOCAB_SIZE is {cls.CATEGORY_VOCAB_SIZE}. "
                        f"Raise it -- leaving it would silently alias this field onto "
                        f"the next one's embedding rows instead of raising."
                    )

    @property
    def emits_option_tokens(self) -> bool:
        """
        Whether :meth:`forward` returns per-option tokens alongside the state vector.

        Backbones read this to decide whether option tokens arrive from here or
        have to be built from :meth:`encode_entity_tokens`; see
        :class:`~src.models.mlp.MLPBackbone`, which supports both.

        :return: True when the adapter was built with ``emit_option_tokens``.
        """
        return self._emit_option_tokens

    @property
    def option_token_dim(self) -> int:
        """
        Width of one per-option token, for sizing a pointer head.

        :return: The token width emitted for each option slot.
        :raises ValueError: If the adapter does not emit option tokens.
        """
        if not self._emit_option_tokens:
            raise ValueError("This adapter does not emit option tokens.")
        assert self._entity_dim is not None
        return self._entity_dim

    def _pool_width(self, dim: int) -> int:
        """
        Output width of :meth:`_masked_pool` for per-entity encodings of ``dim``.

        :param dim: Width of a single entity encoding.
        :return: Width after set pooling under the configured mode.
        """
        return dim if self._zone_pooling == "mean" else 3 * dim

    @property
    def entity_dim(self) -> int | None:
        """
        Width of one per-entity encoding, or ``None`` in legacy flat mode.

        Public accessor for ``__init__``'s ``entity_dim`` argument, so a
        backbone can size its own per-entity projections without reaching
        into the adapter's internals.

        :return: The configured ``entity_dim``, or ``None`` if the adapter
            was built with ``entity_dim=None`` (no per-entity encoders).
        """
        return self._entity_dim

    @property
    def group_names(self) -> list[str]:
        """
        Registered group names, in ``in_keys`` order.

        A copy, so callers cannot mutate the adapter's internal list; aligned
        with :attr:`group_feature_widths` and the values :meth:`encode_groups`
        returns.

        :return: One name per registered group.
        """
        return list(self._group_names)

    @property
    def group_segment_ids(self) -> dict[str, torch.Tensor]:
        """
        Per-slot segment id for every registered group with an entity axis.

        A segment groups slots that share an identity a backbone should be
        able to tell apart when it expands a group into per-entity tokens
        (:meth:`encode_entity_tokens`): which seat a Pokemon belongs to,
        which zone a card sits in, whether an option slot is real or the
        synthetic stop action. Segments rather than a full per-slot table, so
        permutation invariance is preserved *within* a segment (bench order
        genuinely doesn't matter) while the segments themselves stay
        distinguishable — a backbone adds a learned embedding indexed by this
        id to each slot's token.

        Rebuilt from the registered buffers on every access (rather than
        cached in a plain dict at construction time) so a tensor moved by
        ``.to(device)`` is always the one returned — a plain dict of buffer
        references would otherwise go stale, since ``.to`` replaces the
        module's buffer entries in place without updating external
        references to the old tensors.

        ``globals``/``select_cats`` have no entity axis and are absent here,
        unlike :attr:`group_slot_counts`, which records ``1`` for them.

        :return: ``{name: segment_ids}``, ``segment_ids`` an int64 tensor of
            shape ``(group_slot_counts[name],)``.
        """
        return {
            name: getattr(self, f"_segment_ids_{name}")
            for name in self._segment_group_names
        }

    def _register_segment_ids(self, name: str, segment_ids: torch.Tensor) -> None:
        """
        Register a group's per-slot segment ids as a non-persistent buffer.

        A buffer (not a plain attribute) so it moves with the module via
        ``.to(device)`` the same way the learned parameters do; see
        :attr:`group_segment_ids` for why it is exposed through a property
        rather than cached in a dict directly. Non-persistent (excluded from
        ``state_dict()``) because it carries no trained information — it is
        rederived identically from ``obs_spec`` on every construction — and a
        persistent buffer would otherwise appear as an unexpected key when
        loading a pre-segment-ids checkpoint's adapter weights into this
        class, or when loading this class's checkpoint into
        ``submission/runtime.py``'s hand-duplicated adapter, which has no
        matching buffer (and, being Torch-only with no ``obs_spec``, cannot
        rederive one).

        :param name: Group name, becomes a key of :attr:`group_segment_ids`.
        :param segment_ids: Int64 tensor of shape
            ``(group_slot_counts[name],)``.
        """
        self._segment_group_names.append(name)
        self.register_buffer(f"_segment_ids_{name}", segment_ids, persistent=False)

    def _option_row_width(self, spec: Composite) -> int:
        """Flattened width of one option row (card+target+attack+cats+scalars+target state)."""
        return (
            2 * self._card_repr_dim
            + self._attack_repr_dim
            + self.OWNER_VALUE_COUNT
            + self.OPTION_CATEGORY_FIELD_COUNT * self._category_embed_dim
            + spec["scalars"].shape[-1]
            + (spec["target_state"].shape[-1] if self._option_target_state else 0)
        )

    def _pokemon_row_width(self, spec: Composite) -> int:
        """Flattened width of one pokemon table row (after internal pooling)."""
        return (
            self._card_repr_dim
            + self._card_repr_dim
            + self._card_repr_dim
            + self._card_repr_dim
            + spec["features"].shape[-1]
            + 1
        )

    def _register_group(self, name: str, spec: TensorSpec | Composite) -> int:
        """
        Return the feature width this group contributes to the output.

        With ``_pool`` active, table groups contribute ``entity_dim``
        (one pooled vector) instead of ``n_slots × row_width``, and card
        groups contribute ``entity_dim + 1`` per zone. Every group but
        ``globals``/``select_cats`` (no entity axis) also gets a segment id
        buffer registered here (see :attr:`group_segment_ids`), beside
        :attr:`group_slot_counts` for the same reason: both describe the
        shape of what :meth:`encode_entity_tokens` emits for this group.
        """
        if name == "globals":
            self.group_slot_counts[name] = 1
            return self._global_scales.shape[0]
        if name == "select_cats":
            self.group_slot_counts[name] = 1
            return self.SELECT_CATEGORY_FIELD_COUNT * self._category_embed_dim
        if name == "context_card_ids":
            width = self._entity_dim if self._pool else self._card_repr_dim
            assert width is not None
            n_slots = spec.shape[-1]
            self.group_slot_counts[name] = n_slots
            # One id per slot: contextCard (slot 0) and effect (slot 1) are
            # different kinds of reference, not interchangeable entities, so
            # they must not share a segment the way e.g. bench slots do.
            self._register_segment_ids(name, torch.arange(n_slots, dtype=torch.int64))
            return n_slots * width
        if name == "stadium_id":
            width = self._entity_dim if self._pool else self._card_repr_dim
            assert width is not None
            self.group_slot_counts[name] = 1
            self._register_segment_ids(name, torch.zeros(1, dtype=torch.int64))
            return width
        if name == "options":
            n_slots = spec["card_id"].shape[-1]
            self.group_slot_counts[name] = n_slots
            # 0 for the real option slots, 1 for the final synthetic stop
            # slot: the stop action is not "an option like the others" (it
            # has no card/target/attack), so it earns its own identity
            # rather than sharing the real slots' segment (finding 2b).
            option_segment_ids = torch.zeros(n_slots, dtype=torch.int64)
            option_segment_ids[-1] = 1
            self._register_segment_ids(name, option_segment_ids)
            if self._pool:
                assert self._entity_dim is not None
                self._option_encoder = nn.Linear(
                    self._option_row_width(spec), self._entity_dim
                )
                return self._entity_dim
            return n_slots * self._option_row_width(spec)
        if name == "pokemon":
            rows = spec["card_id"].shape[-1]
            self.group_slot_counts[name] = rows
            # Seat, not slot: 0 for the agent's active+bench (the first half of
            # the rows), 1 for the opponent's. One segment per seat rather than
            # a per-slot table keeps a bench permutation-invariant within its
            # seat while the seats stay distinguishable. An option's
            # scalars[inPlayIndex] still cannot be bound to a specific bench
            # token; that was never possible through this table.
            self._register_segment_ids(
                name,
                torch.cat(
                    [
                        torch.zeros(rows // 2, dtype=torch.int64),
                        torch.ones(rows - rows // 2, dtype=torch.int64),
                    ]
                ),
            )
            if self._pool:
                assert self._entity_dim is not None
                self._pokemon_encoder = nn.Linear(
                    self._pokemon_row_width(spec), self._entity_dim
                )
                seats = 2 if self._pokemon_seat_split else 1
                return seats * self._pool_width(self._entity_dim)
            return rows * self._pokemon_row_width(spec)
        if name in ("my", "opp", "select_deck", "looking"):
            if not isinstance(spec, Composite):
                raise ValueError(
                    f"Zone group '{name}' must be a composite of ids/mask leaves."
                )
            pairs: list[tuple[str, str]] = []
            for leaf_name in spec:
                if not isinstance(leaf_name, str):
                    continue
                if leaf_name == "ids" or leaf_name.endswith("_ids"):
                    mask_name = (
                        "mask"
                        if leaf_name == "ids"
                        else leaf_name[: -len("_ids")] + "_mask"
                    )
                    if mask_name not in spec:
                        raise ValueError(
                            f"Zone group '{name}' has '{leaf_name}' without '{mask_name}'."
                        )
                    pairs.append((leaf_name, mask_name))
            if len(pairs) == 0:
                raise ValueError(f"Zone group '{name}' contains no ids/mask pairs.")
            self._zone_pairs[name] = pairs
            self.group_slot_counts[name] = sum(
                spec[ids_name].shape[-1] for ids_name, _ in pairs
            )
            # One id per zone (e.g. hand vs discard vs prize), repeated
            # across that zone's capacity: without it, the same card looks
            # identical whichever zone holds it, since every zone shares the
            # same card_repr + projection (finding 1).
            self._register_segment_ids(
                name,
                torch.cat(
                    [
                        torch.full(
                            (spec[ids_name].shape[-1],), segment, dtype=torch.int64
                        )
                        for segment, (ids_name, _) in enumerate(pairs)
                    ]
                ),
            )
            dim = self._entity_dim if self._pool else self._card_repr_dim
            assert dim is not None
            zw = (self._pool_width(dim) if self._pool else dim) + 1
            return len(pairs) * zw
        raise ValueError(f"Unknown structured observation group '{name}'.")

    def encode_groups(
        self, *inputs: torch.Tensor | TensorDictBase
    ) -> list[torch.Tensor]:
        """
        Encode each structured observation group separately.

        :return: One ``(*batch, group_feature_widths[i])`` tensor per group, in
            :attr:`group_feature_widths` order (before concatenation). A
            token-per-group backbone (e.g. a transformer trunk) uses this
            directly instead of :meth:`forward`'s single flat vector.
        """
        if len(inputs) != len(self._group_names):
            raise ValueError(
                f"StructuredObsAdapter expected {len(self._group_names)} inputs, got {len(inputs)}."
            )
        parts: list[torch.Tensor] = []
        for name, value in zip(self._group_names, inputs, strict=True):
            if name == "globals":
                assert isinstance(value, torch.Tensor)
                parts.append(value / self._global_scales)
            elif name == "select_cats":
                parts.append(
                    self._embed_categories(value, self._select_category_offsets)
                )
            elif name in ("context_card_ids", "stadium_id"):
                parts.append(self._encode_card_ids(value))
            elif name == "options":
                parts.append(self._encode_options(value))
            elif name == "pokemon":
                parts.append(self._encode_pokemon(value))
            else:
                parts.append(self._encode_zone_group(name, value))
        return parts

    def encode_entity_tokens(
        self,
        *inputs: torch.Tensor | TensorDictBase,
        groups: Sequence[str],
    ) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        """
        Encode selected groups as per-entity token sequences, unpooled.

        Where :meth:`encode_groups` collapses each group to one vector by
        masked-mean pooling, this returns the individual entities — one token
        per option row, per Pokemon row, per card in a zone — so an attention
        trunk can relate them to each other, and a pointer head can score them
        individually. This is the token path the class docstring describes;
        it reuses the same per-entity encoders as the pooled path, so it adds
        no parameters and needs no ``pool=False`` construction.

        Only a subset of groups is requested at a time because the padded slot
        counts are large (``options`` alone is ``max_options + 1``): attention
        is quadratic in the token count, and the opponent forward runs on CPU
        inside every environment worker. Ask for what the trunk actually needs.

        :param inputs: One entry per registered group, in ``in_keys`` order —
            the same arguments :meth:`forward` takes.
        :param groups: Names of the groups to expand. ``globals`` and
            ``select_cats`` are single vectors with no entity axis and are
            rejected; use :meth:`encode_groups` for those.
        :return: ``{name: (tokens, valid_mask)}`` where ``tokens`` is
            ``(*batch, group_slot_counts[name], entity_dim)`` and
            ``valid_mask`` is ``(*batch, group_slot_counts[name])`` — True for
            a real entity, False for padding.
        :raises ValueError: If the input count is wrong, a requested group was
            never registered, a requested group has no entity axis, or the
            adapter was built with ``entity_dim=None`` (legacy flat mode).
        """
        if len(inputs) != len(self._group_names):
            raise ValueError(
                f"StructuredObsAdapter expected {len(self._group_names)} inputs, got {len(inputs)}."
            )
        if not self._pool:
            raise ValueError(
                "encode_entity_tokens needs the per-entity encoders, which exist only when "
                "the adapter is built with entity_dim set (the default)."
            )
        by_name = dict(zip(self._group_names, inputs, strict=True))
        tokens: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        for name in groups:
            if name not in by_name:
                raise ValueError(
                    f"Group '{name}' is not among the adapter's registered groups "
                    f"{self._group_names}."
                )
            if name in ("globals", "select_cats"):
                raise ValueError(
                    f"Group '{name}' is a single feature vector with no entity axis; "
                    "it is already one token via encode_groups()."
                )
            value = by_name[name]
            if name == "options":
                assert isinstance(value, TensorDictBase)
                tokens[name] = (
                    self._option_encoder(self._option_rows(value)),
                    self._option_validity(value),
                )
            elif name == "pokemon":
                assert isinstance(value, TensorDictBase)
                tokens[name] = (
                    self._pokemon_encoder(self._pokemon_rows(value)),
                    value.get("mask"),
                )
            elif name in ("context_card_ids", "stadium_id"):
                assert isinstance(value, torch.Tensor)
                tokens[name] = (self._card_proj(self._card_repr(value)), value != 0)
            else:
                assert isinstance(value, TensorDictBase)
                card_parts: list[torch.Tensor] = []
                mask_parts: list[torch.Tensor] = []
                for ids_name, mask_name in self._zone_pairs[name]:
                    card_parts.append(
                        self._card_proj(self._card_repr(value.get(ids_name)))
                    )
                    mask_parts.append(value.get(mask_name))
                tokens[name] = (
                    torch.cat(card_parts, dim=-2),
                    torch.cat(mask_parts, dim=-1),
                )
        return tokens

    def forward(
        self,
        *inputs: torch.Tensor | TensorDictBase,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Encode the structured observation groups.

        :param inputs: One value per registered group, in ``in_keys`` order.
        :return: ``(*batch, out_features)`` flat vector; or, when the adapter
            was built with ``emit_option_tokens``, a
            ``(state_vector, option_tokens)`` pair with tokens shaped
            ``(*batch, group_slot_counts["options"], option_token_dim)``.
        """
        state = torch.cat(self.encode_groups(*inputs), dim=-1)
        if not self._emit_option_tokens:
            return state
        option_tokens, _ = self.encode_entity_tokens(*inputs, groups=["options"])[
            "options"
        ]
        return state, option_tokens

    # ── Card / attack / category helpers ──────────────────────────────

    def _card_repr(self, card_ids: torch.Tensor) -> torch.Tensor:
        """
        Learned + static + categorical + attack-pool card representation,
        shape ``(*ids, card_repr_dim)``.
        """
        attack_ids = self._card_attack_ids[card_ids]
        return torch.cat(
            [
                self._card_embedding(card_ids),
                self._card_static[card_ids],
                self._embed_categories(
                    self._card_categories[card_ids], self._card_category_offsets
                ),
                self._masked_mean(self._attack_repr(attack_ids), attack_ids != 0),
            ],
            dim=-1,
        )

    def _attack_repr(self, attack_ids: torch.Tensor) -> torch.Tensor:
        """Learned + static attack representation."""
        return torch.cat(
            [self._attack_embedding(attack_ids), self._attack_static[attack_ids]],
            dim=-1,
        )

    def _embed_categories(
        self, values: torch.Tensor, field_offsets: torch.Tensor
    ) -> torch.Tensor:
        """Embed categorical fields through the shared table, flattening the result."""
        return self._flatten_rows(self._category_embedding(values + field_offsets))

    # ── Pooled per-entity encoders ─────────────────────────────────────

    def _encode_card_ids(self, card_ids: torch.Tensor) -> torch.Tensor:
        """Encode card IDs, projecting to entity_dim when pooling is active."""
        reprs = self._card_repr(card_ids)
        if self._pool:
            return self._flatten_rows(self._card_proj(reprs))
        return self._flatten_rows(reprs)

    def _option_rows(self, options: TensorDictBase) -> torch.Tensor:
        """Raw per-option feature rows, shape ``(*batch, n_slots, row_width)``."""
        scalars = options.get("scalars")
        scaled = torch.where(
            scalars < 0.0,
            scalars.new_full((), -1.0),
            scalars / self.OPTION_SCALAR_SCALE,
        )
        return torch.cat(
            [
                self._card_repr(options.get("card_id")),
                self._card_repr(options.get("target_id")),
                self._attack_repr(options.get("attack_id")),
                nn.functional.one_hot(options.get("owner"), self.OWNER_VALUE_COUNT).to(
                    torch.float32
                ),
                self._category_embedding(
                    options.get("cats") + self._option_category_offsets
                ).flatten(-2),
                scaled,
            ]
            # The targeted Pokemon's live state. Already zero-filled by the
            # encoder for options that target nothing, with column 0 the
            # resolved flag, so no absent-value sentinel is needed here.
            + (
                [options.get("target_state") / self._option_target_scales]
                if self._option_target_state
                else []
            ),
            dim=-1,
        )

    def _tool_repr(self, tool_ids: torch.Tensor) -> torch.Tensor:
        """
        Pool a Pokemon's attached tool identities into one card-width vector.

        The first slot always counts, even when it holds the padding id 0.
        That keeps an empty tool list mapped to the id-0 embedding, which is
        what the single scalar ``tool_id`` produced before the cap was
        widened to hold a stacked second tool, so widening it leaves an
        already-trained network's inputs unchanged.

        :param tool_ids: Tool card ids, shape ``(*batch, n_rows, tool_cap)``.
        :return: Pooled representation of shape ``(*batch, n_rows, card_dim)``.
        """
        mask = tool_ids != 0
        mask[..., 0] = True
        return self._masked_mean(self._card_repr(tool_ids), mask)

    def _pokemon_rows(self, pokemon: TensorDictBase) -> torch.Tensor:
        """Raw per-Pokemon feature rows, shape ``(*batch, n_rows, row_width)``."""
        energy_ids = pokemon.get("energy_card_ids")
        pre_evolution_ids = pokemon.get("pre_evolution_ids")
        return torch.cat(
            [
                self._card_repr(pokemon.get("card_id")),
                self._tool_repr(pokemon.get("tool_id")),
                self._masked_mean(self._card_repr(energy_ids), energy_ids != 0),
                self._masked_mean(
                    self._card_repr(pre_evolution_ids), pre_evolution_ids != 0
                ),
                pokemon.get("features") / self._pokemon_feature_scales,
                pokemon.get("mask").to(torch.float32).unsqueeze(-1),
            ],
            dim=-1,
        )

    @staticmethod
    def _option_validity(options: TensorDictBase) -> torch.Tensor:
        """
        A slot is real when the encoder wrote an option type into it, plus
        the final slot, which is the always-present synthetic stop action.

        ``cats[..., 0] == int(option.type) + 1`` for a real option (0 means
        absent) — see ``StructuredObservationEncoder._encode_options``. This
        cannot be ``card_id != 0`` (finding 8): ``card_id`` comes from
        ``OptionReferenceResolver.resolve``, which returns 0 for
        YES/NO/NUMBER/RETREAT/END options — they have no associated card, so
        a selection offering only those would look like every slot is
        padding, and the pooled ``options`` vector would be exactly zero.

        :param options: The ``options`` group's tensordict.
        :return: Boolean tensor of shape ``(*batch, n_slots)``.
        """
        # `!= 0` allocates a new tensor rather than viewing `cats`, so writing
        # the stop slot below does not mutate the caller's input.
        validity = options.get("cats")[..., 0] != 0
        validity[..., -1] = True
        return validity

    def _encode_options(self, options: TensorDictBase) -> torch.Tensor:
        """
        Encode the option table. With pooling: ``(*batch, entity_dim)``
        (masked-mean over valid rows, see :meth:`_option_validity`). Legacy: flat.
        """
        rows = self._option_rows(options)
        if self._pool:
            rows = self._option_encoder(rows)
            return self._masked_mean(rows, self._option_validity(options))
        return self._flatten_rows(rows)

    def _encode_pokemon(self, pokemon: TensorDictBase) -> torch.Tensor:
        """
        Encode the board table. With pooling: ``(*batch, _pool_width(entity_dim))``,
        or twice that under ``pokemon_seat_split``. Legacy: flat.

        The row layout is agent active + agent bench, then opponent active +
        opponent bench (see
        ``StructuredObservationEncoder._encode_pokemon``), so the two seats
        are the two halves of the row axis and splitting there needs no
        extra bookkeeping. Pooling stays *within* a seat, which keeps the
        bench an unordered set — the property the single pool had — while
        making the seats distinguishable.
        """
        rows = self._pokemon_rows(pokemon)
        if not self._pool:
            return self._flatten_rows(rows)
        encoded = self._pokemon_encoder(rows)
        mask = pokemon.get("mask")
        if not self._pokemon_seat_split:
            return self._masked_pool(encoded, mask)
        half = encoded.shape[-2] // 2
        return torch.cat(
            [
                self._masked_pool(encoded[..., :half, :], mask[..., :half]),
                self._masked_pool(encoded[..., half:, :], mask[..., half:]),
            ],
            dim=-1,
        )

    def _encode_zone_group(self, name: str, zones: TensorDictBase) -> torch.Tensor:
        """
        Pool unordered card-ID zones into per-zone summaries.

        With pooling: N_zones × ``(_pool_width(entity_dim) + 1)`` (card repr +
        fill fraction). Legacy: same but with ``card_repr_dim``.
        """
        parts: list[torch.Tensor] = []
        for ids_name, mask_name in self._zone_pairs[name]:
            mask = zones.get(mask_name)
            capacity = mask.shape[-1]
            fill = mask.to(torch.float32).sum(dim=-1, keepdim=True) / capacity
            reprs = self._card_repr(zones.get(ids_name))
            if self._pool:
                parts.append(self._masked_pool(self._card_proj(reprs), mask))
            else:
                parts.append(self._masked_mean(reprs, mask))
            parts.append(fill)
        return torch.cat(parts, dim=-1)

    # ── Shape utilities ────────────────────────────────────────────────

    @staticmethod
    def _flatten_rows(table: torch.Tensor) -> torch.Tensor:
        return table.reshape(*table.shape[:-2], -1)

    @staticmethod
    def _masked_mean(reprs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        weights = mask.to(torch.float32).unsqueeze(-1)
        return (reprs * weights).sum(dim=-2) / weights.sum(dim=-2).clamp(min=1.0)

    def _masked_pool(self, reprs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Collapse a padded set of entity encodings into a fixed-width summary.

        Under ``"mean"`` this is the plain centroid. Under ``"mean_max_sum"``
        the max and a capacity-normalized sum are concatenated alongside it:
        the mean alone cannot express *presence* (one copy of a card in a
        seven-card hand barely moves the centroid) nor *count*, both of which
        the critic needs to read a board.

        :param reprs: Entity encodings of shape ``(*batch, n_slots, dim)``.
        :param mask: Bool occupancy mask of shape ``(*batch, n_slots)``.
        :return: Pooled summary of shape ``(*batch, _pool_width(dim))``.
        """
        mean = self._masked_mean(reprs, mask)
        if self._zone_pooling == "mean":
            return mean
        occupied = mask.unsqueeze(-1)
        # Empty sets would otherwise max to the sentinel; force them to zero so
        # an absent zone reads as "nothing here" rather than a huge constant.
        maximum = reprs.masked_fill(~occupied, torch.finfo(reprs.dtype).min).amax(
            dim=-2
        )
        maximum = torch.where(
            mask.any(dim=-1, keepdim=True), maximum, torch.zeros_like(maximum)
        )
        total = (reprs * occupied.to(reprs.dtype)).sum(dim=-2) / reprs.shape[-2]
        return torch.cat([mean, maximum, total], dim=-1)
