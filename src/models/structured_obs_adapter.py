



import torch
from tensordict import TensorDictBase
from torch import nn
from torchrl.data import Composite, TensorSpec

from src.env.card_database import CardDatabase


class StructuredObsAdapter(nn.Module):
    """
    Encodes the structured observation for an MLP or Transformer backbone.

    **How features are built (shared between both paths)**

    Three kinds of input, handled differently:

    ── card IDs (context_card_ids, stadium_id, every card in a zone) ──
        ID → nn.Embedding(8, padding_idx=0) + static features(12) → [20]
        [20] → Linear(20→entity_dim) → [entity_dim]

    ── table rows (options, pokemon) ──
        concat all row fields → [87] or [101] → Linear(→entity_dim) → [entity_dim]
        (one token per row; padding rows included, filtered later)

    ── scalars (globals, select_cats) ──
        globals [41] ÷ fixed scales, select_cats [2] → Embed(4) → [8]

    **Transformer path (pool=False) — token sequence**

    Each group independently produces tokens.  Padding rows are filtered out
    (options: card_id≠0, pokemon: mask=True, zones: mask).  The remaining
    tokens from all groups are concatenated along the sequence axis.

    ::

        group              tokens   example count   dims
        ─────              ──────   ─────────────   ────
        globals               1     pad 41→64       [64]
        select_cats           1     pad 8→64        [64]
        context_card_ids      2     2 cards         [2, 64]
        stadium_id            1     1 card          [1, 64]
        options           N_opt     3 real options  [3, 64]
        pokemon           N_pkm     4 on board      [4, 64]
        my.hand           N_hnd     5 in hand       [5, 64]
        my.discard        N_dis     2 discarded     [2, 64]
        my.prizes         N_prz     6 face-down     [6, 64]
        opp.discard       N_odis    4 discarded     [4, 64]
        opp.prizes        N_oprz    6 face-down     [6, 64]
        select_deck       N_dek     3 during search [3, 64]
        looking           N_look    1 inspected     [1, 64]
                                       ───
                             torch.cat → [~40, 64]  →  Transformer

        Face-down cards (prizes) have ID=0.  Embedding returns zeros, the
        token is all zeros, but the slot is still present — the Transformer
        knows *how many* prizes remain even though it can't see what they are.

    **MLP path (pool=True) — one flat vector**

    Same per-entity encoding.  Then each group is collapsed to a single fixed
    vector by masked set pooling (skip padding).  Zones also get a fill
    fraction (cards present / zone capacity).  Under the default
    ``zone_pooling="mean_max_sum"`` each set contributes ``3 × 64`` (mean,
    element-wise max, capacity-normalized sum) instead of the mean alone: a
    centroid cannot say whether a specific card is *present*, only what the
    average card looks like, which is not enough to read a hand or a board.

    ::

        group              dims (mean_max_sum)   (legacy mean)
        ─────              ───────────────────   ─────────────
        globals                    41                  41
        select_cats                 8                   8
        context_card_ids          128                 128
        stadium_id                 64                  64
        options                    64                  64   (pooled digest only)
        pokemon                   192                  64
        my (3 zones)              579                 195
        opp (2 zones)             386                 130
        select_deck (1)           193                  65
        looking (1)               193                  65
                                 ────                 ───
                      torch.cat → 1848                 824   →  MLP

    **Option tokens (emit_option_tokens=True)**

    The pooled ``options`` entry above is permutation-invariant, so on its own
    it tells the policy *nothing* about which action sits in which slot.  With
    ``emit_option_tokens`` the adapter additionally returns the per-slot table
    ``(*batch, n_slots, entity_dim)``, row ``i`` describing action ``i``, for a
    pointer head to score against.  This is what makes the policy able to
    choose an action on its merits rather than on its index.
    """

    #: Supported set-pooling modes for zone/board groups (see ``zone_pooling``).
    POOL_MODES = ("mean", "mean_max_sum")

    #: Rows reserved per categorical field in the shared category table.
    CATEGORY_VOCAB_SIZE = 64
    #: Field order in the shared category table: selection type, selection
    #: context, then the option table's four category columns.
    SELECT_CATEGORY_FIELD_COUNT = 2
    OPTION_CATEGORY_FIELD_COUNT = 4
    #: ``options.owner`` values: 0 = none, 1 = agent, 2 = opponent.
    OWNER_VALUE_COUNT = 3
    #: Scale for the option scalar columns (zone indices/counts, cap 60).
    OPTION_SCALAR_SCALE = 60.0

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
    _select_category_offsets: torch.Tensor
    _option_category_offsets: torch.Tensor
    _global_scales: torch.Tensor
    _pokemon_feature_scales: torch.Tensor

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
        :param pool: If True, masked-mean pool per-entity encodings (MLP
            path). If False, return the token sequence (Transformer path,
            not yet implemented). Ignored when ``entity_dim`` is None.
        :param emit_option_tokens: If True, :meth:`forward` additionally
            returns the per-option token table, so a pointer head can score
            each action slot against what that option actually *is*. Without
            it the option table is only ever seen through its pooled mean,
            which is permutation-invariant and therefore carries no
            slot-to-option correspondence at all.
        :param zone_pooling: How unordered card sets (zones and the Pokémon
            board) are collapsed. ``"mean"`` is the centroid alone;
            ``"mean_max_sum"`` concatenates the mean, the element-wise max
            and a capacity-normalized sum, which lets the trunk answer "is
            card X present" and "how many" rather than only "what is the
            average card here".
        :raises ValueError: If ``zone_pooling`` is not a supported mode.
        """
        super().__init__()
        if zone_pooling not in self.POOL_MODES:
            raise ValueError(
                f"Unknown zone_pooling '{zone_pooling}'; expected one of {list(self.POOL_MODES)}."
            )
        self._zone_pooling = zone_pooling
        self._emit_option_tokens = emit_option_tokens
        database = card_database if card_database is not None else CardDatabase()
        card_static = database.card_features
        attack_static = database.attack_features
        self.register_buffer(
            "_card_static", card_static / card_static.abs().amax(dim=0).clamp(min=1.0)
        )
        self.register_buffer(
            "_attack_static", attack_static / attack_static.abs().amax(dim=0).clamp(min=1.0)
        )
        self._card_repr_dim = card_embed_dim + card_static.shape[1]
        self._attack_repr_dim = attack_embed_dim + attack_static.shape[1]
        self._category_embed_dim: int = category_embed_dim
        self._entity_dim: int | None = entity_dim
        self._pool: bool = pool and entity_dim is not None

        self._card_embedding = nn.Embedding(card_static.shape[0], card_embed_dim, padding_idx=0)
        self._attack_embedding = nn.Embedding(attack_static.shape[0], attack_embed_dim, padding_idx=0)
        field_count = self.SELECT_CATEGORY_FIELD_COUNT + self.OPTION_CATEGORY_FIELD_COUNT
        self._category_embedding = nn.Embedding(
            field_count * self.CATEGORY_VOCAB_SIZE, category_embed_dim
        )
        self.register_buffer(
            "_select_category_offsets",
            torch.arange(self.SELECT_CATEGORY_FIELD_COUNT, dtype=torch.int64) * self.CATEGORY_VOCAB_SIZE,
        )
        self.register_buffer(
            "_option_category_offsets",
            (torch.arange(self.OPTION_CATEGORY_FIELD_COUNT, dtype=torch.int64)
             + self.SELECT_CATEGORY_FIELD_COUNT) * self.CATEGORY_VOCAB_SIZE,
        )
        global_scales = self.GAME_SCALES + self.SELECT_SCALES + 2 * self.PLAYER_SCALES
        self.register_buffer("_global_scales", torch.tensor(global_scales, dtype=torch.float32))
        self.register_buffer(
            "_pokemon_feature_scales",
            torch.tensor(self.POKEMON_FEATURE_SCALES, dtype=torch.float32),
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
        self.out_features = 0
        for key in in_keys:
            name = key[-1] if isinstance(key, tuple) else key
            self._group_names.append(name)
            self.out_features += self._register_group(name, obs_spec[key])
        if self._emit_option_tokens and "options" not in self._group_names:
            raise ValueError(
                "emit_option_tokens=True requires the 'options' group in in_keys; "
                f"got {self._group_names}."
            )

    @property
    def emits_option_tokens(self) -> bool:
        """
        Whether :meth:`forward` returns per-option tokens alongside the state vector.

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

    def _option_row_width(self, spec: Composite) -> int:
        """Flattened width of one option table row (card+target+attack+cats+scalars)."""
        return (
            2 * self._card_repr_dim
            + self._attack_repr_dim
            + self.OWNER_VALUE_COUNT
            + self.OPTION_CATEGORY_FIELD_COUNT * self._category_embed_dim
            + spec["scalars"].shape[-1]
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
        groups contribute ``entity_dim + 1`` per zone.
        """
        if name == "globals":
            return self._global_scales.shape[0]
        if name == "select_cats":
            return self.SELECT_CATEGORY_FIELD_COUNT * self._category_embed_dim
        if name == "context_card_ids":
            width = self._entity_dim if self._pool else self._card_repr_dim
            assert width is not None
            return spec.shape[-1] * width
        if name == "stadium_id":
            width = self._entity_dim if self._pool else self._card_repr_dim
            assert width is not None
            return width
        if name == "options":
            if self._pool:
                assert self._entity_dim is not None
                self._option_encoder = nn.Linear(self._option_row_width(spec), self._entity_dim)
                return self._entity_dim
            return spec["card_id"].shape[-1] * self._option_row_width(spec)
        if name == "pokemon":
            if self._pool:
                assert self._entity_dim is not None
                self._pokemon_encoder = nn.Linear(self._pokemon_row_width(spec), self._entity_dim)
                return self._pool_width(self._entity_dim)
            return spec["card_id"].shape[-1] * self._pokemon_row_width(spec)
        if name in ("my", "opp", "select_deck", "looking"):
            if not isinstance(spec, Composite):
                raise ValueError(f"Zone group '{name}' must be a composite of ids/mask leaves.")
            pairs: list[tuple[str, str]] = []
            for leaf_name in spec:
                if not isinstance(leaf_name, str):
                    continue
                if leaf_name == "ids" or leaf_name.endswith("_ids"):
                    mask_name = "mask" if leaf_name == "ids" else leaf_name[: -len("_ids")] + "_mask"
                    if mask_name not in spec:
                        raise ValueError(f"Zone group '{name}' has '{leaf_name}' without '{mask_name}'.")
                    pairs.append((leaf_name, mask_name))
            if len(pairs) == 0:
                raise ValueError(f"Zone group '{name}' contains no ids/mask pairs.")
            self._zone_pairs[name] = pairs
            dim = self._entity_dim if self._pool else self._card_repr_dim
            assert dim is not None
            zone_width = (self._pool_width(dim) if self._pool else dim) + 1
            return len(pairs) * zone_width
        raise ValueError(f"Unknown structured observation group '{name}'.")

    def forward(
            self,
            *inputs: torch.Tensor | TensorDictBase,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Encode the structured observation groups.

        :param inputs: One value per registered group, in ``in_keys`` order.
        :return: ``(*batch, out_features)`` state vector; or, when the adapter
            emits option tokens, a ``(state_vector, option_tokens)`` pair with
            tokens shaped ``(*batch, n_slots, option_token_dim)``.
        """
        if len(inputs) != len(self._group_names):
            raise ValueError(
                f"StructuredObsAdapter expected {len(self._group_names)} inputs, got {len(inputs)}."
            )
        parts: list[torch.Tensor] = []
        option_tokens: torch.Tensor | None = None
        for name, value in zip(self._group_names, inputs, strict=True):
            if name == "globals":
                assert isinstance(value, torch.Tensor)
                parts.append(value / self._global_scales)
            elif name == "select_cats":
                parts.append(self._embed_categories(value, self._select_category_offsets))
            elif name in ("context_card_ids", "stadium_id"):
                parts.append(self._encode_card_ids(value))
            elif name == "options":
                assert isinstance(value, TensorDictBase)
                parts.append(self._encode_options(value))
                if self._emit_option_tokens:
                    option_tokens = self._option_tokens(value)
            elif name == "pokemon":
                parts.append(self._encode_pokemon(value))
            else:
                parts.append(self._encode_zone_group(name, value))
        state = torch.cat(parts, dim=-1)
        if option_tokens is None:
            return state
        return state, option_tokens

    # ── Card / attack / category helpers ──────────────────────────────

    def _card_repr(self, card_ids: torch.Tensor) -> torch.Tensor:
        """Learned + static card representation, shape ``(*ids, card_repr_dim)``."""
        return torch.cat([self._card_embedding(card_ids), self._card_static[card_ids]], dim=-1)

    def _attack_repr(self, attack_ids: torch.Tensor) -> torch.Tensor:
        """Learned + static attack representation."""
        return torch.cat([self._attack_embedding(attack_ids), self._attack_static[attack_ids]], dim=-1)

    def _embed_categories(self, values: torch.Tensor, field_offsets: torch.Tensor) -> torch.Tensor:
        """Embed categorical fields through the shared table, flattening the result."""
        return self._flatten_rows(self._category_embedding(values + field_offsets))

    # ── Pooled per-entity encoders ─────────────────────────────────────

    def _encode_card_ids(self, card_ids: torch.Tensor) -> torch.Tensor:
        """Encode card IDs, projecting to entity_dim when pooling is active."""
        reprs = self._card_repr(card_ids)
        if self._pool:
            return self._flatten_rows(self._card_proj(reprs))
        return self._flatten_rows(reprs)

    def _option_tokens(self, options: TensorDictBase) -> torch.Tensor:
        """
        Per-option token table, one row per action slot.

        Row ``i`` encodes option ``i`` of the current selection, aligned with
        the environment's action mask, so a pointer head can score slot ``i``
        from what that option actually does.

        :param options: The ``options`` observation group.
        :return: Tokens of shape ``(*batch, n_slots, entity_dim)``.
        :raises ValueError: If the adapter is not in the pooled/projected mode.
        """
        if not self._pool:
            raise ValueError("Option tokens require the projected (pooled) adapter mode.")
        return self._option_encoder(self._option_rows(options))

    def _option_rows(self, options: TensorDictBase) -> torch.Tensor:
        """
        Concatenate the raw per-option row features before projection.

        :param options: The ``options`` observation group.
        :return: Rows of shape ``(*batch, n_slots, option_row_width)``.
        """
        scalars = options.get("scalars")
        scaled = torch.where(
            scalars < 0.0, scalars.new_full((), -1.0), scalars / self.OPTION_SCALAR_SCALE
        )
        rows = torch.cat(
            [
                self._card_repr(options.get("card_id")),
                self._card_repr(options.get("target_id")),
                self._attack_repr(options.get("attack_id")),
                nn.functional.one_hot(options.get("owner"), self.OWNER_VALUE_COUNT).to(torch.float32),
                self._category_embedding(options.get("cats") + self._option_category_offsets).flatten(-2),
                scaled,
            ],
            dim=-1,
        )
        return rows

    def _encode_options(self, options: TensorDictBase) -> torch.Tensor:
        """
        Pooled option summary for the state vector.

        This is a permutation-invariant digest of "what can I do right now",
        which is legitimate *context* but carries no slot correspondence. The
        per-slot detail lives in :meth:`_option_tokens`.

        :param options: The ``options`` observation group.
        :return: ``(*batch, entity_dim)`` when pooling, else the flat table.
        """
        rows = self._option_rows(options)
        if self._pool:
            return self._masked_mean(
                self._option_encoder(rows), options.get("card_id") != 0
            )
        return self._flatten_rows(rows)

    def _encode_pokemon(self, pokemon: TensorDictBase) -> torch.Tensor:
        """
        Encode the board table. With pooling: ``(*batch, entity_dim)``. Legacy: flat.
        """
        energy_ids = pokemon.get("energy_card_ids")
        pre_evolution_ids = pokemon.get("pre_evolution_ids")
        rows = torch.cat(
            [
                self._card_repr(pokemon.get("card_id")),
                self._card_repr(pokemon.get("tool_id")),
                self._masked_mean(self._card_repr(energy_ids), energy_ids != 0),
                self._masked_mean(self._card_repr(pre_evolution_ids), pre_evolution_ids != 0),
                pokemon.get("features") / self._pokemon_feature_scales,
                pokemon.get("mask").to(torch.float32).unsqueeze(-1),
            ],
            dim=-1,
        )
        if self._pool:
            return self._masked_pool(self._pokemon_encoder(rows), pokemon.get("mask"))
        return self._flatten_rows(rows)

    def _encode_zone_group(self, name: str, zones: TensorDictBase) -> torch.Tensor:
        """
        Pool unordered card-ID zones into per-zone summaries.

        With pooling: N_zones × ``(entity_dim + 1)`` (card repr + fill fraction).
        Legacy: same but with ``card_repr_dim``.
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
        maximum = reprs.masked_fill(~occupied, torch.finfo(reprs.dtype).min).amax(dim=-2)
        maximum = torch.where(mask.any(dim=-1, keepdim=True), maximum, torch.zeros_like(maximum))
        total = (reprs * occupied.to(reprs.dtype)).sum(dim=-2) / reprs.shape[-2]
        return torch.cat([mean, maximum, total], dim=-1)
