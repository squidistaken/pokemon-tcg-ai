import torch
from tensordict import TensorDictBase
from torch import nn
from torch.nn import functional as nn_functional
from torchrl.data import Composite, TensorSpec

from src.env.card_database import CardDatabase


class StructuredObsAdapter(nn.Module):
    """
    Model-side front end turning the structured observation into one sane
    float vector for a flat trunk.

    The :class:`~src.env.structured_observation_encoder.
    StructuredObservationEncoder` deliberately emits raw material — integer
    card/attack IDs, unscaled counts, padded ID tables — and defers every
    modelling decision to the model. This module is that deferred decision
    for the MLP path:

    - Card and attack IDs go through learned embeddings (``padding_idx=0``
      so ID 0 = none/padding/face-down stays a zero vector), concatenated
      with the card's fixed :class:`~src.env.card_database.CardDatabase`
      feature row (normalized per column), so identity is learnable and
      real card properties are available from step one.
    - Small categorical fields (selection type/context, per-option
      categories) use a shared embedding table with a per-field row offset,
      so the same integer means different things in different fields.
    - Float leaves (``globals``, ``pokemon.features``, ``options.scalars``)
      are divided by fixed scale constants mirroring the encoder's layout
      and the engine's constants (deck 60, prizes 6, bench 8, ...), keeping
      every input O(1) for a saturating activation.
    - Unordered ID zones (hand, discards, prizes, ``select_deck``,
      ``looking``, per-Pokemon energy/pre-evolutions) are masked-mean pooled
      into one card representation plus a fill fraction, which removes the
      padded/positional waste and makes those zones permutation-invariant.
    - Slotted tables (``options``, ``pokemon``) keep one row per slot —
      row order is the action/board correspondence — and are flattened
      after embedding.

    The scale constants and group handling are intentionally coupled to the
    encoder's documented layout (``docs/torchrl_environment.md``); width
    checks against the observation spec raise at construction if the two
    drift apart.
    """

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

    def __init__(
            self,
            obs_spec: Composite,
            in_keys: list[str | tuple[str, ...]],
            card_embed_dim: int = 8,
            attack_embed_dim: int = 8,
            category_embed_dim: int = 4,
            card_database: CardDatabase | None = None,
    ) -> None:
        """
        :param obs_spec: Full environment observation spec; the entries named
            by ``in_keys`` must match the structured encoder's groups.
        :param in_keys: Observation keys the backbone consumes, in order;
            each names either one of the encoder's leaf fields (``globals``,
            ``select_cats``, ``context_card_ids``, ``stadium_id``) or one of
            its composite groups (``options``, ``pokemon``, ``my``, ``opp``,
            ``select_deck``, ``looking``).
        :param card_embed_dim: Width of the learned per-card embedding.
        :param attack_embed_dim: Width of the learned per-attack embedding.
        :param category_embed_dim: Width of the shared categorical embedding.
        :param card_database: Static card/attack lookup tables; built from
            the engine when not supplied.
        :raises ValueError: If an in-key names an unknown group, or a leaf
            width disagrees with the scale constants (encoder layout drift).
        """
        super().__init__()
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
        self._category_embed_dim = category_embed_dim

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

        self._group_names: list[str] = []
        self._zone_pairs: dict[str, list[tuple[str, str]]] = {}
        self.out_features = 0
        for key in in_keys:
            name = key[-1] if isinstance(key, tuple) else key
            self._group_names.append(name)
            self.out_features += self._register_group(name, obs_spec[key])

    def _register_group(self, name: str, spec: TensorSpec | Composite) -> int:
        """
        Validate one in-key's spec against the expected encoder layout.

        For the zone groups this also records the ``*_ids``/``*_mask`` leaf
        pairs (in spec order) that :meth:`_encode_zone_group` will pool.

        :param name: Group name (the in-key's last element).
        :param spec: The observation spec entry for this group.
        :return: Flattened feature width this group contributes.
        :raises ValueError: On an unknown group name or a width mismatch.
        """
        if name == "globals":
            expected = self._global_scales.shape[0]
            if spec.shape[-1] != expected:
                raise ValueError(
                    f"globals has {spec.shape[-1]} entries but the adapter's scale table "
                    f"expects {expected}; the encoder layout and adapter have drifted apart."
                )
            return expected
        if name == "select_cats":
            if spec.shape[-1] != self.SELECT_CATEGORY_FIELD_COUNT:
                raise ValueError(
                    f"select_cats has {spec.shape[-1]} fields, expected "
                    f"{self.SELECT_CATEGORY_FIELD_COUNT}."
                )
            return self.SELECT_CATEGORY_FIELD_COUNT * self._category_embed_dim
        if name == "context_card_ids":
            return spec.shape[-1] * self._card_repr_dim
        if name == "stadium_id":
            return spec.shape[-1] * self._card_repr_dim
        if name == "options":
            if spec["cats"].shape[-1] != self.OPTION_CATEGORY_FIELD_COUNT:
                raise ValueError(
                    f"options.cats has {spec['cats'].shape[-1]} fields, expected "
                    f"{self.OPTION_CATEGORY_FIELD_COUNT}."
                )
            n_slots = spec["card_id"].shape[-1]
            row_width = (
                2 * self._card_repr_dim
                + self._attack_repr_dim
                + self.OWNER_VALUE_COUNT
                + self.OPTION_CATEGORY_FIELD_COUNT * self._category_embed_dim
                + spec["scalars"].shape[-1]
            )
            return n_slots * row_width
        if name == "pokemon":
            expected = self._pokemon_feature_scales.shape[0]
            if spec["features"].shape[-1] != expected:
                raise ValueError(
                    f"pokemon.features has {spec['features'].shape[-1]} columns but the "
                    f"adapter's scale table expects {expected}."
                )
            rows = spec["card_id"].shape[-1]
            row_width = 4 * self._card_repr_dim + expected + 1
            return rows * row_width
        if name in ("my", "opp", "select_deck", "looking"):
            pairs: list[tuple[str, str]] = []
            for leaf_name in spec.keys():
                if leaf_name == "ids" or leaf_name.endswith("_ids"):
                    mask_name = "mask" if leaf_name == "ids" else leaf_name[: -len("_ids")] + "_mask"
                    if mask_name not in spec.keys():
                        raise ValueError(f"Zone group '{name}' has '{leaf_name}' without '{mask_name}'.")
                    pairs.append((leaf_name, mask_name))
            if len(pairs) == 0:
                raise ValueError(f"Zone group '{name}' contains no ids/mask pairs.")
            self._zone_pairs[name] = pairs
            return len(pairs) * (self._card_repr_dim + 1)
        raise ValueError(
            f"Unknown structured observation group '{name}'; the adapter handles the "
            f"StructuredObservationEncoder groups only."
        )

    def forward(self, *inputs: torch.Tensor | TensorDictBase) -> torch.Tensor:
        """
        Encode the structured observation groups into one flat float vector.

        :param inputs: One entry per constructor ``in_keys``, in order: leaf
            tensors shaped ``(*batch, features)`` or group tensordicts.
        :return: Float32 tensor of shape ``(*batch, out_features)``.
        :raises ValueError: If the number of inputs does not match ``in_keys``.
        """
        if len(inputs) != len(self._group_names):
            raise ValueError(
                f"StructuredObsAdapter expected {len(self._group_names)} inputs for groups "
                f"{self._group_names}, got {len(inputs)}."
            )
        parts: list[torch.Tensor] = []
        for name, value in zip(self._group_names, inputs):
            if name == "globals":
                parts.append(value / self._global_scales)
            elif name == "select_cats":
                parts.append(self._embed_categories(value, self._select_category_offsets))
            elif name in ("context_card_ids", "stadium_id"):
                parts.append(self._flatten_rows(self._card_repr(value)))
            elif name == "options":
                parts.append(self._encode_options(value))
            elif name == "pokemon":
                parts.append(self._encode_pokemon(value))
            else:
                parts.append(self._encode_zone_group(name, value))
        return parts[0] if len(parts) == 1 else torch.cat(parts, dim=-1)

    def _card_repr(self, card_ids: torch.Tensor) -> torch.Tensor:
        """
        Look up the learned-plus-static representation of card IDs.

        :param card_ids: Int64 tensor of card IDs (0 = none/padding).
        :return: Tensor of shape ``(*card_ids.shape, card_repr_dim)``.
        """
        return torch.cat([self._card_embedding(card_ids), self._card_static[card_ids]], dim=-1)

    def _attack_repr(self, attack_ids: torch.Tensor) -> torch.Tensor:
        """
        Look up the learned-plus-static representation of attack IDs.

        :param attack_ids: Int64 tensor of attack IDs (0 = none/padding).
        :return: Tensor of shape ``(*attack_ids.shape, attack_repr_dim)``.
        """
        return torch.cat(
            [self._attack_embedding(attack_ids), self._attack_static[attack_ids]], dim=-1
        )

    def _embed_categories(self, values: torch.Tensor, field_offsets: torch.Tensor) -> torch.Tensor:
        """
        Embed a block of categorical fields through the shared table.

        :param values: Int64 tensor ``(*batch, n_fields)`` of shifted enum
            values (0 = absent), one column per field.
        :param field_offsets: Row offset per field into the shared table.
        :return: Tensor ``(*batch, n_fields * category_embed_dim)``.
        """
        return self._flatten_rows(self._category_embedding(values + field_offsets))

    @staticmethod
    def _flatten_rows(table: torch.Tensor) -> torch.Tensor:
        """
        Merge the last two dimensions (rows x features) into one.

        :param table: Tensor of shape ``(*batch, rows, features)``.
        :return: Tensor of shape ``(*batch, rows * features)``.
        """
        return table.reshape(*table.shape[:-2], -1)

    @staticmethod
    def _masked_mean(reprs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Average representations over a set dimension, ignoring padding.

        :param reprs: Tensor ``(*batch, set_size, features)``.
        :param mask: Bool tensor ``(*batch, set_size)`` marking real entries.
        :return: Tensor ``(*batch, features)``; zero where the set is empty.
        """
        weights = mask.to(torch.float32).unsqueeze(-1)
        return (reprs * weights).sum(dim=-2) / weights.sum(dim=-2).clamp(min=1.0)

    def _encode_options(self, options: TensorDictBase) -> torch.Tensor:
        """
        Encode the per-slot option table, keeping the slot-to-action rows.

        :param options: Group tensordict with ``card_id``/``target_id``/
            ``attack_id``/``owner``/``cats``/``scalars`` leaves.
        :return: Tensor ``(*batch, n_slots * option_row_width)``.
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
                nn_functional.one_hot(options.get("owner"), self.OWNER_VALUE_COUNT).to(torch.float32),
                self._category_embedding(options.get("cats") + self._option_category_offsets).flatten(-2),
                scaled,
            ],
            dim=-1,
        )
        return self._flatten_rows(rows)

    def _encode_pokemon(self, pokemon: TensorDictBase) -> torch.Tensor:
        """
        Encode the per-slot board table, pooling each row's attached cards.

        :param pokemon: Group tensordict with ``card_id``/``tool_id``/
            ``energy_card_ids``/``pre_evolution_ids``/``features``/``mask``.
        :return: Tensor ``(*batch, rows * pokemon_row_width)``.
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
        return self._flatten_rows(rows)

    def _encode_zone_group(self, name: str, zones: TensorDictBase) -> torch.Tensor:
        """
        Pool a group of unordered card-ID zones into per-zone summaries.

        Each ``ids``/``mask`` pair becomes the masked mean of its card
        representations plus the zone's fill fraction, so zone content is
        permutation-invariant and independent of the padded capacity.

        :param name: Zone group name registered at construction.
        :param zones: Group tensordict holding the ids/mask leaf pairs.
        :return: Tensor ``(*batch, n_zones * (card_repr_dim + 1))``.
        """
        parts: list[torch.Tensor] = []
        for ids_name, mask_name in self._zone_pairs[name]:
            mask = zones.get(mask_name)
            capacity = mask.shape[-1]
            fill = mask.to(torch.float32).sum(dim=-1, keepdim=True) / capacity
            parts.append(self._masked_mean(self._card_repr(zones.get(ids_name)), mask))
            parts.append(fill)
        return torch.cat(parts, dim=-1)
