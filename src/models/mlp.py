from typing import cast

import torch
from tensordict import TensorDictBase
from torch import nn
from torchrl.modules import MLP

from src.models.backbone import Backbone, activation_class


class MLPBackbone(Backbone):
    """
    Flat multi-layer-perceptron trunk — the literature's proven PPO baseline.

    Turns every input into a per-sample feature vector, concatenates them
    along the last dimension, and runs the result through a fully-connected
    stack. This is the honest control every richer (per-option-token)
    backbone must beat. By default it emits no per-option tokens
    (:attr:`produces_option_repr` is False) and pairs with
    :class:`~src.models.heads.LinearPolicyHead`; set ``option_tokens=True`` to
    also emit ``option_repr`` and pair with
    :class:`~src.models.heads.PointerPolicyHead` instead — the pointer head's
    gain, if any, without paying the transformer's attention cost.

    With a :class:`~src.models.structured_obs_adapter.StructuredObsAdapter`
    attached (the standard pairing, wired by
    :func:`~src.policies.ppo_actor.build_actor_critic`), the adapter performs
    the model-side half of the observation contract — embedding card/attack
    IDs, normalizing scalars, pooling unordered zones — before the MLP.
    Without one, each input is naively flattened and cast to float: plain
    tensors past their last dimension, nested
    :class:`~tensordict.TensorDictBase` groups leaf by leaf past their own
    ``batch_size``.
    """

    def __init__(
            self,
            input_dim: int,
            out_features: int,
            num_cells: list[int],
            activation: str = "tanh",
            in_keys: list[str] | None = None,
            adapter: nn.Module | None = None,
            option_tokens: bool = False,
    ) -> None:
        """
        :param input_dim: Width of the concatenated feature vector fed to the
            MLP; with an adapter this must equal its ``out_features``.
        :param out_features: Width of the produced ``state_repr``.
        :param num_cells: Hidden layer widths of the MLP.
        :param activation: Hidden activation name (see :func:`activation_class`).
        :param in_keys: Observation keys to consume; defaults to
            ``["observation"]`` (a single pre-built feature vector).
        :param adapter: Optional :class:`~src.models.structured_obs_adapter.
            StructuredObsAdapter` handling the structured groups; None falls
            back to naive flattening.
        :param option_tokens: Emit per-option tokens as ``option_repr`` for a
            pointer head, by projecting
            ``adapter.encode_entity_tokens(groups=["options"])`` to
            ``out_features`` — the same cheap per-entity projection
            :class:`~src.models.transformer.TransformerBackbone` uses when
            ``option_tokens=True`` and ``encoded_option_repr=False``,
            including that group's stop-slot segment embedding (see
            :attr:`~src.models.structured_obs_adapter.StructuredObsAdapter.group_segment_ids`).
            Sets :attr:`produces_option_repr`. Requires an ``adapter`` built
            with ``entity_dim`` set (the default) and an ``options`` group.
        :raises ValueError: If the adapter's output width disagrees with
            ``input_dim``, or ``option_tokens`` is set without a usable
            adapter.
        """
        super().__init__(in_keys=in_keys or ["observation"], out_features=out_features)
        if adapter is not None and adapter.out_features != input_dim:
            raise ValueError(
                f"Adapter produces {adapter.out_features} features but the MLP expects "
                f"input_dim={input_dim}."
            )
        if option_tokens and adapter is None:
            raise ValueError(
                "option_tokens=True needs a StructuredObsAdapter to encode per-option "
                "tokens from; pass one, or leave option_tokens=False for the flat head."
            )
        if option_tokens:
            group_slot_counts = cast(dict[str, int], cast(nn.Module, adapter).group_slot_counts)
            if "options" not in group_slot_counts:
                raise ValueError(
                    "option_tokens=True needs an 'options' group registered on the adapter to "
                    "encode per-option tokens from; this adapter registered only "
                    f"{sorted(group_slot_counts)}."
                )
            entity_dim = cast(int | None, cast(nn.Module, adapter).entity_dim)
            if entity_dim is None:
                raise ValueError(
                    "option_tokens=True needs per-entity encodings, which require the "
                    "adapter to be built with entity_dim set (the default)."
                )
        self.input_dim = input_dim
        self.adapter = adapter
        self.option_tokens = bool(option_tokens)
        self.produces_option_repr = self.option_tokens
        self.mlp = MLP(
            in_features=input_dim,
            out_features=out_features,
            num_cells=list(num_cells),
            activation_class=activation_class(activation),
        )
        if self.option_tokens:
            self.option_projection = nn.Linear(int(entity_dim), out_features)
            segment_ids = cast(torch.Tensor, cast(nn.Module, adapter).group_segment_ids["options"])
            #: One learned vector per option segment (real slot vs. the
            #: synthetic stop slot), mirroring
            #: ``TransformerBackbone.entity_segment_embedding`` so the stop
            #: action gets its own identity here too (finding 2b) rather than
            #: being indistinguishable from a padded slot.
            self.option_segment_embedding = nn.Parameter(
                torch.zeros(int(segment_ids.max().item()) + 1, out_features)
            )
            nn.init.normal_(self.option_segment_embedding, std=0.02)

    def forward(self, *inputs: torch.Tensor | TensorDictBase):
        """
        Vectorize, concatenate and encode the inputs into ``state_repr``.

        :param inputs: One entry per :attr:`in_keys`: either a per-sample
            feature vector shaped ``(*batch, features)``, or a nested
            tensordict group handed to the adapter (or naively flattened
            leaf by leaf when no adapter is attached).
        :return: ``state_repr`` of shape ``(*batch, out_features)``, or a
            ``(state_repr, option_repr)`` pair when :attr:`option_tokens` is
            set, with ``option_repr`` shaped ``(*batch, n_slots, out_features)``.
        """
        if len(inputs) != len(self.in_keys):
            raise ValueError(
                f"MLPBackbone expected {len(self.in_keys)} inputs for keys "
                f"{self.in_keys}, got {len(inputs)}."
            )
        if self.adapter is not None:
            state_repr = self.mlp(self.adapter(*inputs))
        else:
            flat_parts: list[torch.Tensor] = []
            for value in inputs:
                if isinstance(value, TensorDictBase):
                    batch_ndim = len(value.batch_size)
                    for leaf_key in value.keys(include_nested=True, leaves_only=True):
                        leaf = value.get(leaf_key)
                        flat_parts.append(leaf.reshape(*leaf.shape[:batch_ndim], -1).to(torch.float32))
                else:
                    flat_parts.append(value.reshape(*value.shape[:-1], -1).to(torch.float32))
            features = flat_parts[0] if len(flat_parts) == 1 else torch.cat(flat_parts, dim=-1)
            state_repr = self.mlp(features)

        if not self.option_tokens:
            return state_repr
        adapter = cast(nn.Module, self.adapter)
        option_rows, _ = adapter.encode_entity_tokens(*inputs, groups=["options"])["options"]
        segment_ids = cast(torch.Tensor, adapter.group_segment_ids["options"])
        option_repr = self.option_projection(option_rows) + self.option_segment_embedding[segment_ids]
        return state_repr, option_repr