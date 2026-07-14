from abc import ABC, abstractmethod

import torch
from tensordict import TensorDictBase
from torch import nn
from torchrl.modules import MLP

_ACTIVATIONS: dict[str, type[nn.Module]] = {
    "tanh": nn.Tanh,
    "relu": nn.ReLU,
    "elu": nn.ELU,
    "gelu": nn.GELU,
}


def activation_class(name: str) -> type[nn.Module]:
    """
    Resolve an activation name to its :class:`torch.nn.Module` class.

    :param name: Activation name (``tanh``, ``relu``, ``elu``, ``gelu``).
    :return: The matching activation module class.
    :raises ValueError: If the name is unknown.
    """
    try:
        return _ACTIVATIONS[name.lower()]
    except KeyError as error:
        raise ValueError(
            f"Unknown activation '{name}'; expected one of {sorted(_ACTIVATIONS)}."
        ) from error


class Backbone(nn.Module, ABC):
    """
    Shared trunk of the actor-critic: observation tensors -> latent state.

    A backbone reads the observation tensors named by :attr:`in_keys` and
    produces a per-state latent ``state_repr`` consumed by both heads. Richer
    backbones (Deep Sets / Set Transformer / temporal, Phase 2+) additionally
    emit per-option token representations; :attr:`produces_option_repr`
    advertises whether they do, so the assembly in
    :func:`src.policies.ppo_actor.build_ppo_actor_critic` and the pointer head
    can be wired accordingly. Every backbone exposes the same interface, so
    they are swappable purely through ``conf/model/backbone``.

    Implementations set :attr:`in_keys` (the observation tensordict keys they
    consume) and :attr:`out_features` (the width of ``state_repr``), and return
    from :meth:`forward` either a single ``state_repr`` tensor (when
    :attr:`produces_option_repr` is False) or a ``(state_repr, option_repr)``
    tuple.
    """

    #: Whether :meth:`forward` returns per-option token representations.
    produces_option_repr: bool = False

    def __init__(self, in_keys: list[str], out_features: int) -> None:
        """
        :param in_keys: Observation tensordict keys this backbone consumes.
        :param out_features: Width of the produced ``state_repr``.
        """
        super().__init__()
        self.in_keys = list(in_keys)
        self.out_features = out_features

    @abstractmethod
    def forward(self, *inputs: torch.Tensor | TensorDictBase):
        """
        Encode the observation tensors into a latent state.

        :param inputs: One entry per :attr:`in_keys`, in order: either a
            tensor shaped ``(*batch, features)``, or a nested tensordict (for
            an ``in_key`` naming a group of observation fields) whose leaves
            an implementation is responsible for interpreting.
        :return: ``state_repr`` of shape ``(*batch, out_features)``, or a
            ``(state_repr, option_repr)`` tuple when
            :attr:`produces_option_repr` is True.
        """
        raise NotImplementedError


class MLPBackbone(Backbone):
    """
    Flat multi-layer-perceptron trunk — the literature's proven PPO baseline.

    Flattens every input into a per-sample feature vector, concatenates them
    along the last dimension, and runs the result through a fully-connected
    stack. This is the honest control every richer (permutation-invariant)
    backbone must beat. It reads only global state, so it emits no per-option
    tokens (:attr:`produces_option_repr` is False) and pairs with
    :class:`~src.models.heads.LinearPolicyHead`.

    Each positional input is either a plain tensor already shaped
    ``(*batch, features)`` (used as-is, e.g. the flat 36-dim observation or
    the structured encoder's scalar leaves like ``globals``), or a nested
    :class:`~tensordict.TensorDictBase` (e.g. the structured encoder's
    ``options``/``pokemon``/zone groups), whose leaves are individually
    flattened past its own ``batch_size`` and concatenated. Card/attack IDs
    and boolean masks are cast to float and concatenated like any other
    feature — this is a deliberately naive baseline; embedding lookups and
    permutation-invariant pooling are future backbones' job, not this one's.
    """

    produces_option_repr = False

    def __init__(
            self,
            input_dim: int,
            out_features: int,
            num_cells: list[int],
            activation: str = "tanh",
            in_keys: list[str] | None = None,
    ) -> None:
        """
        :param input_dim: Summed width of the concatenated input vectors.
        :param out_features: Width of the produced ``state_repr``.
        :param num_cells: Hidden layer widths of the MLP.
        :param activation: Hidden activation name (see :func:`activation_class`).
        :param in_keys: Observation keys to consume; defaults to
            ``["observation"]`` (the flat observation).
        """
        super().__init__(in_keys=in_keys or ["observation"], out_features=out_features)
        self.input_dim = input_dim
        self.mlp = MLP(
            in_features=input_dim,
            out_features=out_features,
            num_cells=list(num_cells),
            activation_class=activation_class(activation),
        )

    def forward(self, *inputs: torch.Tensor | TensorDictBase) -> torch.Tensor:
        """
        Flatten, concatenate and encode the inputs into ``state_repr``.

        :param inputs: One entry per :attr:`in_keys`: either a per-sample
            feature vector shaped ``(*batch, features)``, or a nested
            tensordict whose leaves are flattened past its own ``batch_size``.
        :return: ``state_repr`` of shape ``(*batch, out_features)``.
        """
        if len(inputs) != len(self.in_keys):
            raise ValueError(
                f"MLPBackbone expected {len(self.in_keys)} inputs for keys "
                f"{self.in_keys}, got {len(inputs)}."
            )
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
        return self.mlp(features)
