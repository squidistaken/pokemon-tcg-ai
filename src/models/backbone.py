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

    Turns every input into a per-sample feature vector, concatenates them
    along the last dimension, and runs the result through a fully-connected
    stack. This is the honest control every richer (per-option-token)
    backbone must beat. It emits no per-option tokens
    (:attr:`produces_option_repr` is False) and pairs with
    :class:`~src.models.heads.LinearPolicyHead`.

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

    produces_option_repr = False

    def __init__(
            self,
            input_dim: int,
            out_features: int,
            num_cells: list[int],
            activation: str = "tanh",
            in_keys: list[str] | None = None,
            adapter: nn.Module | None = None,
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
        :raises ValueError: If the adapter's output width disagrees with
            ``input_dim``.
        """
        super().__init__(in_keys=in_keys or ["observation"], out_features=out_features)
        if adapter is not None and adapter.out_features != input_dim:
            raise ValueError(
                f"Adapter produces {adapter.out_features} features but the MLP expects "
                f"input_dim={input_dim}."
            )
        self.input_dim = input_dim
        self.adapter = adapter
        self.mlp = MLP(
            in_features=input_dim,
            out_features=out_features,
            num_cells=list(num_cells),
            activation_class=activation_class(activation),
        )

    def forward(self, *inputs: torch.Tensor | TensorDictBase) -> torch.Tensor:
        """
        Vectorize, concatenate and encode the inputs into ``state_repr``.

        :param inputs: One entry per :attr:`in_keys`: either a per-sample
            feature vector shaped ``(*batch, features)``, or a nested
            tensordict group handed to the adapter (or naively flattened
            leaf by leaf when no adapter is attached).
        :return: ``state_repr`` of shape ``(*batch, out_features)``.
        """
        if len(inputs) != len(self.in_keys):
            raise ValueError(
                f"MLPBackbone expected {len(self.in_keys)} inputs for keys "
                f"{self.in_keys}, got {len(inputs)}."
            )
        if self.adapter is not None:
            return self.mlp(self.adapter(*inputs))
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
