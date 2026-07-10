from abc import ABC, abstractmethod

import torch
import torch.nn as nn
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
    def forward(self, *inputs: torch.Tensor):
        """
        Encode the observation tensors into a latent state.

        :param inputs: One tensor per entry of :attr:`in_keys`, in order, each
            shaped ``(*batch, features)``.
        :return: ``state_repr`` of shape ``(*batch, out_features)``, or a
            ``(state_repr, option_repr)`` tuple when
            :attr:`produces_option_repr` is True.
        """
        raise NotImplementedError


class MLPBackbone(Backbone):
    """
    Flat multi-layer-perceptron trunk — the literature's proven PPO baseline.

    Concatenates its per-sample feature vectors along the last dimension and
    runs the result through a fully-connected stack. It runs on the flat
    36-dim observation and is the honest
    control every richer backbone must beat. It reads only global state, so it
    emits no per-option tokens (:attr:`produces_option_repr` is False) and
    pairs with :class:`~src.models.heads.LinearPolicyHead`.

    Every input is expected to be a per-sample feature vector (its last
    dimension is the feature axis); 
    
    #TODO: flattening multi-dimensional token tables needed once we tokenize observations.
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

    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        """
        Concatenate the input vectors and encode them into ``state_repr``.

        :param inputs: One per-sample feature vector per entry of
            :attr:`in_keys`, each shaped ``(*batch, features)`` with matching
            leading dimensions.
        :return: ``state_repr`` of shape ``(*batch, out_features)``.
        """
        if len(inputs) != len(self.in_keys):
            raise ValueError(
                f"MLPBackbone expected {len(self.in_keys)} inputs for keys "
                f"{self.in_keys}, got {len(inputs)}."
            )
        features = inputs[0] if len(inputs) == 1 else torch.cat(inputs, dim=-1)
        return self.mlp(features.to(torch.float32))
