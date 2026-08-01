from abc import ABC, abstractmethod

import torch
from tensordict import TensorDictBase
from torch import nn

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

