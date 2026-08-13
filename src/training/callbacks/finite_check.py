from collections.abc import Mapping

import torch


def non_finite_entries(state_dict: Mapping[str, torch.Tensor]) -> list[str]:
    """
    Name every floating-point tensor in a state dict holding a NaN or an Inf.

    Guards the two writers that persist a model. A single non-finite gradient
    makes ``clip_grad_norm_`` scale the whole gradient vector by NaN, which
    turns every parameter NaN in one optimizer step; persisting that state
    overwrites the last resumable one, so every later restart reloads a dead
    model and its workers fail on the first forward pass.

    :param state_dict: Mapping of parameter name to tensor, as returned by
        :meth:`torch.nn.Module.state_dict`.
    :return: Names of the non-finite tensors, empty when the state is clean.
    """
    return [
        name
        for name, tensor in state_dict.items()
        if torch.is_tensor(tensor)
        and tensor.is_floating_point()
        and not torch.isfinite(tensor).all()
    ]
