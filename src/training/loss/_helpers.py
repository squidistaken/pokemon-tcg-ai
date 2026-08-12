from __future__ import annotations

import torch
from tensordict import TensorDictBase


def _sum_loss_keys(loss_vals: TensorDictBase) -> torch.Tensor:
    """
    Sum the scalar entries whose key starts with ``loss_``.

    TorchRL loss modules (:class:`~torchrl.objectives.ClipPPOLoss`) return a
    tensordict that mixes the optimizable ``loss_*`` terms (``loss_objective``,
    ``loss_critic``, ``loss_entropy``) with non-differentiable diagnostics
    (``ESS``, ``kl_approx``, ``clip_fraction``, ``entropy``,
    ``explained_variance``). Only the former belong in the backward pass; this
    helper selects them by the ``loss_`` prefix, so the same optimization loop
    drives any such loss module unchanged.

    :param loss_vals: TensorDict returned by a loss module's ``forward``.
    :return: Scalar tensor: the sum of every ``loss_``-prefixed entry.
    :raises KeyError: If the loss module returned no ``loss_``-prefixed entry.
    """
    total: torch.Tensor | None = None
    for key, value in loss_vals.items():
        if key.startswith("loss_"):
            total = value if total is None else total + value
    if total is None:
        raise KeyError(
            "loss module returned no 'loss_'-prefixed entries to optimize; "
            f"got keys {sorted(loss_vals.keys())}"
        )
    return total
