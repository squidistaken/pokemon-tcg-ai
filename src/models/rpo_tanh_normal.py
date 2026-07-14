from __future__ import annotations

import torch
from torchrl.modules import TanhNormal


class RPOTanhNormal(TanhNormal):
    """TanhNormal with RPO (Robust Policy Optimization) mean perturbation.

    When ``rpo_enabled`` is True (toggled by the trainer around loss
    computation), adds ``Uniform(-rpo_alpha, rpo_alpha)`` noise to ``loc``
    before constructing the distribution.  During rollout collection the
    flag stays False so behaviour is identical to plain TanhNormal.

    .. note::
        This is the **continuous**-action RPO distribution and is currently
        **inert** in this project: the TCG policy is discrete and uses
        :class:`~src.models.masked_rpo_categorical.MaskedRPOCategorical` for the
        active RPO path. ``RPOTanhNormal`` is kept wired for a future continuous
        head but has no effect while the action space is a masked ``Categorical``.

    Reference: Liang et al., "RPO: Robust Policy Optimization"
    """

    rpo_alpha: float = 0.5
    rpo_enabled: bool = False

    def __init__(self, loc: torch.Tensor, scale: torch.Tensor, *args, **kwargs):
        if RPOTanhNormal.rpo_enabled and RPOTanhNormal.rpo_alpha > 0:
            z = (torch.rand_like(loc) * 2.0 - 1.0) * RPOTanhNormal.rpo_alpha
            loc = loc + z
        super().__init__(loc, scale, *args, **kwargs)
