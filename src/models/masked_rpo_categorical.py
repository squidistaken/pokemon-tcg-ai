from __future__ import annotations

import torch
from torchrl.modules.distributions import MaskedCategorical


class MaskedRPOCategorical(MaskedCategorical):
    """MaskedCategorical with RPO (Robust Policy Optimization) logit perturbation.

    Discrete analogue of :class:`~src.models.rpo_tanh_normal.RPOTanhNormal`.
    When ``rpo_enabled`` is True (toggled by the trainer around loss
    computation), adds ``Uniform(-rpo_alpha, rpo_alpha)`` noise to ``logits``
    *before* delegating to
    :class:`~torchrl.modules.distributions.MaskedCategorical`. During rollout
    collection the flag stays False, so behaviour is identical to a plain
    ``MaskedCategorical``; only the PPO loss-recompute pass sees noise.

    Safety: the parent constructor masks illegal actions (``_mask_logits`` sets
    them to ``-inf``) *after* receiving ``logits``, so perturbation runs strictly
    before masking and can never make a masked-out action samplable — a stronger
    guarantee than the continuous case has.

    .. note::
        This is a **novel discrete-RPO extension**. RPO is originally a
        *continuous*-control technique (perturbing a Gaussian mean); its benefit
        for a masked-discrete action space is **not empirically validated** in
        this project.

    .. note::
        ``rpo_enabled`` is *process-global mutable state* (a class attribute),
        mirroring ``RPOTanhNormal``. This is safe here because collection and
        optimization run sequentially in a single process, and the trainer flips
        it inside a ``try/finally``; it would be unsafe under asynchronous
        collection sharing this class.

    Reference: Liang et al., "RPO: Robust Policy Optimization"
    """

    rpo_alpha: float = 0.5
    rpo_enabled: bool = False

    def __init__(self, logits: torch.Tensor | None = None, *args, **kwargs) -> None:
        if (
            MaskedRPOCategorical.rpo_enabled
            and MaskedRPOCategorical.rpo_alpha > 0
            and logits is not None
        ):
            noise = (torch.rand_like(logits) * 2.0 - 1.0) * MaskedRPOCategorical.rpo_alpha
            logits = logits + noise
        super().__init__(logits, *args, **kwargs)
