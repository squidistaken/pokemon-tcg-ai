import copy
import logging

import torch
from tensordict import TensorDict
from torch import nn

logger = logging.getLogger(__name__)


class KLAnchor(nn.Module):
    """
    Penalize divergence from a frozen (behaviour-cloned) policy.

    AlphaStar's style KL divergence term against the supervised policy for
    the whole reinforcement-learning run rather than only at initialization.
    That is what this does: the reference network is frozen at construction and
    every update pays ``coefficient * KL(current || reference)`` on the states
    the collector actually visited.

    Assumes a discrete output probability

    :param reference: Frozen actor-critic supplying the reference distribution;
        it is deep-copied so later updates to the learner cannot touch it.
    :param coefficient: Weight on the KL term. 0 disables the anchor.
    :param device: Device the reference runs on.
    """

    def __init__(
        self,
        reference: nn.Module,
        coefficient: float,
        device: torch.device | str = "cpu",
    ) -> None:
        super().__init__()
        self._reference = copy.deepcopy(reference).to(device).eval()
        for parameter in self._reference.parameters():
            parameter.requires_grad_(False)
        self._coefficient = float(coefficient)

    @property
    def coefficient(self) -> float:
        """
        :return: Current KL weight.
        """
        return self._coefficient

    #: Fill for illegal actions. A true ``-inf`` gives the right forward value
    #: but a NaN gradient: the illegal term is ``0 * (-inf - -inf)``, and
    #: autograd walks both sides of a ``torch.where`` whichever one is
    #: selected. A large finite value drives the probability to zero while
    #: every gradient stays finite.
    ILLEGAL_LOGIT = -1e9

    @classmethod
    def _masked_log_probs(
        cls, logits: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Log-softmax over the legal actions only.

        The illegal slots carry arbitrary logits, so including them would make
        the two distributions differ on actions neither policy can take.

        :param logits: Raw policy logits, ``(batch, n_actions)``.
        :param mask: Boolean legality mask of the same shape.
        :return: Log-probabilities, finite everywhere.
        """
        return torch.log_softmax(
            logits.masked_fill(~mask, cls.ILLEGAL_LOGIT), dim=-1
        )

    def forward(self, minibatch: TensorDict, policy_logits: torch.Tensor):
        """
        Compute the anchor penalty for one minibatch.

        :param minibatch: Collected transitions, carrying ``action_mask``.
        :param policy_logits: The learner's logits for those states.
        :return: ``(weighted loss, mean KL)``; both zero when disabled.
        """
        zero = policy_logits.new_zeros(())
        if self._coefficient == 0.0:
            return zero, zero
        mask = minibatch.get("action_mask")
        with torch.no_grad():
            reference = self._reference(minibatch.select("observation").clone())
            reference_logits = reference.get("logits").reshape(policy_logits.shape)
            reference_log_probs = self._masked_log_probs(reference_logits, mask)
        log_probs = self._masked_log_probs(policy_logits, mask)
        # Illegal slots hold ~zero probability after the fill, so they add
        # nothing to the sum and every term stays finite.
        contribution = log_probs.exp() * (log_probs - reference_log_probs)
        kl = contribution.sum(dim=-1).mean()
        return self._coefficient * kl, kl.detach()
