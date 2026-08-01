from __future__ import annotations

from typing import cast

import torch
from tensordict import TensorDict, is_tensor_collection
from tensordict.nn import TensorDictParams
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.ppo import _sum_td_features
from torchrl.objectives.utils import _reduce


class DiscoPPOLoss(ClipPPOLoss):
    """Discovered Policy Optimization surrogate objective (Lu et al., NeurIPS 2022).

    Replaces the clipped PPO actor loss with the piecewise surrogate::

        f(r, A) = ReLU((r-1)*A - α*tanh((r-1)*A/α))   if A >= 0
                  ReLU(log(r)*A - β*tanh(log(r)*A/β))  if A <  0

    Critic loss, entropy bonus, and advantage normalisation are inherited
    from ``ClipPPOLoss`` unchanged.

    :param disco_alpha: Smoothness parameter for the positive-advantage branch.
        Larger values approach plain PPO clipping.
    :param disco_beta: Smoothness parameter for the negative-advantage branch.
    """

    actor_network_params: TensorDictParams
    critic_network_params: TensorDictParams
    target_actor_network_params: TensorDictParams
    target_critic_network_params: TensorDictParams

    def __init__(
        self,
        *args,
        disco_alpha: float = 2.0,
        disco_beta: float = 0.6,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.disco_alpha = disco_alpha
        self.disco_beta = disco_beta

    def forward(self, tensordict):
        tensordict = tensordict.clone(False)

        # ── Advantage ─────────────────────────────────────────────────
        advantage = tensordict.get(self.tensor_keys.advantage, None, as_padded_tensor=True)
        if advantage is None:
            self.value_estimator(
                tensordict,
                params=self._cached_critic_network_params_detached,
                target_params=self.target_critic_network_params,
            )
            advantage = tensordict.get(self.tensor_keys.advantage)
        if self.normalize_advantage and advantage.numel() > 1:
            loc = advantage.mean()
            scale = advantage.std().clamp_min(1e-8)
            advantage = (advantage - loc) / scale

        # ── IS ratio ──────────────────────────────────────────────────
        log_weight, dist, kl_approx = self._log_weight(
            tensordict, adv_shape=advantage.shape[:-1]
        )
        log_ratio = log_weight.squeeze(-1)
        ratio = log_ratio.exp()
        adv = advantage.squeeze(-1)

        # ── DiscoPO piecewise surrogate ───────────────────────────────
        alpha, beta = self.disco_alpha, self.disco_beta

        u = (ratio - 1.0) * adv
        f_pos = torch.relu(u - alpha * torch.tanh(u / alpha))

        v = log_ratio * adv
        f_neg = torch.relu(v - beta * torch.tanh(v / beta))

        clip_penalty = torch.where(adv >= 0, f_pos, f_neg)
        gain = ratio * adv - clip_penalty

        # ── ESS ───────────────────────────────────────────────────────
        with torch.no_grad():
            lw = log_weight.squeeze()
            ess = (2 * lw.logsumexp(0) - (2 * lw).logsumexp(0)).exp()
            batch = log_weight.shape[0]

        td_out = TensorDict({"loss_objective": -gain})
        td_out.set("clip_fraction", torch.zeros((), device=gain.device))
        td_out.set("kl_approx", kl_approx.detach().mean())

        # ── Entropy bonus ─────────────────────────────────────────────
        if self.entropy_bonus:
            entropy = self._get_entropy(dist, adv_shape=advantage.shape[:-1])
            if is_tensor_collection(entropy):
                td_out.set("composite_entropy", entropy.detach())
                td_out.set("entropy", _sum_td_features(entropy).detach().mean())
            else:
                td_out.set("entropy", entropy.detach().mean())
            td_out.set("loss_entropy", self._weighted_loss_entropy(entropy))

        # ── Critic loss ───────────────────────────────────────────────
        if self._has_critic:
            loss_critic, value_clip_fraction, explained_variance = self.loss_critic(tensordict)
            td_out.set("loss_critic", loss_critic)
            if value_clip_fraction is not None:
                td_out.set("value_clip_fraction", value_clip_fraction)
            if explained_variance is not None:
                td_out.set("explained_variance", explained_variance)

        td_out.set("ESS", _reduce(ess, self.reduction) / batch)
        td_out = td_out.named_apply(
            lambda name, value: cast(torch.Tensor, _reduce(value, reduction=self.reduction)).squeeze(-1)
            if name.startswith("loss_")
            else value,
        )
        self._clear_weakrefs(
            tensordict, td_out,
            "actor_network_params", "critic_network_params",
            "target_actor_network_params", "target_critic_network_params",
        )
        return td_out
