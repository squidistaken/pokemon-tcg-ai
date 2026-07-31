from __future__ import annotations

import contextlib
import logging
from collections.abc import Callable, Iterable, Mapping
from typing import Any, cast

import torch
from tensordict import TensorDict
from tensordict.nn import ProbabilisticTensorDictSequential
from torch import nn
from torchrl.data import Categorical
from torchrl.envs import EnvBase
from torchrl.modules import ActorValueOperator
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE

from src.models.actor_critic import ActorCritic
from src.policies.ppo_actor import build_ppo_operator
from src.training.callbacks import TrainingCallback
from src.training.curriculum import Curriculum
from src.training.evaluator import Evaluator
from src.training.loss._helpers import _sum_loss_keys
from src.training.multi_evaluator import MultiEvaluator
from src.training.trainer import Trainer

logger = logging.getLogger(__name__)


class PPOTrainer(Trainer):
    """
    Clipped-PPO trainer with invalid-action masking over the TCG environment.

    Extends :class:`~src.training.trainer.Trainer` by implementing the
    :meth:`_update` hook with the canonical on-policy PPO loop: estimate
    advantages with :class:`~torchrl.objectives.value.GAE`, then run a
    minibatch loop over :class:`~torchrl.objectives.ClipPPOLoss`
    (``loss_objective + loss_critic + loss_entropy``) with gradient clipping.

    The collection policy and both loss networks are views of a single
    shared-trunk :class:`~torchrl.modules.ActorValueOperator`, so the backbone
    runs once per step and its parameters are optimized once despite feeding
    both heads. The wrapped :class:`~src.models.actor_critic.ActorCritic` is
    kept for snapshotting into a self-play pool.

    * GAE is computed once per collected batch (before the epoch loop),
      not re-estimated every epoch as before. This is the more common PPO
      formulation but is a real learning-dynamics change.
    * Minibatching uses a shuffled permutation with contiguous slicing, so
      the final (smaller) minibatch is used;
    """

    def __init__(
            self,
            env_factories: list[Callable[[], EnvBase]],
            actor_critic: ActorCritic,
            action_spec: Categorical,
            frames_per_batch: int,
            total_frames: int,
            clip_epsilon: float = 0.2,
            entropy_bonus: bool = True,
            entropy_coeff: float = 0.01,
            gamma: float = 0.99,
            lmbda: float = 0.95,
            average_gae: bool = True,
            lr: float = 3.0e-4,
            num_epochs: int = 4,
            sub_batch_size: int = 256,
            max_grad_norm: float = 1.0,
            device: torch.device | str = "cpu",
            use_parallel_env: bool = True,
            mp_start_method: str = "fork",
            serial_for_single: bool = True,
            target_kl: float | None = None,
            target_kl_multiplier: float = 1.5,
            use_amp: bool = False,
            compile_loss: bool = False,
            compile_policy: bool = False,
            lr_anneal: bool = False,
            ent_anneal: bool = False,
            ent_warm_frac: float = 0.5,
            reward_scaling: float = 1.0,
            callbacks: Iterable[TrainingCallback] | None = None,
            run_config: Mapping[str, Any] | None = None,
            evaluator: Evaluator | MultiEvaluator | None = None,
            eval_interval: int = 0,
            curriculum: Curriculum | None = None,
    ) -> None:
        """
        :param env_factories: One environment factory per worker.
        :param actor_critic: Shared-trunk actor-critic to train; its modules
            are reused by the collection policy and both loss networks.
        :param action_spec: Action spec used to wire the masked policy.
        :param frames_per_batch: Frames collected per collector iteration.
        :param total_frames: Total frames to collect over the run.
        :param clip_epsilon: PPO surrogate clipping range.
        :param entropy_bonus: Add the entropy term to the loss, rewarding
            higher-entropy policies as a regularizer against premature collapse
            onto a single action. On by default. When ``False`` the term is
            dropped entirely and ``entropy_coeff`` has no effect.
        :param entropy_coeff: Entropy-bonus weight (the annealing start value).
            Ignored unless ``entropy_bonus`` is set.
        :param gamma: Discount factor for GAE.
        :param lmbda: GAE trace-decay factor.
        :param average_gae: Standardize the advantages (subtract the mean,
            divide by the std) over each collected batch. On by default: this is
            the standard PPO formulation and it keeps the surrogate objective's
            scale independent of the reward magnitude.
        :param lr: Adam learning rate (the annealing start value).
        :param num_epochs: Optimization epochs over each collected batch.
        :param sub_batch_size: Minibatch size for the inner epoch loop.
        :param max_grad_norm: Global gradient-norm clipping threshold.
        :param device: Device for optimization tensors.
        :param use_parallel_env: Use ParallelEnv instead of SerialEnv.
        :param mp_start_method: Multiprocessing start method for ParallelEnv.
        :param serial_for_single: Fall back to a single-process env for one worker.
        :param target_kl: If set, stop the epoch loop early once mean
            ``kl_approx`` exceeds ``target_kl_multiplier * target_kl``. Checked
            *between* epochs, so a single bad epoch still fully applies.
        :param target_kl_multiplier: Multiplier applied to ``target_kl`` for the
            early-stop threshold.
        :param use_amp: Enable automatic mixed precision for the update step
            (``float16`` + GradScaler on CUDA, ``bfloat16`` elsewhere). Uses the
            modern ``torch.amp`` API (not the deprecated ``torch.cuda.amp``),
            required because the test suite runs ``filterwarnings=error``.
        :param compile_loss: Wrap the loss module with ``torch.compile``.
        :param compile_policy: Enable ``torch.compile`` on the collection policy
            via the Collector. Defaults ``False`` (compile is slow/fragile on the
            CPU dev box); the colleague's original defaulted it on.
        :param lr_anneal: Linearly anneal the learning rate to 0 over training.
        :param ent_anneal: Anneal the entropy coefficient (see ``ent_warm_frac``).
        :param ent_warm_frac: Fraction of training for which the entropy
            coefficient is held at its initial value before it linearly decays
            to 0. **Inferred schedule** — the flag comes from the colleague's
            file but the schedule lived in an unseen base class.
        :param reward_scaling: Accepted for interface parity. **No effect on the
            current stats**: win/draw rates in :meth:`Trainer.train` are computed
            from the reward *sign*, not its magnitude; this would only matter for
            future magnitude logging.
        :param callbacks: Metric observers, forwarded to
            :class:`~src.training.trainer.Trainer`. The PPO losses returned by
            :meth:`_update` reach them without any extra wiring here.
        :param run_config: Opaque run metadata forwarded to
            :class:`~src.training.trainer.Trainer`.
        :param evaluator: Fixed-opponent evaluator forwarded to
            :class:`~src.training.trainer.Trainer`; required for a readable
            learning curve under self-play.
        :param eval_interval: Frames between evaluation rounds; ``0`` disables.
        :param curriculum: Level curriculum scored from each collected batch and
            republished to the environment workers. None trains on whatever
            distribution the deck sampler already provides.
        """
        self._actor_critic = actor_critic
        self._curriculum = curriculum

        self._operator = cast(
            ActorValueOperator,
            build_ppo_operator(actor_critic, action_spec).to(device),
        )
        super().__init__(
            env_factories=env_factories,
            policy=self._operator.get_policy_operator(),
            frames_per_batch=frames_per_batch,
            total_frames=total_frames,
            use_parallel_env=use_parallel_env,
            mp_start_method=mp_start_method,
            serial_for_single=serial_for_single,
            callbacks=callbacks,
            run_config=run_config,
            evaluator=evaluator,
            eval_interval=eval_interval,
        )
        self._device = torch.device(device)
        self._num_epochs = num_epochs
        self._sub_batch_size = min(sub_batch_size, frames_per_batch)
        self._max_grad_norm = max_grad_norm
        self._target_kl = target_kl
        self._target_kl_multiplier = target_kl_multiplier
        self._reward_scaling = reward_scaling
        self._compile_policy = compile_policy

        self._advantage = GAE(
            gamma=gamma,
            lmbda=lmbda,
            value_network=self._operator.get_value_operator(),
            average_gae=average_gae,
        )
        self._loss = ClipPPOLoss(
            actor_network=cast(
                ProbabilisticTensorDictSequential, self._operator.get_policy_operator()
            ),
            critic_network=self._operator.get_value_operator(),
            clip_epsilon=clip_epsilon,
            entropy_bonus=entropy_bonus,
            entropy_coeff=entropy_coeff,
        )
        # The optimizer is hardcoded here, but there is no real reason for us
        # to change it.
        self._optim = torch.optim.Adam(self._loss.parameters(), lr=lr)

        self._clip_params = list(self._loss.parameters())

        self._loss_fwd = torch.compile(self._loss) if compile_loss else self._loss

        # Annealing state.
        self._lr_anneal = lr_anneal
        # Annealing a coefficient the loss never applies is a silent no-op, so
        # disable it rather than let the run look like it is scheduling entropy.
        if ent_anneal and not entropy_bonus:
            logger.warning(
                "ent_anneal=True has no effect while entropy_bonus=False; "
                "disabling entropy annealing."
            )
            ent_anneal = False
        self._ent_anneal = ent_anneal
        self._ent_warm_frac = ent_warm_frac
        self._initial_lr = lr
        self._initial_entropy_coeff = entropy_coeff
        self._total_updates = max(1, total_frames // frames_per_batch)
        self._updates_done = 0

        # AMP: modern ``torch.amp`` API (not the deprecated ``torch.cuda.amp``).
        if use_amp:
            amp_dtype = torch.float16 if self._device.type == "cuda" else torch.bfloat16
            self._autocast_ctx: contextlib.AbstractContextManager = torch.amp.autocast(
                device_type=self._device.type, dtype=amp_dtype
            )
            self._scaler = (
                torch.amp.GradScaler(self._device.type)
                if self._device.type == "cuda"
                else None
            )
        else:
            self._autocast_ctx = contextlib.nullcontext()
            self._scaler = None

        logger.info(
            "PPOTrainer initialized: device=%s frames_per_batch=%d total_frames=%d "
            "num_epochs=%d sub_batch_size=%d lr=%g gamma=%g lmbda=%g clip_epsilon=%g "
            "entropy_bonus=%s entropy_coeff=%g target_kl=%s use_amp=%s lr_anneal=%s "
            "ent_anneal=%s",
            self._device,
            frames_per_batch,
            total_frames,
            num_epochs,
            self._sub_batch_size,
            lr,
            gamma,
            lmbda,
            clip_epsilon,
            entropy_bonus,
            entropy_coeff,
            target_kl,
            use_amp,
            lr_anneal,
            ent_anneal,
        )

    @property
    def actor_critic(self) -> ActorCritic:
        """
        The actor-critic being trained (for checkpointing / snapshots).

        :return: The wrapped :class:`ActorCritic`.
        """
        return self._actor_critic

    def _collector_kwargs(self) -> dict:
        """
        Collector kwargs for the (possibly GPU-resident) policy.

        ``policy_device`` tells the Collector to cast rollout data onto the
        policy's device for the forward pass and back for env stepping;
        without it, data collected by the CPU-only env workers is fed
        straight into a CUDA policy and errors on the first mismatched
        buffer. The env stays on CPU regardless (``env_device`` unset).

        :return: Mapping splatted into the ``Collector(...)`` construction.
        """
        kwargs: dict = {"policy_device": self._device}
        if self._compile_policy:
            kwargs["compile_policy"] = True
        return kwargs

    def _set_entropy_coeff(self, value: float) -> None:
        """
        Set the loss module's entropy coefficient (annealing helper).

        :param value: New entropy-bonus weight.
        """
        coeff = self._loss.entropy_coeff
        if isinstance(coeff, torch.Tensor):
            with torch.no_grad():
                coeff.fill_(value)
        else:
            self._loss.entropy_coeff = value

    def _maybe_anneal(self) -> None:
        """
        Apply linear LR / entropy-coefficient annealing for the current update.

        Driven by training progress ``updates_done / total_updates``. LR decays
        linearly to 0. The entropy coefficient is held at its initial value for
        the first ``ent_warm_frac`` of training, then decays linearly to 0.

        Schedule shape is **inferred** (see class docstring / arch doc): the
        flags come from the colleague's file, the schedule did not.
        """
        if not (self._lr_anneal or self._ent_anneal):
            return
        progress = min(1.0, self._updates_done / self._total_updates)
        if self._lr_anneal:
            new_lr = self._initial_lr * (1.0 - progress)
            for group in self._optim.param_groups:
                group["lr"] = new_lr
        if self._ent_anneal:
            if self._ent_warm_frac >= 1.0 or progress <= self._ent_warm_frac:
                coeff = self._initial_entropy_coeff
            else:
                decay = (progress - self._ent_warm_frac) / (1.0 - self._ent_warm_frac)
                coeff = self._initial_entropy_coeff * (1.0 - decay)
            self._set_entropy_coeff(coeff)
        logger.debug(
            "Annealing step %d/%d (progress=%.3f): lr=%s entropy_coeff=%s",
            self._updates_done,
            self._total_updates,
            progress,
            self._optim.param_groups[0]["lr"] if self._lr_anneal else "unchanged",
            coeff if self._ent_anneal else "unchanged",
        )

    def _update(self, data: TensorDict) -> dict[str, float] | None:
        """
        Compute advantages once, then run ``num_epochs`` of minibatch PPO updates.

        :param data: One ``(B, T)`` batch from the collector. The stored
            ``action_log_prob`` from collection is the old policy's, as PPO
            requires.
        :return: Mean losses / diagnostics / grad-norm over all applied
            minibatch updates, or ``None`` if every minibatch had a NaN/Inf loss.
        """
        self._maybe_anneal()
        data = data.to(self._device)

        # GAE once per batch (not per epoch): the more common PPO formulation.
        with torch.no_grad():
            self._advantage(data)
            # Before the reshape: the curriculum attributes residuals to
            # episodes, which needs the (workers, time) layout to find episode
            # boundaries within each collector row.
            if self._curriculum is not None:
                self._curriculum.observe(data)
                self._curriculum.publish()
            data_flat = data.reshape(-1)

        batch = data_flat.batch_size[0]
        loss_accum: dict[str, float] = {}
        grad_norm_accum = 0.0
        loss_counts = 0
        skipped_minibatches = 0

        logger.debug(
            "Update %d starting: batch=%d num_epochs=%d sub_batch_size=%d",
            self._updates_done + 1,
            batch,
            self._num_epochs,
            self._sub_batch_size,
        )

        for _ in range(self._num_epochs):
            perm = torch.randperm(batch, device=self._device)
            data_shuffled = data_flat[perm]
            epoch_kl = 0.0
            n_minibatches = 0

            for start in range(0, batch, self._sub_batch_size):
                mb = data_shuffled[start : start + self._sub_batch_size]

                with self._autocast_ctx:
                    loss_vals = self._loss_fwd(mb)
                    total_loss = _sum_loss_keys(loss_vals)

                if not torch.isfinite(total_loss):
                    skipped_minibatches += 1
                    continue

                if self._scaler is not None:
                    self._scaler.scale(total_loss).backward()
                    self._scaler.unscale_(self._optim)
                    grad_norm = nn.utils.clip_grad_norm_(self._clip_params, self._max_grad_norm)
                    self._scaler.step(self._optim)
                    self._scaler.update()
                else:
                    total_loss.backward()
                    grad_norm = nn.utils.clip_grad_norm_(self._clip_params, self._max_grad_norm)
                    self._optim.step()
                self._optim.zero_grad(set_to_none=True)

                # Log only the optimizable ``loss_*`` terms (plus grad-norm
                # below). Diagnostics like ``explained_variance`` are
                # deliberately excluded: they can be non-finite on tiny/near-
                # constant-target batches and would poison a finite-loss check.
                for key, value in loss_vals.items():
                    if (
                        key.startswith("loss_")
                        and isinstance(value, torch.Tensor)
                        and value.numel() == 1
                    ):
                        loss_accum[key] = loss_accum.get(key, 0.0) + float(value.detach())
                grad_norm_accum += float(grad_norm)
                loss_counts += 1

                if self._target_kl is not None and "kl_approx" in loss_vals:
                    epoch_kl += float(loss_vals["kl_approx"])
                    n_minibatches += 1

            if (
                self._target_kl is not None
                and n_minibatches > 0
                and (epoch_kl / n_minibatches) > self._target_kl_multiplier * self._target_kl
            ):
                logger.info(
                    "Early-stopping epoch loop: mean kl_approx=%.4f exceeded "
                    "target_kl_multiplier * target_kl=%.4f",
                    epoch_kl / n_minibatches,
                    self._target_kl_multiplier * self._target_kl,
                )
                break

        self._updates_done += 1

        if skipped_minibatches > 0:
            logger.warning(
                "Skipped %d minibatch update(s) due to NaN/Inf loss (%d applied). "
                "If this persists, the loss is misconfigured and the policy is "
                "not learning.",
                skipped_minibatches,
                loss_counts,
            )
        if loss_counts == 0:
            logger.warning("Update %d produced no applied minibatches; skipping.", self._updates_done)
            return None
        result = {key: value / loss_counts for key, value in loss_accum.items()}
        result["grad_norm"] = grad_norm_accum / loss_counts
        if self._curriculum is not None:
            result.update(self._curriculum.metrics())
        logger.debug("Update %d finished: %s", self._updates_done, result)
        return result
