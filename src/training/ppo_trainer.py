from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any, cast

import torch
from tensordict import TensorDict
from tensordict.nn import ProbabilisticTensorDictSequential
from torch import nn
from torchrl.data import Categorical
from torchrl.envs import EnvBase
from torchrl.modules import ActorValueOperator
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE, ValueEstimatorBase, VTrace

from src.curriculum.curriculum import Curriculum
from src.models.actor_critic import ActorCritic
from src.policies.ppo_actor import build_ppo_operator
from src.training.callbacks import TrainingCallback, TrainStateCallback
from src.training.collectors import (
    AsyncCollectorOptions,
    CollectorKind,
    is_off_policy,
    parse_collector_kind,
)
from src.training.evaluator import Evaluator
from src.training.loss._helpers import _sum_loss_keys
from src.training.multi_evaluator import MultiEvaluator
from src.training.trainer import Trainer

logger = logging.getLogger(__name__)

#: Non-optimizable diagnostics reported alongside the ``loss_*`` terms, none of
#: them inferable from the losses: unweighted policy ``entropy``, how much of
#: the batch the surrogate clips, how degenerate the importance weights are,
#: how far the epochs drift off-policy, and whether the critic predicts returns.
#:
#: Each is averaged over the minibatches where it came back finite, and is
#: absent from the metrics when none did. That per-key guard is why they stay
#: out of the ``loss_``-prefixed terms and the finite-loss check.
_LOGGED_DIAGNOSTICS = (
    "entropy",
    "clip_fraction",
    "ESS",
    "kl_approx",
    "explained_variance",
)

#: Advantage estimators the trainer can be configured with. ``gae`` assumes the
#: batch came from one behaviour policy, which ``collector.type=sync``
#: guarantees. ``vtrace`` does not: it re-scores the collected actions under the
#: current policy and clips the importance ratios, correcting a batch collected
#: while the weights moved underneath it.
_VALUE_ESTIMATORS = ("gae", "vtrace")


def _reject_unstable_curriculum_rows(
    collector_type: str | CollectorKind, curriculum: Curriculum | None
) -> None:
    """
    Refuse the one collector/curriculum pairing that silently misattributes scores.

    :class:`~src.curriculum.curriculum.Curriculum` accumulates a residual per
    *collector row*, which assumes row ``r`` of the next batch continues the
    same environment's episode as row ``r`` of this one. ``sync`` and
    ``multi_sync`` stack workers in a fixed order and honour that.

    ``multi_async`` yields whichever worker finished first, so row identity
    shuffles between batches and one matchup's residuals land on another's
    score. ``traj_ids`` cannot repair it: they identify a trajectory, not a
    worker. That is a silent scientific error rather than a crash, and
    ``multi_sync`` measured faster anyway, so the pairing is rejected outright.

    :param collector_type: Configured collector kind.
    :param curriculum: The curriculum, if one is active.
    :raises ValueError: If a curriculum is paired with ``multi_async``.
    """
    if curriculum is None:
        return
    if parse_collector_kind(collector_type) is not CollectorKind.MULTI_ASYNC:
        return
    raise ValueError(
        "collector.type=multi_async cannot be combined with the level curriculum: "
        "it yields rollouts in completion order, so the curriculum's per-row "
        "episode accounting would attribute one matchup's residuals to another. "
        "Use collector.type=multi_sync (faster in every measurement anyway), or "
        "disable the curriculum with env.curriculum.enabled=false."
    )


class PPOTrainer(Trainer):
    """
    Clipped-PPO trainer with invalid-action masking over the TCG environment.

    Implements :class:`~src.training.trainer.Trainer`'s :meth:`_update` hook
    with the on-policy PPO loop: estimate advantages once per collected batch,
    then run a minibatch loop over :class:`~torchrl.objectives.ClipPPOLoss`
    (``loss_objective + loss_critic + loss_entropy``) with gradient clipping.
    Minibatching is a shuffled permutation with contiguous slicing, so the final
    smaller minibatch is used.

    The collection policy and both loss networks are views of a single
    shared-trunk :class:`~torchrl.modules.ActorValueOperator`, so the backbone
    runs once per step and is optimized once despite feeding both heads. The
    wrapped :class:`~src.models.actor_critic.ActorCritic` is kept for
    snapshotting into a self-play pool.
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
        gae_num_chunks: int | None = None,
        value_estimator: str = "gae",
        vtrace_rho_thresh: float = 1.0,
        vtrace_c_thresh: float = 1.0,
        lr: float = 3.0e-4,
        num_epochs: int = 4,
        sub_batch_size: int = 256,
        max_grad_norm: float = 1.0,
        device: torch.device | str = "cpu",
        collector_device: torch.device | str | None = None,
        use_parallel_env: bool = True,
        mp_start_method: str = "fork",
        serial_for_single: bool = True,
        target_kl: float | None = None,
        target_kl_multiplier: float = 1.5,
        compile_loss: bool = False,
        compile_policy: bool = False,
        lr_anneal: bool = False,
        ent_anneal: bool = False,
        ent_warm_frac: float = 0.5,
        callbacks: Iterable[TrainingCallback] | None = None,
        run_config: Mapping[str, Any] | None = None,
        evaluator: Evaluator | MultiEvaluator | None = None,
        eval_interval: int = 0,
        curriculum: Curriculum | None = None,
        max_collector_restarts: int = 0,
        rebuild_env_factories: Callable[[int], list[Callable[[], EnvBase]]]
        | None = None,
        pipe_timeout: float | None = None,
        start_frames: int = 0,
        collector_type: str | CollectorKind = CollectorKind.SYNC,
        async_options: AsyncCollectorOptions | None = None,
        train_state_path: str | Path | None = None,
        train_state_interval: int = 0,
        resume_state: Mapping[str, Any] | None = None,
    ) -> None:
        """
        :param env_factories: One environment factory per worker.
        :param actor_critic: Shared-trunk actor-critic to train; its modules
            are reused by the collection policy and both loss networks.
        :param action_spec: Action spec used to wire the masked policy.
        :param frames_per_batch: Frames collected per collector iteration.
        :param total_frames: Total frames to collect over the run.
        :param clip_epsilon: PPO surrogate clipping range.
        :param entropy_bonus: Add the entropy term to the loss, as a regularizer
            against collapse onto a single action. When ``False`` the term is
            dropped and ``entropy_coeff`` has no effect.
        :param entropy_coeff: Entropy-bonus weight (the annealing start value).
            Ignored unless ``entropy_bonus`` is set.
        :param gamma: Discount factor for GAE.
        :param lmbda: GAE trace-decay factor.
        :param average_gae: Standardize the advantages over each collected batch,
            which keeps the surrogate's scale independent of the reward
            magnitude.
        :param gae_num_chunks: Split the GAE critic pass into this many chunks
            along the worker dimension. ``None`` runs the batch in one ``vmap``
            over a stacked current/next pair, peaking at ~2x the batch, which
            exhausts VRAM at ``frames_per_batch`` 16384. Chunking is exact:
            worker rows are independent trajectories and GAE reduces along
            time.
        :param value_estimator: Advantage estimator, one of
            :data:`_VALUE_ESTIMATORS`. ``gae`` is correct only when the whole
            batch came from one behaviour policy. ``vtrace`` re-scores the
            collected actions under the current policy and clips the importance
            ratios, which is what makes a stale asynchronous batch usable, at
            one extra actor pass per batch. It ignores ``lmbda``.
        :param vtrace_rho_thresh: V-trace's rho-bar, bounding the importance
            ratio in the TD term. It sets which policy's value function the
            critic converges to: ``1.0`` (the IMPALA default) is the behaviour
            policy's, and raising it trades variance for a target nearer the
            current policy.
        :param vtrace_c_thresh: V-trace's c-bar, the ceiling on the ratio inside
            the trace. It controls how far a correction propagates back in time,
            and so the variance of the estimate, without moving the fixed point.
        :param lr: Adam learning rate (the annealing start value).
        :param num_epochs: Optimization epochs over each collected batch.
        :param sub_batch_size: Minibatch size for the inner epoch loop.
        :param max_grad_norm: Global gradient-norm clipping threshold.
        :param device: Device for optimization tensors.
        :param collector_device: Device the *collection* policy runs on. ``None``
            follows ``device``, which is right for ``collector.type=sync``. Set
            it to ``cpu`` alongside ``device=cuda`` under the multiprocess
            collectors, which would otherwise put a CUDA context in every worker
            for batch-size-1 forwards.
        :param use_parallel_env: Use ParallelEnv instead of SerialEnv.
        :param mp_start_method: Multiprocessing start method for ParallelEnv.
        :param serial_for_single: Fall back to a single-process env for one worker.
        :param target_kl: If set, stop the epoch loop early once mean
            ``kl_approx`` exceeds ``target_kl_multiplier * target_kl``. Checked
            *between* epochs, so a single bad epoch still fully applies.
        :param target_kl_multiplier: Multiplier applied to ``target_kl`` for the
            early-stop threshold.
        :param compile_loss: Wrap the loss module with ``torch.compile``.
        :param compile_policy: Enable ``torch.compile`` on the collection policy
            via the Collector. Off by default: compile is slow and fragile on
            the CPU dev box.
        :param lr_anneal: Linearly anneal the learning rate to 0 over training.
        :param ent_anneal: Anneal the entropy coefficient (see ``ent_warm_frac``).
        :param ent_warm_frac: Fraction of training for which the entropy
            coefficient is held at its initial value before it decays linearly
            to 0.
        :param callbacks: Metric observers, forwarded to
            :class:`~src.training.trainer.Trainer`.
        :param run_config: Opaque run metadata forwarded to
            :class:`~src.training.trainer.Trainer`.
        :param evaluator: Fixed-opponent evaluator forwarded to
            :class:`~src.training.trainer.Trainer`; required for a readable
            learning curve under self-play.
        :param eval_interval: Frames between evaluation rounds; ``0`` disables.
        :param curriculum: Level curriculum scored from each collected batch and
            republished to the environment workers. None trains on whatever
            distribution the deck sampler already provides.
        :param max_collector_restarts: Dead worker pools to survive, forwarded
            to :class:`~src.training.trainer.Trainer`.
        :param rebuild_env_factories: Per-restart factory builder, forwarded to
            :class:`~src.training.trainer.Trainer`.
        :param pipe_timeout: Worker-pool detection timeout, forwarded to
            :class:`~src.training.trainer.Trainer`. ``None`` keeps torchrl's
            default.
        :param start_frames: Frames inherited from a warm-start checkpoint,
            forwarded to :class:`~src.training.trainer.Trainer`. Reporting only;
            ``total_frames`` still counts the frames this run collects.
        :param collector_type: Collector kind, forwarded to
            :class:`~src.training.trainer.Trainer`. Anything other than ``sync``
            collects off-policy to some degree, which is what
            ``value_estimator="vtrace"`` corrects for.
        :param async_options: Asynchronous-collector settings, forwarded to
            :class:`~src.training.trainer.Trainer`.
        :param train_state_path: Rolling file the optimizer state is written to,
            so an interrupted run can be continued without restarting Adam's
            moments. None writes no training state.
        :param train_state_interval: Frames between training-state writes.
            ``0`` writes only at the end of the run.
        :param resume_state: Optimizer state dict from a previous run's training
            state, loaded into the fresh optimizer. None starts Adam cold, which
            is what a warm start from a weights-only snapshot must do.
        """
        self._actor_critic = actor_critic
        self._curriculum = curriculum
        _reject_unstable_curriculum_rows(collector_type, curriculum)

        self._operator = cast(
            ActorValueOperator,
            build_ppo_operator(actor_critic, action_spec).to(device),
        )
        # Nothing downstream reads `hidden`, `option_repr` or `logits` back
        # from the collected batch, yet `option_repr` alone is ~64.5 KiB/frame
        # (~258 MiB per 4096-frame batch), which `_update` then copies again
        # per epoch. Trimming at the policy boundary also skips the Collector's
        # preallocation, which a `postproc=ExcludeTransform` would not.
        # Safe because `get_policy_operator()` returns a fresh wrapper over the
        # shared submodules: GAE and ClipPPOLoss build their own, and the
        # pointer head reads `option_repr` within this same forward.
        collection_policy = self._operator.get_policy_operator().select_out_keys(
            "action", "action_log_prob"
        )
        super().__init__(
            env_factories=env_factories,
            policy=collection_policy,
            frames_per_batch=frames_per_batch,
            total_frames=total_frames,
            use_parallel_env=use_parallel_env,
            mp_start_method=mp_start_method,
            serial_for_single=serial_for_single,
            callbacks=callbacks,
            run_config=run_config,
            evaluator=evaluator,
            eval_interval=eval_interval,
            max_collector_restarts=max_collector_restarts,
            rebuild_env_factories=rebuild_env_factories,
            pipe_timeout=pipe_timeout,
            start_frames=start_frames,
            collector_type=collector_type,
            async_options=async_options,
        )
        self._device = torch.device(device)
        self._collector_device = (
            self._device if collector_device is None else torch.device(collector_device)
        )
        self._num_epochs = num_epochs
        self._sub_batch_size = min(sub_batch_size, frames_per_batch)
        self._max_grad_norm = max_grad_norm
        self._target_kl = target_kl
        self._target_kl_multiplier = target_kl_multiplier
        self._compile_policy = compile_policy

        self._value_estimator_name = value_estimator
        if is_off_policy(self._collector_kind) and value_estimator == "gae":
            logger.warning(
                "collector.type=%s keeps collecting through the update, but "
                "agent.value_estimator=gae assumes every action in a batch came from "
                "the weights currently loaded. The importance ratios PPO clips will "
                "be biased by however far the policy moved during collection; "
                "agent.value_estimator=vtrace is what corrects that.",
                self._collector_kind.value,
            )
        self._advantage = self._build_advantage(
            value_estimator=value_estimator,
            gamma=gamma,
            lmbda=lmbda,
            average_gae=average_gae,
            num_chunks=gae_num_chunks,
            rho_thresh=vtrace_rho_thresh,
            c_thresh=vtrace_c_thresh,
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
        if resume_state is not None:
            self._optim.load_state_dict(resume_state)
            logger.info(
                "Restored optimizer state; Adam's moments continue rather than "
                "restarting from zero."
            )
        # Attached here rather than in the caller because the optimizer whose
        # state it preserves does not exist until this point.
        if train_state_path is not None:
            self._callbacks.append(
                TrainStateCallback(
                    actor_critic=actor_critic,
                    optimizer=self._optim,
                    path=train_state_path,
                    interval=train_state_interval,
                )
            )

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

        logger.info(
            "PPOTrainer initialized: device=%s frames_per_batch=%d total_frames=%d "
            "num_epochs=%d sub_batch_size=%d lr=%g gamma=%g lmbda=%g clip_epsilon=%g "
            "entropy_bonus=%s entropy_coeff=%g target_kl=%s lr_anneal=%s "
            "ent_anneal=%s collector=%s value_estimator=%s",
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
            lr_anneal,
            ent_anneal,
            self._collector_kind.value,
            value_estimator,
        )

    @property
    def actor_critic(self) -> ActorCritic:
        """
        The actor-critic being trained (for checkpointing / snapshots).

        :return: The wrapped :class:`ActorCritic`.
        """
        return self._actor_critic

    def _build_advantage(
        self,
        *,
        value_estimator: str,
        gamma: float,
        lmbda: float,
        average_gae: bool,
        num_chunks: int | None,
        rho_thresh: float,
        c_thresh: float,
    ) -> ValueEstimatorBase:
        """
        Build the configured advantage estimator over the shared value operator.

        Both estimators are constructed against fresh views of the same
        shared-trunk operator, exactly as the loss module is, so nothing is
        duplicated and the critic they read is the one being optimized. V-trace
        additionally takes the actor, because correcting for off-policy drift
        means re-evaluating the collected actions under the current policy --
        which is precisely what GAE has no way to do.

        :param value_estimator: Name from :data:`_VALUE_ESTIMATORS`.
        :param gamma: Discount factor.
        :param lmbda: Trace-decay factor; GAE only.
        :param average_gae: Standardize the advantages over the batch.
        :param num_chunks: Chunks for the critic pass along the worker dimension.
        :param rho_thresh: V-trace rho-bar.
        :param c_thresh: V-trace c-bar.
        :return: The estimator, called once per collected batch in
            :meth:`_update`.
        :raises ValueError: If ``value_estimator`` is not a known name.
        """
        if value_estimator == "gae":
            return GAE(
                gamma=gamma,
                lmbda=lmbda,
                value_network=self._operator.get_value_operator(),
                average_gae=average_gae,
                num_chunks=num_chunks,
            )
        if value_estimator == "vtrace":
            logger.info(
                "Using V-trace (rho_thresh=%g, c_thresh=%g); agent.lmbda=%g is "
                "unused, V-trace has no trace-decay parameter.",
                rho_thresh,
                c_thresh,
                lmbda,
            )
            return VTrace(
                gamma=gamma,
                actor_network=cast(
                    ProbabilisticTensorDictSequential,
                    self._operator.get_policy_operator(),
                ),
                value_network=self._operator.get_value_operator(),
                rho_thresh=rho_thresh,
                c_thresh=c_thresh,
                average_adv=average_gae,
                num_chunks=num_chunks,
            )
        known = ", ".join(_VALUE_ESTIMATORS)
        raise ValueError(
            f"Unknown agent.value_estimator '{value_estimator}'; expected one of: "
            f"{known}."
        )

    def _collector_kwargs(self) -> dict:
        """
        Collector kwargs for the (possibly GPU-resident) policy.

        ``policy_device`` tells the Collector to cast rollout data onto the
        policy's device for the forward pass and back for env stepping;
        without it, data collected by the CPU-only env workers is fed
        straight into a CUDA policy and errors on the first mismatched
        buffer. The env stays on CPU regardless (``env_device`` unset).

        This is deliberately *not* pinned to :attr:`_device`. Under
        ``collector.type=multi_sync`` the two want opposite answers: collection
        runs a policy copy inside every worker, which must stay on CPU or open
        one CUDA context per worker for batch-size-1 forwards, while the update
        is a large batched pass that CUDA is worth ~7.8x on
        (``docs/training-performance.md`` section 2). ``agent.collector_device``
        is what lets a run have both.

        :return: Mapping splatted into the collector construction.
        """
        kwargs: dict = {"policy_device": self._collector_device}
        if self._compile_policy:
            kwargs["compile_policy"] = True
        return kwargs

    def _prepare_restart(self, restart_index: int) -> None:
        """
        Drop the curriculum's half-finished episodes before the pool is rebuilt.

        The curriculum accumulates a residual sum per collector row and commits
        it when that row reports ``done``. A replacement pool starts every row
        on a fresh battle, so the residuals banked against the dead pool belong
        to games that will never finish. Left in place they would be committed
        under the *next* episode's level, scoring a matchup with another
        matchup's evidence.

        :param restart_index: 1-based index of the restart about to happen.
        """
        super()._prepare_restart(restart_index)
        if self._curriculum is not None:
            self._curriculum.abandon_open_episodes()

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
        :return: Mean losses, grad-norm and the :data:`_LOGGED_DIAGNOSTICS`
            over all applied minibatch updates, or ``None`` if every minibatch
            had a NaN/Inf loss.
        """
        self._maybe_anneal()
        data = data.to(self._device)

        # Advantages once per batch (not per epoch): the more common PPO
        # formulation. Under V-trace this is also the only point at which the
        # collected actions are re-scored against the current policy.
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
        diagnostic_accum: dict[str, float] = {}
        diagnostic_counts: dict[str, int] = {}
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
            epoch_kl = 0.0
            n_minibatches = 0

            for start in range(0, batch, self._sub_batch_size):
                # Indexed per minibatch: materialising data_flat[perm] clones the
                # whole batch on the GPU once per epoch (docs/wsl-crash-diagnosis.md).
                mb = data_flat[perm[start : start + self._sub_batch_size]]

                loss_vals = self._loss_fwd(mb)
                total_loss = _sum_loss_keys(loss_vals)

                if not torch.isfinite(total_loss):
                    skipped_minibatches += 1
                    continue

                total_loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(
                    self._clip_params, self._max_grad_norm
                )
                self._optim.step()
                self._optim.zero_grad(set_to_none=True)

                # The optimizable ``loss_*`` terms, unconditionally: they are
                # the ones that just passed the finite-loss check above. The
                # non-optimizable diagnostics are accumulated separately below,
                # because they need a per-key finiteness guard.
                for key, value in loss_vals.items():
                    if (
                        key.startswith("loss_")
                        and isinstance(value, torch.Tensor)
                        and value.numel() == 1
                    ):
                        loss_accum[key] = loss_accum.get(key, 0.0) + float(
                            value.detach()
                        )
                grad_norm_accum += float(grad_norm)
                loss_counts += 1

                # The non-optimizable reads on PPO's health, each guarded on
                # finiteness so a single bad batch drops one sample rather than
                # poisoning the average; see _LOGGED_DIAGNOSTICS.
                for key in _LOGGED_DIAGNOSTICS:
                    value = loss_vals.get(key)
                    if (
                        isinstance(value, torch.Tensor)
                        and value.numel() == 1
                        and torch.isfinite(value)
                    ):
                        diagnostic_accum[key] = diagnostic_accum.get(key, 0.0) + float(
                            value.detach()
                        )
                        diagnostic_counts[key] = diagnostic_counts.get(key, 0) + 1

                if self._target_kl is not None and "kl_approx" in loss_vals:
                    epoch_kl += float(loss_vals["kl_approx"])
                    n_minibatches += 1

            if (
                self._target_kl is not None
                and n_minibatches > 0
                and (epoch_kl / n_minibatches)
                > self._target_kl_multiplier * self._target_kl
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
            logger.warning(
                "Update %d produced no applied minibatches; skipping.",
                self._updates_done,
            )
            return None
        result = {key: value / loss_counts for key, value in loss_accum.items()}
        result["grad_norm"] = grad_norm_accum / loss_counts
        result.update(
            {
                key: total / diagnostic_counts[key]
                for key, total in diagnostic_accum.items()
            }
        )
        if self._curriculum is not None:
            result.update(self._curriculum.metrics())
        logger.debug("Update %d finished: %s", self._updates_done, result)
        return result
