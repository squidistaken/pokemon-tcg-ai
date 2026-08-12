from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class CurriculumHandles:
    """
    Shared-memory channel carrying the level distribution to the env workers.

    The level buffer lives in the learner, because the score it maintains is a
    critic residual the environments cannot see. The workers only need the
    result: which matchup to deal next. That is published here as three tensors
    in shared memory, written by the learner once per batch and read by every
    worker at each episode reset.

    Shared tensors rather than a :class:`multiprocessing.Manager` proxy: a
    proxy routes every element access through a server process, whereas these
    are mapped into each worker and read like ordinary memory. Under the
    ``fork`` start method the mapping is inherited, so nothing is pickled.

    A worker may read while the learner is writing. The consequence is that one
    episode is sampled from a half-updated distribution, which is harmless: the
    weights are a heuristic and the next reset sees the finished write. Nothing
    is torn in a way that can produce an invalid index, because ``size`` is
    only ever raised after the slots below it are populated.

    :param probabilities: Sampling probability per buffer slot, shape
        ``(capacity,)``. Only the first ``size`` entries are meaningful.
    :param pair_ids: Matchup identifier occupying each slot, shape
        ``(capacity,)``. Decoded with
        :meth:`~src.curriculum.archetype_index.ArchetypeIndex.unpair`.
    :param size: Number of populated slots, shape ``(1,)``.
    """

    probabilities: torch.Tensor
    pair_ids: torch.Tensor
    size: torch.Tensor

    @classmethod
    def allocate(cls, capacity: int) -> "CurriculumHandles":
        """
        Allocate an empty channel in shared memory.

        :param capacity: Maximum number of buffer slots.
        :return: Handles whose tensors are shared with any forked worker.
        :raises ValueError: If ``capacity`` is not positive.
        """
        if capacity <= 0:
            raise ValueError(f"capacity must be positive, got {capacity}")
        return cls(
            probabilities=torch.zeros(capacity, dtype=torch.float32).share_memory_(),
            pair_ids=torch.zeros(capacity, dtype=torch.int64).share_memory_(),
            size=torch.zeros(1, dtype=torch.int64).share_memory_(),
        )

    @staticmethod
    def require_shared_start_method(start_method: str) -> None:
        """
        Reject a worker start method that would break the shared channel.

        Only ``fork`` hands the mapping to the child. Under ``spawn`` or
        ``forkserver`` the tensors are pickled, so every worker gets a private
        copy: rollouts still run, but the curriculum is silently frozen at
        whatever distribution existed when the workers started. That failure is
        invisible in the metrics, so it is worth refusing up front.

        This validates the method a :class:`~torchrl.envs.ParallelEnv` is
        actually built with, which is a constructor argument and need not match
        the interpreter's global default.

        :param start_method: Configured multiprocessing start method.
        :raises ValueError: If it is anything other than ``fork``.
        """
        if start_method != "fork":
            raise ValueError(
                f"the level curriculum needs env.mp_start_method='fork' to share its "
                f"distribution with the environment workers, got {start_method!r}. "
                f"Under {start_method!r} each worker would receive a private copy and "
                f"never see a curriculum update."
            )

    @property
    def capacity(self) -> int:
        """
        Number of slots the channel can hold.

        :return: The allocated capacity.
        """
        return int(self.probabilities.shape[0])

    def publish(self, pair_ids: torch.Tensor, probabilities: torch.Tensor) -> None:
        """
        Overwrite the channel with a new distribution.

        ``size`` is written last, so a worker reading concurrently never sees a
        slot count that outruns the slots backing it.

        :param pair_ids: Matchup identifier per slot, shape ``(n,)``.
        :param probabilities: Sampling probability per slot, shape ``(n,)``.
        :raises ValueError: If the two disagree in length or exceed capacity.
        """
        if pair_ids.shape != probabilities.shape:
            raise ValueError(
                f"pair_ids {tuple(pair_ids.shape)} and probabilities "
                f"{tuple(probabilities.shape)} must have the same shape"
            )
        count = int(pair_ids.shape[0])
        if count > self.capacity:
            raise ValueError(
                f"cannot publish {count} slots into a channel of capacity {self.capacity}"
            )
        self.pair_ids[:count] = pair_ids.to(torch.int64)
        self.probabilities[:count] = probabilities.to(torch.float32)
        self.size[0] = count
