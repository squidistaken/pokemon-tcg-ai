import logging

from torchrl import _utils as torchrl_utils

logger = logging.getLogger(__name__)

#: torchrl's own default, in seconds (2h47m). Kept here so callers can report
#: what they are overriding without importing torchrl internals themselves.
TORCHRL_DEFAULT_PIPE_TIMEOUT = 10000.0


def apply_pipe_timeout(seconds: float | None) -> None:
    """
    Shorten how long a ``ParallelEnv`` pool waits on itself before raising.

    Both sides of the pool read ``BATCHED_PIPE_TIMEOUT`` when the workers start
    -- the parent in ``_BatchedEnv._start_workers``, each worker inside
    ``_run_worker_pipe_shared_mem`` -- so setting the module attribute before
    the pool is built, and therefore before the fork that copies it into every
    worker, reaches both. Setting the environment variable of the same name
    would not: torchrl reads it once, at import.

    The default is 10000 seconds. When one worker stops answering, the parent
    spins on its completion flag while its siblings block waiting for their next
    command, so nothing reports the fault until that timer expires; the
    collector's restart budget then rebuilds the pool in seconds. The timeout is
    the cost of *noticing* a fault, not of recovering from it, and 10000 seconds
    of it turned two recoverable incidents into 5.6 idle hours of an 8.9-hour
    run.

    :param seconds: New timeout. ``None`` opts out, leaving torchrl's default in
        place; any number must be positive.
    :raises ValueError: If ``seconds`` is zero or negative. Such a value is
        rejected rather than read as an opt-out, because a config typo would
        then silently restore the 2h47m wait this exists to remove -- and a
        literal zero-second timeout, which is what the number looks like it
        asks for, would fail the pool on its first step.
    """
    if seconds is None:
        return
    if seconds <= 0:
        raise ValueError(
            f"pipe_timeout must be positive or None, got {seconds}. Use None to "
            f"keep torchrl's {TORCHRL_DEFAULT_PIPE_TIMEOUT:.0f}s default."
        )
    torchrl_utils.BATCHED_PIPE_TIMEOUT = float(seconds)
    logger.debug(
        "Worker-pool timeout set to %.0fs (torchrl default is %.0fs).",
        seconds,
        TORCHRL_DEFAULT_PIPE_TIMEOUT,
    )
