import gc
import multiprocessing
import os
import sys
import weakref

import pytest

from src.training.callbacks.wandb_fork_guard import WandbForkGuard

#: Holds the finalized object across the fork, so the child can drop it the way
#: torchrl's worker drops W&B's service handle when it collects garbage.
_INHERITED: list[object] = []


def _blocking_cleanup() -> None:
    """
    Stand-in for W&B's finalizer, which waits on a reply that never arrives.

    The real hook waits on a ``concurrent.futures`` future that is fulfilled by
    an asyncio thread fork did not copy into the child. Blocking outright
    reproduces the same observable behaviour without reaching into W&B
    internals.
    """
    while True:
        pass


class _Holder:
    """
    Object a finalizer attaches to, standing in for W&B's service handle.

    ``peer`` is pointed back at the instance so that only a cycle collection
    reclaims it, which is what makes ``gc.collect`` -- rather than a refcount
    drop -- the trigger, as it is in torchrl's worker.
    """

    peer: "_Holder | None" = None


def _collect_in_child() -> None:
    """
    Child-side body mirroring torchrl's worker ``close`` handler.

    ``_run_worker_pipe_shared_mem`` ends its close branch with an explicit
    ``gc.collect()`` (``batched_envs.py:3014``), and that is where the inherited
    finalizer fires -- inside the handler, before the worker ever reaches
    interpreter shutdown. A worker stuck here never returns, so the parent's
    unbounded ``Process.join`` never returns either.
    """
    _INHERITED.clear()
    gc.collect()


def _child_exits_cleanly(install_guard: bool) -> bool:
    """
    Fork a child holding an inherited blocking finalizer and see whether it exits.

    The finalizer is registered before the fork, exactly as ``wandb.init``
    leaves one behind before ``ParallelEnv`` starts its workers.

    :param install_guard: Register :class:`WandbForkGuard` before forking.
    :return: True if the child exited within the grace period.
    """
    holder = _Holder()
    holder.peer = holder
    finalizer = weakref.finalize(holder, _blocking_cleanup)
    _INHERITED.append(holder)
    del holder
    if install_guard:
        WandbForkGuard(module_prefix=__name__).install()

    process = multiprocessing.get_context("fork").Process(target=_collect_in_child)
    process.start()
    process.join(timeout=10.0)
    exited = not process.is_alive()
    if not exited:
        process.kill()
        process.join(timeout=10.0)

    finalizer.detach()
    _INHERITED.clear()
    gc.collect()
    return exited


@pytest.mark.skipif(
    sys.platform == "win32", reason="fork start method is unavailable on Windows"
)
def test_inherited_finalizer_hangs_the_child() -> None:
    """
    The bug: a forked child wedges when it collects the inherited handle.
    """
    assert _child_exits_cleanly(install_guard=False) is False


@pytest.mark.skipif(
    sys.platform == "win32", reason="fork start method is unavailable on Windows"
)
def test_guard_lets_the_child_exit() -> None:
    """
    With the guard installed the child detaches the hook and exits normally.
    """
    assert _child_exits_cleanly(install_guard=True) is True


def test_guard_leaves_unrelated_finalizers_armed() -> None:
    """
    Only finalizers defined in the guarded package are detached.
    """
    calls: list[str] = []

    class _Other:
        pass

    other = _Other()
    weakref.finalize(other, calls.append, "ran")

    WandbForkGuard(module_prefix="some.other.package").disarm()

    del other
    assert calls == ["ran"]


def test_install_is_idempotent() -> None:
    """
    Repeat installs register the fork handler once, not once per call.
    """
    guard = WandbForkGuard(module_prefix="nothing.matches.this")
    registered: list[object] = []
    original = os.register_at_fork

    def _record(**kwargs: object) -> None:
        registered.append(kwargs)

    os.register_at_fork = _record  # type: ignore[assignment]
    try:
        guard.install()
        guard.install()
        guard.install()
    finally:
        os.register_at_fork = original  # type: ignore[assignment]

    assert len(registered) == 1
