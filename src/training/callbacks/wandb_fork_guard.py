from __future__ import annotations

import logging
import os
import weakref
from typing import Any

logger = logging.getLogger(__name__)


class WandbForkGuard:
    """
    Stops forked children from inheriting W&B's blocking exit hooks.

    ``wandb.init`` leaves a ``weakref.finalize`` hook on its internal service
    handle. The hook asks W&B's background service to release the run's API
    resources and waits for the reply on a ``concurrent.futures`` future with no
    timeout. That reply is delivered by an asyncio loop running on a dedicated
    thread inside the W&B client.

    ``ParallelEnv`` starts its workers with ``fork``, which copies the parent's
    memory but none of its threads. Every worker therefore inherits a live
    finalizer whose loop thread does not exist. torchrl's worker ``close``
    handler ends with an explicit ``gc.collect()``
    (``batched_envs.py:3014``); that collection reclaims the inherited handle
    and fires the hook, so the worker wedges *inside the close handler* and
    never reaches interpreter shutdown at all. Because torchrl's teardown then
    ends in an unbounded ``Process.join``, the parent never finishes either.
    Observed as a 5M-frame run reaching 100% of its budget and hanging
    indefinitely with all 32 workers alive.

    Detaching the finalizers in the child is safe because the resources they
    release belong to the parent's run: the child neither opened them nor is
    entitled to close them, and the parent runs its own cleanup at exit. Only
    children are touched, so the parent's W&B logging is unchanged.
    """

    def __init__(self, module_prefix: str = "wandb") -> None:
        """
        :param module_prefix: Top-level module whose finalizers are detached in
            forked children. Anything defined in this module or a submodule of
            it is considered W&B's.
        """
        self._module_prefix = module_prefix
        self._installed = False

    def install(self) -> None:
        """
        Register the child-side handler, once per process.

        Must be called after ``wandb.init`` and before the first fork, i.e.
        before the collector builds its worker pool. Registering twice would
        run the handler twice per fork, which is harmless but pointless, so
        repeat calls are ignored.
        """
        if self._installed:
            return
        os.register_at_fork(after_in_child=self.disarm)
        self._installed = True

    def disarm(self) -> None:
        """
        Detach every inherited W&B finalizer, running in the freshly forked child.

        Registered as the fork handler by :meth:`install`, and public so a caller
        managing its own fork point can invoke it directly.

        A fork handler that raises leaves the child in an undefined state, so
        failure is swallowed rather than propagated: the worst case is the hang
        this guard exists to prevent, which is no worse than not running at all.
        """
        try:
            detached = self._detach_finalizers()
        except Exception:
            logger.warning(
                "Could not detach inherited W&B finalizers; this worker may hang "
                "at exit.", exc_info=True
            )
            return
        if detached:
            logger.debug("Detached %d inherited W&B finalizer(s).", detached)

    def _detach_finalizers(self) -> int:
        """
        Disable the W&B entries of this process's ``weakref.finalize`` registry.

        ``weakref`` offers no public way to enumerate live finalizers. The
        public alternative -- filtering ``gc.get_objects()`` -- is the wrong
        trade here: it increments the refcount of every tracked object, which in
        a freshly forked child dirties the page behind each one. Measured at
        ~101 MiB of copy-on-write per million tracked objects, and this pool
        forks 32 workers on a machine the project already runs near the memory
        ceiling of. Reading the registry touches a single dict instead.

        The registry is reached through ``getattr`` because it is private and
        untyped; absent, this is a Python whose internals have moved, and there
        is then nothing to detach and nothing to fail over.

        :return: Number of finalizers detached.
        """
        registry: dict[weakref.finalize, Any] = getattr(weakref.finalize, "_registry", {})
        detached = 0
        for finalizer in list(registry):
            func = getattr(registry.get(finalizer), "func", None)
            if func is None or not self._is_wandb(func):
                continue
            finalizer.detach()
            detached += 1
        return detached

    def _is_wandb(self, func: object) -> bool:
        """
        :param func: Callable a finalizer would invoke.
        :return: True if it was defined in W&B's package.
        """
        module = getattr(func, "__module__", None)
        if not isinstance(module, str):
            return False
        return module == self._module_prefix or module.startswith(f"{self._module_prefix}.")
