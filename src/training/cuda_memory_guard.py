import logging

import torch

logger = logging.getLogger(__name__)


class CudaMemoryGuard:
    """
    Caps how much of a GPU one training process is allowed to reserve.

    Uncapped, PyTorch's allocator grows until the *driver* refuses a request,
    and that refusal is not a Python exception. On WSL the paging layer logs
    ``dxgk: dxgkio_make_resident: Ioctl failed: -12`` and every subsequent CUDA
    call in the process fails with ``CUDA driver error: device not ready``. The
    context is dead at that point, so no amount of care inside the update loop
    can recover it -- the run is lost even though the failure was ordinary
    memory pressure.

    Reserving a slice below the card's capacity moves that failure back inside
    PyTorch, where over-allocating raises :class:`torch.OutOfMemoryError` on the
    request that went too far and leaves the rest of the process intact. The
    headroom is not waste: a desktop compositor sharing the card needs its own
    surfaces to stay resident, and on this box that is ~1.6 GB of a 24 GB card
    before training starts.

    The cap applies to the caller's process only, so it must be set before the
    worker pool forks for the workers to inherit it.
    """

    def __init__(self, device: torch.device | str, fraction: float | None) -> None:
        """
        :param device: Device the trainer runs on; non-CUDA devices are ignored.
        :param fraction: Share of the card's total memory this process may
            reserve, in ``(0, 1]``. None disables the cap.
        :raises ValueError: If ``fraction`` is outside ``(0, 1]``.
        """
        if fraction is not None and not 0.0 < fraction <= 1.0:
            raise ValueError(
                f"cuda_memory_fraction must be in (0, 1], got {fraction}."
            )
        self._device = torch.device(device)
        self._fraction = fraction

    def apply(self) -> None:
        """
        Impose the cap, if one applies to this device.

        Silently does nothing when the trainer is not on CUDA or no fraction was
        configured, so the same call site serves every device.
        """
        if self._fraction is None or self._device.type != "cuda":
            return
        if not torch.cuda.is_available():
            logger.warning(
                "cuda_memory_fraction is set but no CUDA device is available; "
                "leaving the allocator uncapped."
            )
            return
        # "cuda" without an index is what a config carries; the allocator wants
        # a concrete device, which for an indexless one is the current default.
        index = (
            self._device.index
            if self._device.index is not None
            else torch.cuda.current_device()
        )
        torch.cuda.set_per_process_memory_fraction(self._fraction, index)
        total_bytes = torch.cuda.get_device_properties(index).total_memory
        logger.info(
            "Capping %s at %.0f%% of its %.1f GiB (%.1f GiB reservable); past that "
            "the allocator raises OutOfMemoryError instead of letting the driver "
            "fail the process.",
            self._device,
            self._fraction * 100,
            total_bytes / 1024**3,
            total_bytes * self._fraction / 1024**3,
        )
