"""Shared Rich console for the analysis report.

Every report/plot module prints to the single :data:`console` defined here so
that ``record=True`` captures the whole run for :func:`~scraper.analysis.reporting.save_report`.
"""

from __future__ import annotations

from rich.console import Console
from rich.panel import Panel

console = Console(record=True, width=100)


def caveat(message: str) -> None:
    """
    Print an interpretation caveat.

    :param message: The caveat text.
    """
    console.print(
        Panel(
            message,
            title="Interpretation",
            title_align="left",
            border_style="yellow",
            padding=(0, 1),
        )
    )
