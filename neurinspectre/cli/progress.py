"""
Progress utilities for NeurInSpectre CLI.
"""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from typing import Iterator, Optional

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)


class ProgressReporter:
    def __init__(
        self,
        *,
        description: str,
        total: Optional[int],
        enabled: bool,
        console: Console,
        show_queries: bool = False,
    ) -> None:
        self._enabled = enabled
        self._show_queries = bool(show_queries)
        columns = [
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        ]
        if self._show_queries:
            columns.append(TextColumn("Queries {task.fields[queries]:>8}"))
        self._progress = Progress(*columns, console=console, expand=True)
        self._task_id = None
        self._description = description
        self._total = total

    def __enter__(self) -> "ProgressReporter":
        if not self._enabled:
            return self
        self._progress.start()
        self._task_id = self._progress.add_task(
            self._description,
            total=self._total,
            queries=0 if self._show_queries else None,
        )
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if not self._enabled:
            return
        self._progress.stop()

    def advance(
        self,
        amount: int = 1,
        *,
        description: Optional[str] = None,
        queries: Optional[int] = None,
    ) -> None:
        if not self._enabled or self._task_id is None:
            return
        fields = {}
        if description:
            fields["description"] = description
        if self._show_queries and queries is not None:
            fields["queries"] = int(queries)
        self._progress.update(self._task_id, advance=amount, **fields)

    def set_phase(self, label: str, phase: int, total_phases: int) -> None:
        description = f"Phase {phase}/{total_phases}: {label}"
        self._description = description
        if not self._enabled or self._task_id is None:
            return
        self._progress.update(self._task_id, description=description)


@contextmanager
def status(console: Console, message: str, enabled: bool) -> Iterator[None]:
    if not enabled:
        yield None
        return
    with console.status(message):
        yield None
