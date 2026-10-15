"""Live progress on stderr: rich bars on a terminal, periodic plain lines in logs."""

import time
from types import TracebackType
from typing import Self

from rich.console import Console
from rich.progress import (
	BarColumn,
	MofNCompleteColumn,
	Progress,
	SpinnerColumn,
	TaskID,
	TaskProgressColumn,
	TextColumn,
	TimeElapsedColumn,
	TimeRemainingColumn,
)

console = Console(stderr=True, highlight=False)


class LiveProgress:
	"""Tracks one bar per description. Without a terminal, prints a line every interval.

	The bars start with the first update, so tqdm output from work before it, such as a
	tokenizer download, is not captured by the live display.
	"""

	def __init__(self, interval_seconds: float = 30.0) -> None:
		self.interval_seconds = interval_seconds
		self.tasks: dict[str, TaskID] = {}
		self.printed: dict[str, float] = {}
		self.finished: set[str] = set()
		self.bars: Progress | None = None
		if console.is_terminal:
			self.bars = Progress(
				SpinnerColumn(),
				TextColumn("[bold]{task.description}"),
				BarColumn(),
				MofNCompleteColumn(),
				TaskProgressColumn(),
				TimeElapsedColumn(),
				TextColumn("eta"),
				TimeRemainingColumn(),
				TextColumn("[dim]{task.fields[detail]}"),
				console=console,
			)

	def __enter__(self) -> Self:
		return self

	def __exit__(
		self,
		kind: type[BaseException] | None,
		error: BaseException | None,
		traceback: TracebackType | None,
	) -> None:
		if self.bars:
			self.bars.stop()

	def update(self, description: str, done: int, total: int, detail: str = "") -> None:
		if self.bars:
			if not self.tasks:
				self.bars.start()
			if description not in self.tasks:
				self.tasks[description] = self.bars.add_task(
					description, total=total, completed=done, detail=detail
				)
			else:
				self.bars.update(
					self.tasks[description], total=total, completed=done, detail=detail
				)
			return
		if description in self.finished:
			return
		now = time.monotonic()
		last = self.printed.get(description)
		if done < total and last is not None and now - last < self.interval_seconds:
			return
		self.printed[description] = now
		if done >= total:
			self.finished.add(description)
		percent = f" ({done / total:.0%})" if total else ""
		suffix = f"  {detail}" if detail else ""
		console.print(f"{description}: {done:,}/{total:,}{percent}{suffix}", soft_wrap=True)
