"""The shared stderr console for command output."""

from rich.console import Console

console = Console(stderr=True, highlight=False)
