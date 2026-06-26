from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable, DefaultDict, List


class EventBus:
  """Simple in-process pub/sub for orchestration events."""

  def __init__(self) -> None:
    self._handlers: DefaultDict[str, List[Callable[..., None]]] = defaultdict(list)

  def on(self, event: str, handler: Callable[..., None]) -> None:
    self._handlers[event].append(handler)

  def emit(self, event: str, **payload: Any) -> None:
    for handler in self._handlers[event]:
      handler(**payload)
