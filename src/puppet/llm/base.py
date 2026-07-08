from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterator

from puppet.core.types import Conversation


class LlmBackend(ABC):
  @abstractmethod
  def stream_reply(self, conversation: Conversation) -> Iterator[str]:
    """Yield token or word chunks for TTS."""

  @abstractmethod
  def cancel(self) -> None:
    """Stop an in-flight generation (barge-in)."""
