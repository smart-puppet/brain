from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterator

from puppet.tts.types import TtsChunk


class TtsBackend(ABC):
  @abstractmethod
  def synthesize_stream(self, text: str) -> Iterator[TtsChunk]:
    """Yield float32 mono PCM chunks at the backend sample rate."""

  @abstractmethod
  def sample_rate(self) -> int:
    pass

  @abstractmethod
  def stop(self) -> None:
    """Interrupt playback / synthesis (barge-in)."""

  @property
  def has_mouth_timeline(self) -> bool:
    return False
