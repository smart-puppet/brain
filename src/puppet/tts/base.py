from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterator

import numpy as np


class TtsBackend(ABC):
  @abstractmethod
  def synthesize_stream(self, text: str) -> Iterator[np.ndarray]:
    """Yield float32 mono PCM chunks at the backend sample rate."""

  @abstractmethod
  def sample_rate(self) -> int:
    pass

  @abstractmethod
  def stop(self) -> None:
    """Interrupt playback / synthesis (barge-in)."""
