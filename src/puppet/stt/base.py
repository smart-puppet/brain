from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterator

import numpy as np

from puppet.core.types import TranscriptSegment


class SttBackend(ABC):
  @abstractmethod
  def feed(self, pcm: np.ndarray, sample_rate: int) -> TranscriptSegment | None:
    """Feed audio; return a segment when new text is available."""

  @abstractmethod
  def finalize(self) -> TranscriptSegment | None:
    """Flush remaining audio at end of turn."""

  @abstractmethod
  def reset(self) -> None:
    """Reset session state."""

  def suspend(self) -> None:
    """Release GPU/native resources while the LLM runs (optional)."""

  def resume(self) -> None:
    """Restore after :meth:`suspend` (optional)."""

  def close(self) -> None:
    """Release native resources (GPU memory, etc.)."""
