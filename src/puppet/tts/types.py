from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MouthEvent:
  """Mouth state change at a sample offset from phrase audio start."""

  sample_offset: int
  open: bool


@dataclass(frozen=True)
class WordCue:
  """Mouth open window relative to the start of one TTS playback chunk."""

  start_ms: int
  end_ms: int


@dataclass(frozen=True)
class TtsChunk:
  samples: np.ndarray
  mouth_timeline: list[MouthEvent] | None = None
  phoneme_hold_ms: list[int] | None = None
  word_cues: list[WordCue] | None = None
