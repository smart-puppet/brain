from __future__ import annotations

from typing import Callable

import numpy as np


def resample_linear(samples: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
  if src_rate == dst_rate or samples.size == 0:
    return samples.astype(np.float32, copy=True)
  ratio = dst_rate / src_rate
  dst_len = max(int(samples.size * ratio), 1)
  src_idx = np.linspace(0, samples.size - 1, num=dst_len, dtype=np.float32)
  return np.interp(src_idx, np.arange(samples.size, dtype=np.float32), samples).astype(np.float32)


def rms_energy(samples: np.ndarray) -> float:
  if samples.size == 0:
    return 0.0
  return float(np.sqrt(np.mean(samples.astype(np.float64) ** 2)))


def prepend_lead_in_silence(
  chunk: np.ndarray,
  sample_rate: int,
  lead_in_ms: int,
) -> np.ndarray:
  """Pad the start of a TTS chunk so ALSA does not clip the first phoneme."""
  if lead_in_ms <= 0:
    return chunk.astype(np.float32, copy=False)
  n = max(1, int(sample_rate * lead_in_ms / 1000))
  silence = np.zeros(n, dtype=np.float32)
  if chunk.size == 0:
    return silence
  return np.concatenate([silence, chunk.astype(np.float32, copy=False)])


def detect_barge_in(
  mic: np.ndarray,
  *,
  threshold: float = 0.02,
  is_speaking: Callable[[], bool],
) -> bool:
  return is_speaking() and rms_energy(mic) > threshold
