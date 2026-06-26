from __future__ import annotations

from typing import Callable

import numpy as np

from puppet.core.audio.buffer import AudioReference


def _normalized_correlation(a: np.ndarray, b: np.ndarray) -> float:
  n = min(a.size, b.size)
  if n <= 0:
    return 0.0
  x = a[:n].astype(np.float64)
  y = b[:n].astype(np.float64)
  denom = float(np.linalg.norm(x) * np.linalg.norm(y))
  if denom <= 1e-10:
    return 0.0
  return float(np.dot(x, y) / denom)


def should_suppress_echo_stt(
  mic: np.ndarray,
  clean: np.ndarray,
  *,
  reference: AudioReference,
  enabled: bool,
  suppress_stt_on_echo: bool,
  echo_ratio_threshold: float,
  min_reference_rms: float,
  min_mic_rms: float,
) -> bool:
  """Skip STT when the mic chunk is dominated by playback echo."""
  if not enabled or not suppress_stt_on_echo:
    return False
  if reference.recent_rms() < min_reference_rms:
    return False
  mic_rms = rms_energy(mic)
  if mic_rms < min_mic_rms:
    return False
  ref = reference.read_for_cancel(mic.size)
  if rms_energy(ref) < min_reference_rms:
    return False
  if _normalized_correlation(mic, ref) < 0.35:
    return False
  clean_rms = rms_energy(clean)
  ratio = clean_rms / mic_rms
  if ratio > echo_ratio_threshold:
    return False
  if _normalized_correlation(clean, ref) > 0.3:
    return True
  return True


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
