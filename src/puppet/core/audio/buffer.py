from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class RingBuffer:
  """Fixed-size ring buffer for mono float32 PCM."""

  capacity_samples: int
  _data: np.ndarray = field(init=False)
  _write_pos: int = 0
  _filled: int = 0

  def __post_init__(self) -> None:
    self._data = np.zeros(self.capacity_samples, dtype=np.float32)

  @property
  def filled(self) -> int:
    return self._filled

  def write(self, samples: np.ndarray) -> None:
    if samples.size == 0:
      return
    samples = samples.astype(np.float32, copy=False)
    for sample in samples:
      self._data[self._write_pos] = sample
      self._write_pos = (self._write_pos + 1) % self.capacity_samples
      self._filled = min(self._filled + 1, self.capacity_samples)

  def read_latest(self, n_samples: int) -> np.ndarray:
    return self.read_delayed(n_samples, delay_samples=0)

  def read_delayed(self, n_samples: int, delay_samples: int) -> np.ndarray:
    """Return ``n_samples`` ending ``delay_samples`` before the write head."""
    delay_samples = max(0, delay_samples)
    if n_samples <= 0:
      return np.zeros(0, dtype=np.float32)
    available = max(0, self._filled - delay_samples)
    n = min(n_samples, available)
    if n <= 0:
      return np.zeros(n_samples, dtype=np.float32)

    end_idx = (self._write_pos - 1 - delay_samples) % self.capacity_samples
    out = np.zeros(n_samples, dtype=np.float32)
    start_out = n_samples - n
    for i in range(n):
      idx = (end_idx - (n - 1 - i)) % self.capacity_samples
      out[start_out + i] = self._data[idx]
    return out

  def clear(self) -> None:
    self._write_pos = 0
    self._filled = 0
    self._data.fill(0.0)


@dataclass
class AudioReference:
  """Stores recent TTS playback aligned for acoustic echo cancellation."""

  sample_rate: int
  max_seconds: float = 2.0
  delay_ms: int = 0
  playback_delay_ms: int = 0
  _buffer: RingBuffer = field(init=False)
  _delay_samples: int = field(init=False)

  def __post_init__(self) -> None:
    capacity = int(self.sample_rate * self.max_seconds)
    self._buffer = RingBuffer(capacity_samples=max(capacity, 1))
    self._delay_samples = max(
      0,
      int(self.sample_rate * (self.delay_ms + self.playback_delay_ms) / 1000),
    )

  @property
  def delay_samples(self) -> int:
    return self._delay_samples

  def write(self, samples: np.ndarray) -> None:
    self._buffer.write(samples)

  def read_aligned(self, n_samples: int) -> np.ndarray:
    return self.read_for_cancel(n_samples)

  def read_for_cancel(self, n_samples: int) -> np.ndarray:
    return self._buffer.read_delayed(n_samples, self._delay_samples)

  def recent_rms(self, window_samples: int = 320) -> float:
    n = min(window_samples, self._buffer.filled)
    if n <= 0:
      return 0.0
    chunk = self._buffer.read_latest(n)
    return float(np.sqrt(np.mean(chunk.astype(np.float64) ** 2)))

  def clear(self) -> None:
    self._buffer.clear()
