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
    samples = samples.astype(np.float32, copy=False).ravel()
    n = samples.size
    cap = self.capacity_samples
    if n >= cap:
      # Only keep the last cap samples.
      self._data[:] = samples[-cap:]
      self._write_pos = 0
      self._filled = cap
      return
    end = self._write_pos + n
    if end <= cap:
      self._data[self._write_pos : end] = samples
    else:
      first = cap - self._write_pos
      self._data[self._write_pos : cap] = samples[:first]
      self._data[: n - first] = samples[first:]
    self._write_pos = end % cap
    self._filled = min(self._filled + n, cap)

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

    start_idx = (self._write_pos - delay_samples - n) % self.capacity_samples
    out = np.zeros(n_samples, dtype=np.float32)
    start_out = n_samples - n
    if start_idx + n <= self.capacity_samples:
      out[start_out:] = self._data[start_idx : start_idx + n]
    else:
      first = self.capacity_samples - start_idx
      out[start_out : start_out + first] = self._data[start_idx:]
      out[start_out + first :] = self._data[: n - first]
    return out

  def clear(self) -> None:
    self._write_pos = 0
    self._filled = 0
    self._data.fill(0.0)
