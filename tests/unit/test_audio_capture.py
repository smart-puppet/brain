from __future__ import annotations

import numpy as np

from puppet.core.audio.capture import int16_bytes_to_float32


def test_int16_bytes_to_float32_mono() -> None:
  raw = np.array([0, 16384, -16384, 32767], dtype=np.int16).tobytes()
  samples = int16_bytes_to_float32(raw, channels=1)
  assert samples.dtype == np.float32
  assert samples[0] == 0.0
  assert abs(samples[1] - 0.5) < 1e-5
  assert abs(samples[2] + 0.5) < 1e-5
  assert abs(samples[3] - 32767 / 32768.0) < 1e-5


def test_int16_bytes_to_float32_stereo_downmix() -> None:
  raw = np.array([[16384, 0], [0, 16384]], dtype=np.int16).tobytes()
  samples = int16_bytes_to_float32(raw, channels=2)
  assert samples.shape == (2,)
  assert abs(samples[0] - 0.25) < 1e-5
  assert abs(samples[1] - 0.25) < 1e-5
