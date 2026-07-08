from __future__ import annotations

import wave
from pathlib import Path

import numpy as np


def load_wav_mono_float32(path: str | Path) -> tuple[np.ndarray, int]:
  """Load a WAV file as mono float32 PCM in [-1, 1]. No resampling."""
  wav_path = Path(path)
  with wave.open(str(wav_path), "rb") as wf:
    channels = wf.getnchannels()
    sample_width = wf.getsampwidth()
    sample_rate = wf.getframerate()
    n_frames = wf.getnframes()
    raw = wf.readframes(n_frames)

  if sample_width != 2:
    raise ValueError(f"{wav_path}: expected 16-bit PCM, got width={sample_width}")
  samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
  if channels > 1:
    samples = samples.reshape(-1, channels).mean(axis=1)
  samples /= 32768.0
  return np.clip(samples, -1.0, 1.0), sample_rate
