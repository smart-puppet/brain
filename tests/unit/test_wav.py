from __future__ import annotations

from pathlib import Path

import numpy as np

from puppet.core.audio.wav import load_wav_mono_float32

ROOT = Path(__file__).resolve().parents[2]
JFK_WAV = ROOT / "tests" / "fixtures" / "jfk.wav"


def test_load_jfk_wav_no_resample() -> None:
  if not JFK_WAV.is_file():
    return
  audio, sample_rate = load_wav_mono_float32(JFK_WAV)
  assert sample_rate == 16000
  assert audio.dtype == np.float32
  assert audio.shape == (176000,)
