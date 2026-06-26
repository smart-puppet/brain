import numpy as np
import pytest

from puppet.core.audio.buffer import AudioReference
from puppet.core.audio.speex_echo import EchoCanceller, SpeexDspUnavailable, float32_to_int16


pytest.importorskip("ctypes.util")

try:
  _aec = EchoCanceller(
    AudioReference(sample_rate=16000, max_seconds=1.0),
    sample_rate=16000,
    frame_size=160,
    filter_length=1024,
  )
except SpeexDspUnavailable:
  pytest.skip("libspeexdsp not available", allow_module_level=True)
else:
  _aec.close()


def _make_aec(ref: AudioReference | None = None) -> EchoCanceller:
  reference = ref or AudioReference(sample_rate=16000, max_seconds=2.0)
  return EchoCanceller(
    reference,
    sample_rate=16000,
    frame_size=160,
    filter_length=2048,
  )


def test_speex_reduces_synthetic_echo() -> None:
  rng = np.random.default_rng(0)
  ref = AudioReference(sample_rate=16000, max_seconds=2.0)
  aec = _make_aec(ref)
  try:
    play = (np.sin(np.linspace(0, 30 * np.pi, 3200)) * 0.25).astype(np.float32)
    ref.write(play)
    mic = (play * 0.85 + rng.normal(0, 0.01, play.size)).astype(np.float32)
    clean = aec.process(mic)
    assert np.sqrt(np.mean(clean.astype(np.float64) ** 2)) < np.sqrt(
      np.mean(mic.astype(np.float64) ** 2)
    ) * 0.55
  finally:
    aec.close()


def test_speex_bypasses_when_reference_is_quiet() -> None:
  ref = AudioReference(sample_rate=16000, max_seconds=1.0)
  aec = _make_aec(ref)
  try:
    mic = np.array([0.2, -0.1, 0.3, 0.05] * 80, dtype=np.float32)
    clean = aec.process(mic)
    assert np.allclose(clean, mic, atol=1e-5)
  finally:
    aec.close()


def test_float32_int16_roundtrip() -> None:
  samples = np.array([0.0, 0.5, -0.5, 1.0, -1.0], dtype=np.float32)
  back = float32_to_int16(samples).astype(np.float32) / 32768.0
  assert np.allclose(back[:3], samples[:3], atol=1e-4)
