from __future__ import annotations

from pathlib import Path

import pytest

from puppet.core.audio.wav import load_wav_mono_float32
from puppet.stt.parakeet import ParakeetStt

ROOT = Path(__file__).resolve().parents[2]
JFK_WAV = ROOT / "tests" / "fixtures" / "jfk.wav"
MODEL = ROOT / "models" / "stt" / "nemotron-3.5-asr-streaming-0.6b-q8_0.gguf"

JFK_EXPECTED = (
  "And so my fellow Americans ask not what your country can do for you. "
  "Ask what you can do for your country."
)

pytestmark = pytest.mark.skipif(
  not JFK_WAV.is_file() or not MODEL.is_file(),
  reason="Requires tests/fixtures/jfk.wav and nemotron STT model",
)


def _has_parakeet_binding() -> bool:
  try:
    import puppet_parakeet  # noqa: F401

    return True
  except ImportError:
    return False


def _strip_trailing_lang_tags(text: str) -> str:
  return text.strip()


def _transcribe_streaming(
  stt: ParakeetStt,
  audio,
  *,
  sample_rate: int,
  chunk_samples: int = 1280,
) -> str:
  parts: list[str] = []
  for offset in range(0, len(audio), chunk_samples):
    segment = stt.feed(audio[offset : offset + chunk_samples], sample_rate)
    if segment and segment.text:
      parts.append(segment.text)
  final = stt.finalize()
  if final and final.text:
    parts.append(final.text)
  return "".join(parts)


def test_jfk_wav_is_16k_mono_int16() -> None:
  audio, sample_rate = load_wav_mono_float32(JFK_WAV)
  assert sample_rate == 16000
  assert audio.ndim == 1
  assert audio.size > 0
  assert float(audio.max()) <= 1.0
  assert float(audio.min()) >= -1.0


@pytest.mark.skipif(not _has_parakeet_binding(), reason="puppet_parakeet not built")
def test_jfk_streaming_transcript() -> None:
  audio, sample_rate = load_wav_mono_float32(JFK_WAV)
  assert sample_rate == 16000

  stt = ParakeetStt(str(MODEL), language="en-US", streaming_chunk_ms=320)
  try:
    transcript = _transcribe_streaming(stt, audio, sample_rate=sample_rate)
    assert _strip_trailing_lang_tags(transcript) == JFK_EXPECTED
  finally:
    stt.close()
