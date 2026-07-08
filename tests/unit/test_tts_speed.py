from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from puppet.tts.piper import PiperTts, speed_to_length_scale


def test_speed_to_length_scale() -> None:
  assert speed_to_length_scale(1.0) is None
  assert speed_to_length_scale(2.0) == pytest.approx(0.5)
  assert speed_to_length_scale(0.5) == pytest.approx(2.0)


def test_speed_to_length_scale_rejects_non_positive() -> None:
  with pytest.raises(ValueError, match="tts.speed must be > 0"):
    speed_to_length_scale(0)


def test_piper_tts_passes_length_scale_to_synthesize(monkeypatch) -> None:
  captured: list[object] = []

  class FakeVoice:
    config = MagicMock(sample_rate=22050)

    def synthesize(self, text, syn_config=None, include_alignments=False):
      del text, include_alignments
      captured.append(syn_config)
      chunk = MagicMock()
      chunk.audio_float_array = np.zeros(10, dtype=np.float32)
      chunk.phoneme_alignments = None
      yield chunk

  monkeypatch.setattr("puppet.tts.piper._load_piper_voice", lambda **kwargs: FakeVoice())
  tts = PiperTts(model_path="models/tts/fake.onnx", speed=1.25)
  list(tts.synthesize_stream("hello"))
  assert captured
  assert captured[0].length_scale == pytest.approx(0.8)


def test_piper_tts_skips_stream_raw_when_speed_not_one(monkeypatch) -> None:
  raw_calls = 0
  synth_calls = 0

  class FakeVoice:
    config = MagicMock(sample_rate=22050)

    def synthesize_stream_raw(self, text, speaker_id=None):
      del text, speaker_id
      nonlocal raw_calls
      raw_calls += 1
      yield np.zeros(10, dtype=np.int16).tobytes()

    def synthesize(self, text, syn_config=None, include_alignments=False):
      del text, syn_config, include_alignments
      nonlocal synth_calls
      synth_calls += 1
      chunk = MagicMock()
      chunk.audio_float_array = np.zeros(10, dtype=np.float32)
      chunk.phoneme_alignments = None
      yield chunk

  monkeypatch.setattr("puppet.tts.piper._load_piper_voice", lambda **kwargs: FakeVoice())
  tts = PiperTts(model_path="models/tts/fake.onnx", speed=1.5)
  list(tts.synthesize_stream("hello"))
  assert raw_calls == 0
  assert synth_calls == 1
