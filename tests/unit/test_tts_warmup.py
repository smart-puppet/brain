from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np

from puppet.tts.piper import PiperTts


def test_tts_warmup_synthesizes_text(monkeypatch) -> None:
  chunks = [np.zeros(100, dtype=np.float32), np.zeros(50, dtype=np.float32)]

  class FakeVoice:
    config = MagicMock(sample_rate=22050)

    def synthesize_stream_raw(self, text: str, speaker_id=None):
      assert text == "."
      yield from (np.zeros(100, dtype=np.int16).tobytes(), np.zeros(50, dtype=np.int16).tobytes())

  monkeypatch.setattr("puppet.tts.piper._load_piper_voice", lambda **kwargs: FakeVoice())
  tts = PiperTts(model_path="models/tts/fake.onnx")
  tts.warmup(text=".")


def test_create_tts_runs_warmup_when_enabled(monkeypatch) -> None:
  warmup_texts: list[str] = []

  class FakeVoice:
    config = MagicMock(sample_rate=22050)

    def synthesize_stream_raw(self, text: str, speaker_id=None):
      yield np.zeros(10, dtype=np.int16).tobytes()

  monkeypatch.setattr("puppet.tts.piper._load_piper_voice", lambda **kwargs: FakeVoice())

  real_init = PiperTts.__init__

  def wrapped_init(self, *args, **kwargs):
    real_init(self, *args, **kwargs)

  monkeypatch.setattr(PiperTts, "__init__", wrapped_init)

  real_warmup = PiperTts.warmup

  def track_warmup(self, *, text: str = ".") -> None:
    warmup_texts.append(text)
    real_warmup(self, text=text)

  monkeypatch.setattr(PiperTts, "warmup", track_warmup)

  from puppet.tts.piper import create_tts

  create_tts(
    {
      "tts": {
        "backend": "piper",
        "model_path": "models/tts/fake.onnx",
        "warmup": True,
        "warmup_text": "Hi",
      }
    }
  )
  assert warmup_texts == ["Hi"]
