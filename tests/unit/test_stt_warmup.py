from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np

from puppet.stt.parakeet import SAMPLE_RATE, ParakeetStt


class _FakeStream:
  def feed(self, pcm: np.ndarray, sample_rate: int) -> tuple[str, bool]:
    return "", False

  def close(self) -> None:
    pass

  def finalize(self) -> str:
    return ""


class _FakeCtx:
  def set_att_context(self, left: int, right: int) -> None:
    pass

  def stream_begin_lang(self, lang: str) -> _FakeStream:
    return _FakeStream()

  def close(self) -> None:
    pass


def _make_stt(monkeypatch) -> ParakeetStt:
  fake_pk = MagicMock()
  fake_pk.load.return_value = _FakeCtx()
  monkeypatch.setitem(sys.modules, "puppet_parakeet", fake_pk)
  monkeypatch.setattr(Path, "is_file", lambda self: True)
  return ParakeetStt("models/stt/fake.gguf")


def test_stt_warmup_feeds_silence_and_resets(monkeypatch) -> None:
  stt = _make_stt(monkeypatch)
  feed_sizes: list[int] = []
  reset_calls = 0
  real_reset = stt.reset

  def track_feed(pcm: np.ndarray, sample_rate: int):
    feed_sizes.append(int(pcm.size))
    return None

  def track_reset() -> None:
    nonlocal reset_calls
    reset_calls += 1
    real_reset()

  stt.feed = track_feed  # type: ignore[method-assign]
  stt.reset = track_reset  # type: ignore[method-assign]
  stt.warmup(duration_ms=100, sample_rate=SAMPLE_RATE)

  assert feed_sizes
  assert sum(feed_sizes) >= int(SAMPLE_RATE * 0.1)
  assert reset_calls == 1


def test_create_stt_passes_thread_config(monkeypatch) -> None:
  thread_calls: list[int] = []
  fake_pk = MagicMock()
  fake_pk.load.return_value = _FakeCtx()
  fake_pk.set_num_threads = lambda n: thread_calls.append(n)
  monkeypatch.setitem(sys.modules, "puppet_parakeet", fake_pk)
  monkeypatch.setattr(Path, "is_file", lambda self: True)

  from puppet.stt.parakeet import create_stt

  create_stt(
    {
      "stt": {
        "backend": "parakeet",
        "model_path": "models/stt/fake.gguf",
        "warmup": False,
        "n_threads": 4,
        "n_batch": 2,
      }
    }
  )
  assert thread_calls == [4]


def test_create_stt_runs_warmup_when_enabled(monkeypatch) -> None:
  stt = _make_stt(monkeypatch)
  warmup_calls: list[int] = []

  def track_warmup(*, duration_ms: int = 1500, sample_rate: int = SAMPLE_RATE) -> None:
    warmup_calls.append(duration_ms)

  stt.warmup = track_warmup  # type: ignore[method-assign]
  monkeypatch.setattr("puppet.stt.parakeet.ParakeetStt", lambda **kwargs: stt)

  from puppet.stt.parakeet import create_stt

  create_stt(
    {
      "stt": {
        "backend": "parakeet",
        "model_path": "models/stt/fake.gguf",
        "warmup": True,
        "warmup_ms": 1200,
      }
    }
  )
  assert warmup_calls == [1200]
