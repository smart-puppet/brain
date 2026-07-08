import numpy as np

from puppet.core.audio.vad import (
  PassthroughVad,
  VadEvent,
  _StreamingVadIterator,
  _window_size,
)


def test_passthrough_vad_always_speech() -> None:
  vad = PassthroughVad()
  assert vad.is_speech is True
  assert vad.feed(np.zeros(320, dtype=np.float32)) == []


def test_window_size() -> None:
  assert _window_size(16000) == 512
  assert _window_size(8000) == 256


class _MockOnnxModel:
  def __init__(self, probs: list[float]) -> None:
    self._probs = probs
    self._idx = 0

  def reset_states(self, batch_size: int = 1) -> None:
    self._idx = 0

  def predict(self, x: np.ndarray, sr: int) -> float:
    if self._idx >= len(self._probs):
      return 0.0
    prob = self._probs[self._idx]
    self._idx += 1
    return prob


def test_streaming_iterator_detects_start_end() -> None:
  model = _MockOnnxModel([0.1, 0.9, 0.9, 0.1, 0.1, 0.1, 0.1])
  it = _StreamingVadIterator(
    model,  # type: ignore[arg-type]
    threshold=0.5,
    sampling_rate=16000,
    min_silence_duration_ms=32,
    speech_pad_ms=0,
  )
  window = np.zeros(512, dtype=np.float32)

  assert it.process(window) is None
  assert it.process(window) == VadEvent(kind="start")
  assert it.triggered is True
  assert it.process(window) is None
  assert it.process(window) is None
  assert it.process(window) == VadEvent(kind="end")
  assert it.triggered is False
