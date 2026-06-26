from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

VadEventKind = Literal["start", "end"]


@dataclass(frozen=True)
class VadEvent:
  kind: VadEventKind


class VoiceActivityDetector(ABC):
  @abstractmethod
  def feed(self, samples: np.ndarray) -> list[VadEvent]:
    """Feed mono float32 PCM; return any speech boundary events."""

  @property
  @abstractmethod
  def is_speech(self) -> bool:
    """True while the stream is inside a speech segment."""

  @abstractmethod
  def reset(self) -> None:
    pass


class PassthroughVad(VoiceActivityDetector):
  """No-op VAD: treats all audio as speech."""

  def feed(self, samples: np.ndarray) -> list[VadEvent]:
    return []

  @property
  def is_speech(self) -> bool:
    return True

  def reset(self) -> None:
    pass


def _window_size(sample_rate: int) -> int:
  return 512 if sample_rate == 16000 else 256


class _OnnxSileroModel:
  """Silero VAD ONNX session (numpy only, no torch)."""

  def __init__(self, model_path: str | Path, *, force_cpu: bool = True) -> None:
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.inter_op_num_threads = 1
    opts.intra_op_num_threads = 1
    providers = ["CPUExecutionProvider"]
    if not force_cpu:
      providers = ort.get_available_providers()
    self._session = ort.InferenceSession(
      str(model_path),
      providers=providers,
      sess_options=opts,
    )
    self._sample_rates = [16000] if "16k" in str(model_path) else [8000, 16000]
    self.reset_states()

  def reset_states(self, batch_size: int = 1) -> None:
    self._state = np.zeros((2, batch_size, 128), dtype=np.float32)
    self._context = np.zeros((batch_size, 0), dtype=np.float32)
    self._last_sr = 0
    self._last_batch_size = 0

  def _validate_input(self, x: np.ndarray, sr: int) -> tuple[np.ndarray, int]:
    if x.ndim == 1:
      x = x.reshape(1, -1)
    if x.ndim > 2:
      raise ValueError(f"Too many dimensions for input audio chunk: {x.ndim}")

    if sr != 16000 and sr % 16000 == 0:
      step = sr // 16000
      x = x[:, ::step]
      sr = 16000

    if sr not in self._sample_rates:
      raise ValueError(f"Supported sampling rates: {self._sample_rates}")
    if sr / x.shape[1] > 31.25:
      raise ValueError("Input audio chunk is too short")
    return x.astype(np.float32, copy=False), sr

  def predict(self, x: np.ndarray, sr: int) -> float:
    x, sr = self._validate_input(x, sr)
    num_samples = 512 if sr == 16000 else 256
    if x.shape[-1] != num_samples:
      raise ValueError(
        f"Expected {num_samples} samples for {sr} Hz, got {x.shape[-1]}"
      )

    batch_size = x.shape[0]
    context_size = 64 if sr == 16000 else 32

    if not self._last_batch_size:
      self.reset_states(batch_size)
    if self._last_sr and self._last_sr != sr:
      self.reset_states(batch_size)
    if self._last_batch_size and self._last_batch_size != batch_size:
      self.reset_states(batch_size)

    if self._context.shape[1] == 0:
      self._context = np.zeros((batch_size, context_size), dtype=np.float32)

    x = np.concatenate([self._context, x], axis=1)
    ort_inputs = {
      "input": x,
      "state": self._state,
      "sr": np.array(sr, dtype=np.int64),
    }
    out, state = self._session.run(None, ort_inputs)
    self._state = state
    self._context = x[:, -context_size:]
    self._last_sr = sr
    self._last_batch_size = batch_size
    return float(np.asarray(out).reshape(-1)[0])


class _StreamingVadIterator:
  """Streaming state machine matching Silero VADIterator semantics."""

  def __init__(
    self,
    model: _OnnxSileroModel,
    *,
    threshold: float,
    sampling_rate: int,
    min_silence_duration_ms: int,
    speech_pad_ms: int,
  ) -> None:
    if sampling_rate not in (8000, 16000):
      raise ValueError("VAD supports 8000 or 16000 Hz sample rates")
    self._model = model
    self._threshold = threshold
    self._sampling_rate = sampling_rate
    self._min_silence_samples = sampling_rate * min_silence_duration_ms / 1000
    self._speech_pad_samples = sampling_rate * speech_pad_ms / 1000
    self.triggered = False
    self._temp_end = 0
    self._current_sample = 0

  def reset_states(self) -> None:
    self._model.reset_states()
    self.triggered = False
    self._temp_end = 0
    self._current_sample = 0

  def process(self, window: np.ndarray) -> VadEvent | None:
    window_size_samples = window.size
    self._current_sample += window_size_samples
    speech_prob = self._model.predict(window, self._sampling_rate)

    if speech_prob >= self._threshold and self._temp_end:
      self._temp_end = 0

    if speech_prob >= self._threshold and not self.triggered:
      self.triggered = True
      return VadEvent(kind="start")

    if speech_prob < self._threshold - 0.15 and self.triggered:
      if not self._temp_end:
        self._temp_end = self._current_sample
      if self._current_sample - self._temp_end < self._min_silence_samples:
        return None
      self._temp_end = 0
      self.triggered = False
      return VadEvent(kind="end")

    return None


class SileroVad(VoiceActivityDetector):
  """Streaming Silero VAD via onnxruntime (512-sample windows at 16 kHz)."""

  def __init__(
    self,
    model_path: str | Path,
    *,
    sample_rate: int = 16000,
    threshold: float = 0.5,
    min_silence_duration_ms: int = 300,
    speech_pad_ms: int = 30,
    force_cpu: bool = True,
  ) -> None:
    path = Path(model_path)
    if not path.is_file():
      raise FileNotFoundError(
        f"Silero VAD model not found: {path}. Run ./scripts/download_models.sh"
      )

    self._window = _window_size(sample_rate)
    self._iterator = _StreamingVadIterator(
      _OnnxSileroModel(path, force_cpu=force_cpu),
      threshold=threshold,
      sampling_rate=sample_rate,
      min_silence_duration_ms=min_silence_duration_ms,
      speech_pad_ms=speech_pad_ms,
    )
    self._pending = np.zeros(0, dtype=np.float32)

  def feed(self, samples: np.ndarray) -> list[VadEvent]:
    if samples.size == 0:
      return []
    self._pending = np.concatenate((self._pending, samples.astype(np.float32, copy=False)))
    events: list[VadEvent] = []
    while self._pending.size >= self._window:
      window = self._pending[: self._window]
      self._pending = self._pending[self._window :]
      event = self._iterator.process(window)
      if event:
        events.append(event)
    return events

  @property
  def is_speech(self) -> bool:
    return self._iterator.triggered

  def reset(self) -> None:
    self._pending = np.zeros(0, dtype=np.float32)
    self._iterator.reset_states()


def create_vad(config: dict[str, Any]) -> VoiceActivityDetector:
  vad_cfg = config.get("vad", {})
  if not vad_cfg.get("enabled", True):
    return PassthroughVad()

  backend = vad_cfg.get("backend", "silero")
  if backend != "silero":
    raise ValueError(f"Unsupported VAD backend: {backend}")

  audio_rate = int(config.get("audio", {}).get("sample_rate", 16000))
  model_path = vad_cfg.get("model_path", "models/vad/silero_vad.onnx")
  return SileroVad(
    model_path=model_path,
    sample_rate=audio_rate,
    threshold=float(vad_cfg.get("threshold", 0.5)),
    min_silence_duration_ms=int(vad_cfg.get("min_silence_duration_ms", 300)),
    speech_pad_ms=int(vad_cfg.get("speech_pad_ms", 30)),
    force_cpu=bool(vad_cfg.get("force_cpu", True)),
  )
