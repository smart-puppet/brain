from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterator

import numpy as np
import pyaudio
import threading

from puppet.core.types import AudioChunk

logger = logging.getLogger(__name__)

# parakeet.cpp streaming API: 16 kHz mono float32 PCM in [-1, 1].
# PyAudio capture: 16 kHz, paInt16 (16-bit signed PCM), mono.
STT_SAMPLE_RATE = 16000
CAPTURE_FORMAT = pyaudio.paInt16
CAPTURE_SAMPLE_WIDTH_BYTES = 2


@dataclass(frozen=True)
class InputDeviceInfo:
  index: int
  name: str
  max_input_channels: int
  default_sample_rate: float
  supports_16k_int16: bool


def list_input_devices(pa: pyaudio.PyAudio | None = None) -> list[InputDeviceInfo]:
  own = pa is None
  if own:
    pa = pyaudio.PyAudio()
  try:
    devices: list[InputDeviceInfo] = []
    for index in range(pa.get_device_count()):
      info = pa.get_device_info_by_index(index)
      if int(info.get("maxInputChannels", 0)) <= 0:
        continue
      supports_16k = pa.is_format_supported(
        STT_SAMPLE_RATE,
        input_device=index,
        input_channels=1,
        input_format=CAPTURE_FORMAT,
      )
      devices.append(
        InputDeviceInfo(
          index=index,
          name=str(info.get("name", f"device-{index}")),
          max_input_channels=int(info["maxInputChannels"]),
          default_sample_rate=float(info["defaultSampleRate"]),
          supports_16k_int16=supports_16k,
        )
      )
    return devices
  finally:
    if own:
      pa.terminate()


def int16_bytes_to_float32(raw: bytes, *, channels: int = 1) -> np.ndarray:
  """Convert PortAudio int16 PCM to mono float32 in [-1, 1]."""
  samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
  if channels > 1:
    samples = samples.reshape(-1, channels).mean(axis=1)
  samples /= 32768.0
  return np.clip(samples, -1.0, 1.0)


class AudioCapture:
  def __init__(
    self,
    *,
    sample_rate: int = STT_SAMPLE_RATE,
    channels: int = 1,
    chunk_ms: int = 20,
    device_index: int | None = None,
  ) -> None:
    if sample_rate != STT_SAMPLE_RATE:
      raise ValueError(f"Audio capture must use {STT_SAMPLE_RATE} Hz for STT, got {sample_rate}")
    if channels != 1:
      raise ValueError("Audio capture must be mono (channels=1) for STT")

    self.sample_rate = STT_SAMPLE_RATE
    self.channels = channels
    self.chunk_ms = chunk_ms
    self.chunk_samples = max(int(self.sample_rate * chunk_ms / 1000), 1)

    self._pa = pyaudio.PyAudio()
    self.device_index = self._resolve_device_index(device_index)
    device_info = self._pa.get_device_info_by_index(self.device_index)
    self.device_name = str(device_info.get("name", f"device-{self.device_index}"))

    if not self._pa.is_format_supported(
      self.sample_rate,
      input_device=self.device_index,
      input_channels=self.channels,
      input_format=CAPTURE_FORMAT,
    ):
      self._pa.terminate()
      raise RuntimeError(
        f"Input device {self.device_index!r} ({self.device_name}) does not support "
        f"{self.sample_rate} Hz int16 mono. "
        f"Run: python scripts/check_mic.py --list-devices "
        f"and set audio.input_device in config/default.yaml"
      )

    try:
      self._stream = self._pa.open(
        format=CAPTURE_FORMAT,
        channels=self.channels,
        rate=self.sample_rate,
        input=True,
        frames_per_buffer=self.chunk_samples,
        input_device_index=self.device_index,
      )
      self._stream.start_stream()
    except Exception as exc:
      self._pa.terminate()
      raise RuntimeError(
        f"Failed to open mic {self.device_index!r} ({self.device_name}) "
        f"at {self.sample_rate} Hz int16 mono"
      ) from exc

    logger.info(
      "Mic opened: %r (index=%s) %s Hz int16 mono, chunk=%s samples (%s ms)",
      self.device_name,
      self.device_index,
      self.sample_rate,
      self.chunk_samples,
      chunk_ms,
    )

  def _resolve_device_index(self, device_index: int | None) -> int:
    if device_index is not None:
      return int(device_index)
    default = self._pa.get_default_input_device_info()
    return int(default["index"])

  def read(self) -> AudioChunk:
    raw = self._stream.read(self.chunk_samples, exception_on_overflow=False)
    samples = int16_bytes_to_float32(raw, channels=self.channels)
    return AudioChunk(samples=samples, sample_rate=self.sample_rate)

  def close(self) -> None:
    self._stream.stop_stream()
    self._stream.close()
    self._pa.terminate()


class AudioPlayback:
  def __init__(
    self,
    *,
    sample_rate: int,
    channels: int = 1,
    device_index: int | None = None,
    frames_per_buffer: int = 2048,
    write_chunk_frames: int = 1024,
  ) -> None:
    self.sample_rate = sample_rate
    self.channels = channels
    self.device_index = device_index
    self._frame_bytes = channels * 2
    stream_bytes = max(self._frame_bytes, frames_per_buffer * self._frame_bytes)
    chunk_bytes = max(self._frame_bytes, write_chunk_frames * self._frame_bytes)
    self._write_chunk_bytes = min(chunk_bytes, stream_bytes)
    self._write_buffer = bytearray()
    self._lock = threading.Lock()
    self._aborted = False
    self._pa = pyaudio.PyAudio()
    self._stream = self._pa.open(
      format=CAPTURE_FORMAT,
      channels=channels,
      rate=sample_rate,
      output=True,
      output_device_index=device_index,
      frames_per_buffer=frames_per_buffer,
    )

  def play_int16(self, pcm: bytes) -> None:
    if not pcm:
      return
    with self._lock:
      if self._aborted:
        return
      self._write_buffer.extend(pcm)
      while len(self._write_buffer) >= self._write_chunk_bytes:
        block = bytes(self._write_buffer[: self._write_chunk_bytes])
        del self._write_buffer[: self._write_chunk_bytes]
        self._stream.write(block, exception_on_underflow=False)

  def warmup(self, silence_ms: int = 150) -> None:
    """Prime the output device so the first real TTS phrase is not clipped."""
    with self._lock:
      self._aborted = False
    n_samples = max(1, int(self.sample_rate * silence_ms / 1000))
    silence = b"\x00\x00" * n_samples
    self.play_int16(silence)
    self.flush()

  def flush(self) -> None:
    with self._lock:
      if not self._write_buffer:
        return
      self._stream.write(bytes(self._write_buffer), exception_on_underflow=False)
      self._write_buffer.clear()

  def abort(self) -> None:
    """Drop buffered PCM that has not been written yet (barge-in / interrupt)."""
    with self._lock:
      self._aborted = True
      self._write_buffer.clear()

  def resume(self) -> None:
    """Allow playback again after :meth:`abort`."""
    with self._lock:
      self._aborted = False

  def play_float32_chunks(self, chunks: Iterator[np.ndarray]) -> None:
    for chunk in chunks:
      pcm = np.clip(chunk * 32767.0, -32768, 32767).astype(np.int16).tobytes()
      self.play_int16(pcm)

  def close(self) -> None:
    self.flush()
    self._stream.stop_stream()
    self._stream.close()
    self._pa.terminate()
