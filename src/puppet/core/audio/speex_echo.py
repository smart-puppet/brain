from __future__ import annotations

import ctypes
import ctypes.util
import logging
from typing import Any

import numpy as np

from puppet.core.audio.buffer import AudioReference

logger = logging.getLogger(__name__)

SPEEX_ECHO_SET_SAMPLING_RATE = 24


class SpeexDspUnavailable(RuntimeError):
  pass


def _load_speexdsp() -> ctypes.CDLL:
  lib_name = ctypes.util.find_library("speexdsp")
  if not lib_name:
    raise SpeexDspUnavailable(
      "libspeexdsp not found. Install: sudo apt install libspeexdsp-dev"
    )
  lib = ctypes.CDLL(lib_name)
  lib.speex_echo_state_init.argtypes = [ctypes.c_int, ctypes.c_int]
  lib.speex_echo_state_init.restype = ctypes.c_void_p
  lib.speex_echo_state_destroy.argtypes = [ctypes.c_void_p]
  lib.speex_echo_state_destroy.restype = None
  lib.speex_echo_cancellation.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_int16),
    ctypes.POINTER(ctypes.c_int16),
    ctypes.POINTER(ctypes.c_int16),
  ]
  lib.speex_echo_cancellation.restype = None
  lib.speex_echo_state_reset.argtypes = [ctypes.c_void_p]
  lib.speex_echo_state_reset.restype = None
  lib.speex_echo_ctl.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
  lib.speex_echo_ctl.restype = None
  return lib


def float32_to_int16(samples: np.ndarray) -> np.ndarray:
  return np.clip(samples * 32767.0, -32768, 32767).astype(np.int16)


def int16_to_float32(samples: np.ndarray) -> np.ndarray:
  return (samples.astype(np.float32) / 32768.0).astype(np.float32)


class EchoCanceller:
  """Acoustic echo cancellation via libspeexdsp."""

  def __init__(
    self,
    reference: AudioReference,
    *,
    sample_rate: int = 16000,
    frame_size: int = 160,
    filter_length: int = 2048,
    suppress_stt_on_echo: bool = True,
    echo_ratio_threshold: float = 0.55,
    min_reference_rms: float = 0.006,
    min_mic_rms: float = 0.008,
    lib: ctypes.CDLL | None = None,
  ) -> None:
    if frame_size <= 0:
      raise ValueError("frame_size must be positive")
    if filter_length < frame_size:
      raise ValueError("filter_length must be >= frame_size")

    self._reference = reference
    self._sample_rate = sample_rate
    self._frame_size = frame_size
    self._suppress_stt_on_echo = suppress_stt_on_echo
    self._echo_ratio_threshold = echo_ratio_threshold
    self._min_reference_rms = min_reference_rms
    self._min_mic_rms = min_mic_rms
    self._enabled = True
    self._lib = lib or _load_speexdsp()
    self._state = self._lib.speex_echo_state_init(frame_size, filter_length)
    if not self._state:
      raise SpeexDspUnavailable("speex_echo_state_init failed")
    rate = ctypes.c_int(sample_rate)
    self._lib.speex_echo_ctl(
      self._state,
      SPEEX_ECHO_SET_SAMPLING_RATE,
      ctypes.byref(rate),
    )
    self._rec = (ctypes.c_int16 * frame_size)()
    self._play = (ctypes.c_int16 * frame_size)()
    self._out = (ctypes.c_int16 * frame_size)()
    self._frame_i16 = np.zeros(frame_size, dtype=np.int16)

  @classmethod
  def from_config(
    cls,
    reference: AudioReference,
    aec_cfg: dict,
    *,
    sample_rate: int,
  ) -> EchoCanceller:
    return cls(
      reference,
      sample_rate=sample_rate,
      frame_size=int(aec_cfg.get("frame_size", 160)),
      filter_length=int(aec_cfg.get("filter_length", 2048)),
      suppress_stt_on_echo=bool(aec_cfg.get("suppress_stt_on_echo", True)),
      echo_ratio_threshold=float(aec_cfg.get("echo_ratio_threshold", 0.55)),
      min_reference_rms=float(aec_cfg.get("min_reference_rms", 0.006)),
      min_mic_rms=float(aec_cfg.get("min_mic_rms", 0.008)),
    )

  @property
  def enabled(self) -> bool:
    return self._enabled

  @enabled.setter
  def enabled(self, value: bool) -> None:
    self._enabled = value

  def close(self) -> None:
    if self._state:
      self._lib.speex_echo_state_destroy(self._state)
      self._state = None

  def __del__(self) -> None:
    try:
      self.close()
    except Exception:
      pass

  def reset_state(self) -> None:
    if self._state:
      self._lib.speex_echo_state_reset(self._state)

  def process(self, mic: np.ndarray, *, adapt: bool = False) -> np.ndarray:
    del adapt  # Speex adapts internally.
    if not self._enabled:
      return mic.astype(np.float32, copy=True)

    ref = self._reference.read_for_cancel(mic.size)
    if mic.size == 0:
      return mic.astype(np.float32, copy=True)

    from puppet.core.audio.aec import rms_energy

    if rms_energy(ref) < self._min_reference_rms:
      return mic.astype(np.float32, copy=True)

    mic_i16 = float32_to_int16(mic)
    ref_i16 = float32_to_int16(ref)
    out_i16 = np.zeros(mic_i16.shape, dtype=np.int16)
    fs = self._frame_size

    for off in range(0, mic_i16.size, fs):
      n = min(fs, mic_i16.size - off)
      self._frame_i16.fill(0)
      self._frame_i16[:n] = mic_i16[off : off + n]
      self._rec[:] = self._frame_i16
      self._frame_i16.fill(0)
      self._frame_i16[:n] = ref_i16[off : off + n]
      self._play[:] = self._frame_i16
      self._lib.speex_echo_cancellation(
        self._state,
        self._rec,
        self._play,
        self._out,
      )
      out_i16[off : off + n] = np.ctypeslib.as_array(self._out)[:n]

    return int16_to_float32(out_i16)

  def should_suppress_stt(self, mic: np.ndarray, clean: np.ndarray) -> bool:
    from puppet.core.audio.aec import should_suppress_echo_stt

    return should_suppress_echo_stt(
      mic,
      clean,
      reference=self._reference,
      enabled=self._enabled,
      suppress_stt_on_echo=self._suppress_stt_on_echo,
      echo_ratio_threshold=self._echo_ratio_threshold,
      min_reference_rms=self._min_reference_rms,
      min_mic_rms=self._min_mic_rms,
    )


def create_echo_canceller(
  reference: AudioReference,
  aec_cfg: dict[str, Any],
  *,
  sample_rate: int,
) -> EchoCanceller:
  try:
    return EchoCanceller.from_config(reference, aec_cfg, sample_rate=sample_rate)
  except SpeexDspUnavailable as exc:
    raise RuntimeError(
      f"SpeexDSP AEC unavailable: {exc}. Install: sudo apt install libspeexdsp-dev"
    ) from exc
