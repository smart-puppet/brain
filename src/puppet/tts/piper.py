from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import onnxruntime
from piper import PiperVoice
from piper.config import PiperConfig

from puppet.tts.base import TtsBackend

logger = logging.getLogger(__name__)


def _load_piper_voice(
  model_path: str,
  config_path: str | None,
  *,
  use_cuda: bool,
  n_threads: int,
) -> PiperVoice:
  if config_path is None:
    config_path = f"{model_path}.json"

  with open(config_path, encoding="utf-8") as config_file:
    config_dict = json.load(config_file)

  if use_cuda:
    providers: list[str | tuple[str, dict[str, Any]]] = [
      ("CUDAExecutionProvider", {"cudnn_conv_algo_search": "HEURISTIC"}),
    ]
  else:
    providers = ["CPUExecutionProvider"]

  sess_options = onnxruntime.SessionOptions()
  if n_threads > 0:
    sess_options.intra_op_num_threads = n_threads
    sess_options.inter_op_num_threads = n_threads

  session = onnxruntime.InferenceSession(
    str(model_path),
    sess_options=sess_options,
    providers=providers,
  )
  return PiperVoice(
    config=PiperConfig.from_dict(config_dict),
    session=session,
  )


class PiperTts(TtsBackend):
  """TTS via piper-tts Python API."""

  def __init__(
    self,
    model_path: str,
    config_path: str | None = None,
    *,
    device: str = "cpu",
    speaker_id: int | None = None,
    n_threads: int = 0,
  ) -> None:
    self._speaker_id = speaker_id
    self._stopped = False
    self._n_threads = max(0, int(n_threads))
    use_cuda = device.lower() == "cuda"
    self._voice = _load_piper_voice(
      model_path=model_path,
      config_path=config_path,
      use_cuda=use_cuda,
      n_threads=self._n_threads,
    )
    self._sample_rate = self._detect_sample_rate(config_path)
    thread_note = str(self._n_threads) if self._n_threads > 0 else "default"
    logger.info(
      "Loaded Piper voice: %s (%d Hz, n_threads=%s)",
      model_path,
      self._sample_rate,
      thread_note,
    )

  def _detect_sample_rate(self, config_path: str | None) -> int:
    if config_path and Path(config_path).is_file():
      cfg = json.loads(Path(config_path).read_text(encoding="utf-8"))
      return int(cfg.get("audio", {}).get("sample_rate", 22050))
    return int(getattr(self._voice.config, "sample_rate", 22050))

  def synthesize_stream(self, text: str) -> Iterator[np.ndarray]:
    self._stopped = False
    text = text.strip()
    if not text:
      return

    if hasattr(self._voice, "synthesize_stream_raw"):
      for chunk in self._voice.synthesize_stream_raw(text, speaker_id=self._speaker_id):
        if self._stopped:
          break
        yield np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0
      return

    for chunk in self._voice.synthesize(text):
      if self._stopped:
        break
      yield np.frombuffer(chunk.audio_int16_bytes, dtype=np.int16).astype(np.float32) / 32768.0

  def sample_rate(self) -> int:
    return self._sample_rate

  def stop(self) -> None:
    self._stopped = True

  def warmup(self, *, text: str = ".") -> None:
    """Run a tiny synthesis so the first spoken reply avoids cold-start latency."""
    started = time.monotonic()
    samples = 0
    try:
      for chunk in self.synthesize_stream(text):
        samples += int(chunk.size)
    except Exception as exc:
      logger.warning("TTS warmup failed: %s", exc)
      return
    elapsed_ms = (time.monotonic() - started) * 1000.0
    audio_ms = (samples / self._sample_rate) * 1000.0 if self._sample_rate else 0.0
    logger.info(
      "TTS warmup complete (%.0f ms synth, %.0f ms audio, text=%r)",
      elapsed_ms,
      audio_ms,
      text,
    )


def create_tts(config: dict[str, Any]) -> TtsBackend:
  tts_cfg = config.get("tts", {})
  backend = tts_cfg.get("backend", "piper")
  if backend != "piper":
    raise ValueError(f"Unsupported TTS backend: {backend}")
  tts = PiperTts(
    model_path=tts_cfg["model_path"],
    config_path=tts_cfg.get("config_path"),
    device=tts_cfg.get("device", "cpu"),
    speaker_id=tts_cfg.get("speaker_id"),
    n_threads=int(tts_cfg.get("n_threads", 0)),
  )
  if tts_cfg.get("warmup", True):
    tts.warmup(text=str(tts_cfg.get("warmup_text", ".")))
  return tts
