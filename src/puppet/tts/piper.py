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

from puppet.tts.alignments import (
  MouthTiming,
  model_has_phoneme_duration,
  phoneme_hold_durations_ms,
  resolve_mouth_mode,
  word_alignments_to_timeline,
  word_cues_from_alignments,
  resolve_alignment_model_path,
  timeline_for_audio_chunk,
)
from puppet.tts.base import TtsBackend
from puppet.tts.types import TtsChunk

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
    include_alignments: bool = False,
    mouth_timing: MouthTiming | None = None,
    mouth_mode: str = "word",
  ) -> None:
    self._speaker_id = speaker_id
    self._stopped = False
    self._n_threads = max(0, int(n_threads))
    self._model_path = model_path
    self._mouth_timing = mouth_timing or MouthTiming()
    self._mouth_mode = mouth_mode
    use_cuda = device.lower() == "cuda"
    self._voice = _load_piper_voice(
      model_path=model_path,
      config_path=config_path,
      use_cuda=use_cuda,
      n_threads=self._n_threads,
    )
    self._include_alignments = include_alignments and model_has_phoneme_duration(model_path)
    self._sample_rate = self._detect_sample_rate(config_path)
    thread_note = str(self._n_threads) if self._n_threads > 0 else "default"
    align_note = "phoneme timeline" if self._include_alignments else "audio only"
    logger.info(
      "Loaded Piper voice: %s (%d Hz, n_threads=%s, %s)",
      model_path,
      self._sample_rate,
      thread_note,
      align_note,
    )

  @property
  def has_mouth_timeline(self) -> bool:
    return self._include_alignments

  def _detect_sample_rate(self, config_path: str | None) -> int:
    if config_path and Path(config_path).is_file():
      cfg = json.loads(Path(config_path).read_text(encoding="utf-8"))
      return int(cfg.get("audio", {}).get("sample_rate", 22050))
    return int(getattr(self._voice.config, "sample_rate", 22050))

  def synthesize_stream(self, text: str) -> Iterator[TtsChunk]:
    self._stopped = False
    text = text.strip()
    if not text:
      return

    if hasattr(self._voice, "synthesize_stream_raw"):
      for chunk in self._voice.synthesize_stream_raw(text, speaker_id=self._speaker_id):
        if self._stopped:
          break
        samples = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0
        yield TtsChunk(samples=samples)
      return

    for chunk in self._voice.synthesize(
      text,
      include_alignments=self._include_alignments,
    ):
      if self._stopped:
        break
      samples = chunk.audio_float_array.astype(np.float32, copy=False)
      timeline = None
      phoneme_hold_ms = None
      word_cues = None
      if self._include_alignments and self._mouth_mode == "word":
        timeline = timeline_for_audio_chunk(
          alignments=chunk.phoneme_alignments,
          audio_samples=int(samples.size),
          sample_rate=self._sample_rate,
          timing=self._mouth_timing,
          granularity="word",
        )
        if timeline:
          logger.debug(
            "TTS mouth timeline (%d word events) for %r",
            len(timeline),
            text[:60],
          )
        word_cues = word_cues_from_alignments(
          chunk.phoneme_alignments,
          sample_rate=self._sample_rate,
          min_open_ms=self._mouth_timing.min_open_ms,
          min_gap_ms=self._mouth_timing.word_min_gap_ms,
        )
        if word_cues:
          logger.debug(
            "TTS word cues (%d) for %r",
            len(word_cues),
            text[:60],
          )
      elif self._include_alignments and self._mouth_mode == "fallback":
        phoneme_hold_ms = phoneme_hold_durations_ms(
          chunk.phoneme_alignments,
          self._sample_rate,
          min_ms=self._mouth_timing.fallback_flip_ms,
        )
        if phoneme_hold_ms:
          varied = sum(1 for ms in phoneme_hold_ms if ms > self._mouth_timing.fallback_flip_ms)
          logger.debug(
            "TTS phoneme holds (%d, %d above %dms floor) for %r",
            len(phoneme_hold_ms),
            varied,
            self._mouth_timing.fallback_flip_ms,
            text[:60],
          )
      yield TtsChunk(
        samples=samples,
        mouth_timeline=timeline,
        phoneme_hold_ms=phoneme_hold_ms or None,
        word_cues=word_cues or None,
      )

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
        samples += int(chunk.samples.size)
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
  mouth_cfg = config.get("puppet", {}).get("mouth", {})
  mouth_enabled = bool(mouth_cfg.get("enabled", False))
  model_path = resolve_alignment_model_path(
    tts_cfg["model_path"],
    prefer_alignments=mouth_enabled,
  )
  if model_path != tts_cfg["model_path"]:
    logger.info("Using phoneme-duration TTS model: %s", model_path)
  mouth_mode = resolve_mouth_mode(config) if mouth_enabled else "word"
  tts = PiperTts(
    model_path=model_path,
    config_path=tts_cfg.get("config_path"),
    device=tts_cfg.get("device", "cpu"),
    speaker_id=tts_cfg.get("speaker_id"),
    n_threads=int(tts_cfg.get("n_threads", 0)),
    include_alignments=mouth_enabled,
    mouth_timing=MouthTiming.from_config(config),
    mouth_mode=mouth_mode,
  )
  if tts_cfg.get("warmup", True):
    tts.warmup(text=str(tts_cfg.get("warmup_text", ".")))
  return tts
