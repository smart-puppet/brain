from __future__ import annotations

import logging
import re
import time
from pathlib import Path
from typing import Any

import numpy as np

from puppet.core.types import TranscriptSegment
from puppet.stt.base import SttBackend

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
FRAME_MS = 80

# Nemotron att_context_size presets: {chunk_ms: (left, right)} in 80 ms frames.
# See https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b
STREAMING_CHUNK_MS_TO_ATT_CONTEXT: dict[int, tuple[int, int]] = {
  80: (56, 0),
  160: (56, 1),
  320: (56, 3),
  560: (56, 6),
  1120: (56, 13),
}

_LANG_TAG_RE = re.compile(r"\s*<[a-z]{2}(?:-[A-Z]{2})?>\s*")


def att_context_for_chunk_ms(chunk_ms: int) -> tuple[int, int]:
  try:
    return STREAMING_CHUNK_MS_TO_ATT_CONTEXT[chunk_ms]
  except KeyError as exc:
    known = ", ".join(str(ms) for ms in sorted(STREAMING_CHUNK_MS_TO_ATT_CONTEXT))
    raise ValueError(f"Unsupported streaming chunk_ms={chunk_ms}. Choose one of: {known}") from exc


def strip_lang_tags(text: str) -> str:
  """Remove nemotron language-ID tags like ``<en-US>`` from transcript text."""
  return _LANG_TAG_RE.sub(" ", text)


class ParakeetStt(SttBackend):
  """STT via parakeet.cpp pybind11 binding."""

  def __init__(
    self,
    model_path: str,
    language: str = "en-US",
    *,
    streaming_chunk_ms: int = 320,
    strip_lang_tags: bool = True,
    n_threads: int = 0,
    n_batch: int = 1,
  ) -> None:
    self._model_path = model_path
    self._language = language
    self._streaming_chunk_ms = streaming_chunk_ms
    self._strip_lang_tags = strip_lang_tags
    self._n_threads = max(0, int(n_threads))
    self._n_batch = max(1, int(n_batch))
    self._feed_samples = max(1, int(SAMPLE_RATE * FRAME_MS / 1000))
    self._ctx = None
    self._stream = None
    self._pending = np.zeros(0, dtype=np.float32)
    self._load()

  def _load(self) -> None:
    try:
      import puppet_parakeet as pk  # type: ignore[import-not-found]
    except ImportError as exc:
      raise RuntimeError(
        "puppet_parakeet binding not installed. Run ./scripts/build_parakeet.sh"
      ) from exc

    path = Path(self._model_path)
    if not path.is_file():
      raise FileNotFoundError(f"STT model not found: {path}")

    self._pk = pk
    if self._n_threads > 0:
      pk.set_num_threads(self._n_threads)
    self._ctx = pk.load(str(path))
    left, right = att_context_for_chunk_ms(self._streaming_chunk_ms)
    self._ctx.set_att_context(left, right)
    self._stream = None
    self._pending = np.zeros(0, dtype=np.float32)
    thread_note = str(self._n_threads) if self._n_threads > 0 else "default"
    if self._n_batch != 1:
      logger.debug(
        "STT n_batch=%d is stored for config parity; streaming Nemotron ignores batch size",
        self._n_batch,
      )
    logger.info(
      "Loaded parakeet model: %s (lang=%s, chunk_ms=%d, att_context=[%d,%d], "
      "n_threads=%s, n_batch=%d)",
      path,
      self._language,
      self._streaming_chunk_ms,
      left,
      right,
      thread_note,
      self._n_batch,
    )

  def _ensure_stream(self) -> None:
    if self._stream is not None:
      return
    self._stream = self._ctx.stream_begin_lang(self._language)

  def _prepare_pcm(self, pcm: np.ndarray, sample_rate: int) -> np.ndarray:
    if pcm.ndim != 1:
      pcm = pcm.reshape(-1)
    if sample_rate != SAMPLE_RATE:
      from puppet.core.audio.pcm import resample_linear

      pcm = resample_linear(pcm.astype(np.float32, copy=False), sample_rate, SAMPLE_RATE)
    return np.clip(pcm.astype(np.float32, copy=False), -1.0, 1.0)

  def _postprocess_text(self, text: str) -> str:
    if not text or not self._strip_lang_tags:
      return text
    return strip_lang_tags(text)

  def _feed_block(self, pcm: np.ndarray) -> tuple[str, bool]:
    self._ensure_stream()
    assert self._stream is not None
    try:
      text, eou = self._stream.feed(pcm, SAMPLE_RATE)
    except Exception:
      logger.exception("STT feed failed; restarting parakeet stream")
      self.reset()
      self._ensure_stream()
      assert self._stream is not None
      text, eou = self._stream.feed(pcm, SAMPLE_RATE)
    text = self._postprocess_text(text)
    return text, eou

  def feed(self, pcm: np.ndarray, sample_rate: int) -> TranscriptSegment | None:
    pcm = self._prepare_pcm(pcm, sample_rate)
    if pcm.size == 0:
      return None

    self._pending = np.concatenate((self._pending, pcm))
    texts: list[str] = []
    saw_eou = False
    while self._pending.size >= self._feed_samples:
      block = self._pending[: self._feed_samples]
      self._pending = self._pending[self._feed_samples :]
      text, eou = self._feed_block(block)
      if text:
        texts.append(text)
      if eou:
        saw_eou = True

    if texts:
      return TranscriptSegment(text="".join(texts), is_final=True, end_of_utterance=saw_eou)
    if saw_eou:
      return TranscriptSegment(text="", is_final=True, end_of_utterance=True)
    return None

  def finalize(self) -> TranscriptSegment | None:
    if self._stream is None:
      return None

    tail_text = ""
    if self._pending.size > 0:
      text, _eou = self._feed_block(self._pending)
      self._pending = np.zeros(0, dtype=np.float32)
      tail_text = text

    text = self._postprocess_text(self._stream.finalize())
    self._stream.close()
    self._stream = None
    self._pending = np.zeros(0, dtype=np.float32)

    combined = f"{tail_text}{text}".strip()
    if not combined:
      return None
    return TranscriptSegment(text=combined, is_final=True, end_of_utterance=False)

  def reset(self) -> None:
    self._pending = np.zeros(0, dtype=np.float32)
    if self._stream is not None:
      self._stream.close()
      self._stream = None

  def warmup(self, *, duration_ms: int = 1500, sample_rate: int = SAMPLE_RATE) -> None:
    """Prime CUDA graphs and streaming decoder with silence before the first utterance."""
    started = time.monotonic()
    n_samples = max(1, int(sample_rate * duration_ms / 1000))
    silence = np.zeros(n_samples, dtype=np.float32)
    chunk_samples = max(self._feed_samples, int(sample_rate * 20 / 1000))
    for offset in range(0, n_samples, chunk_samples):
      self.feed(silence[offset : offset + chunk_samples], sample_rate)
    self.reset()
    elapsed_ms = (time.monotonic() - started) * 1000.0
    logger.info("STT warmup complete (%.0f ms, %d ms silence)", elapsed_ms, duration_ms)

  def suspend(self) -> None:
    """Unload parakeet GPU weights while the LLM decodes."""
    self.reset()
    if self._ctx is not None:
      self._ctx.close()
      self._ctx = None
      logger.debug("STT model unloaded from GPU for LLM decode")

  def resume(self) -> None:
    if self._ctx is not None:
      return
    self._load()
    logger.debug("STT model reloaded after LLM decode")

  def close(self) -> None:
    self.reset()
    if self._ctx is not None:
      self._ctx.close()
      self._ctx = None


def create_stt(config: dict[str, Any]) -> SttBackend:
  stt_cfg = config.get("stt", {})
  backend = stt_cfg.get("backend", "parakeet")
  if backend != "parakeet":
    raise ValueError(f"Unsupported STT backend: {backend}")

  stream_cfg = stt_cfg.get("streaming", {})
  if isinstance(stream_cfg, bool):
    stream_cfg = {} if stream_cfg else {"chunk_ms": 320}

  stt = ParakeetStt(
    model_path=stt_cfg["model_path"],
    language=stt_cfg.get("language", "en-US"),
    streaming_chunk_ms=int(stream_cfg.get("chunk_ms", 320)),
    strip_lang_tags=bool(stream_cfg.get("strip_lang_tags", True)),
    n_threads=int(stt_cfg.get("n_threads", 0)),
    n_batch=int(stt_cfg.get("n_batch", 1)),
  )
  if stt_cfg.get("warmup", True):
    stt.warmup(duration_ms=int(stt_cfg.get("warmup_ms", 1500)))
  return stt
