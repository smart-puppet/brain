from __future__ import annotations

import logging
import queue
import threading
import time
from typing import Callable, Protocol

import numpy as np

from puppet.tts.base import TtsBackend

logger = logging.getLogger(__name__)

_SHUTDOWN = object()
_PHRASE_END = object()


class PhrasePlayback(Protocol):
  def submit(self, text: str) -> None: ...

  def wait_done(self) -> None: ...

  def stop(self) -> None: ...


class PhraseTtsPipeline:
  """Prefetch phrase synthesis while the previous phrase plays."""

  def __init__(
    self,
    tts: TtsBackend,
    *,
    play_chunk: Callable[[str, np.ndarray], None],
    on_phrase_begin: Callable[[str], None] | None = None,
    on_phrase_end: Callable[[str], None] | None = None,
  ) -> None:
    self._tts = tts
    self._play_chunk = play_chunk
    self._on_phrase_begin = on_phrase_begin
    self._on_phrase_end = on_phrase_end
    self._phrase_queue: queue.Queue[str | object] = queue.Queue()
    self._audio_queue: queue.Queue[tuple[str, np.ndarray | object]] = queue.Queue()
    self._lock = threading.Lock()
    self._running = False
    self._stopped = False
    self._synth_thread: threading.Thread | None = None
    self._play_thread: threading.Thread | None = None
    self._inflight_cond = threading.Condition()
    self._phrases_in_flight = 0

  def _ensure_running(self) -> None:
    with self._lock:
      if self._running:
        return
      self._stopped = False
      self._running = True
      self._synth_thread = threading.Thread(
        target=self._synth_loop,
        daemon=True,
        name="puppet-tts-synth",
      )
      self._play_thread = threading.Thread(
        target=self._play_loop,
        daemon=True,
        name="puppet-tts-play",
      )
      self._synth_thread.start()
      self._play_thread.start()

  def submit(self, text: str) -> None:
    text = text.strip()
    if not text:
      return
    self._ensure_running()
    with self._inflight_cond:
      self._phrases_in_flight += 1
    self._phrase_queue.put(text)

  def is_busy(self) -> bool:
    with self._inflight_cond:
      return self._phrases_in_flight > 0

  def wait_done(self, timeout: float | None = None) -> None:
    deadline = None if timeout is None else time.monotonic() + timeout
    with self._inflight_cond:
      while self._phrases_in_flight > 0:
        if deadline is None:
          self._inflight_cond.wait()
          continue
        remaining = deadline - time.monotonic()
        if remaining <= 0:
          break
        self._inflight_cond.wait(timeout=remaining)

  def stop(self) -> None:
    with self._lock:
      if not self._running:
        return
      self._stopped = True
      self._phrase_queue.put(_SHUTDOWN)
      self._audio_queue.put((_SHUTDOWN, _SHUTDOWN))
    self._tts.stop()
    synth = self._synth_thread
    play = self._play_thread
    if synth and synth.is_alive() and synth is not threading.current_thread():
      synth.join(timeout=2.0)
    if play and play.is_alive() and play is not threading.current_thread():
      play.join(timeout=2.0)
    with self._lock:
      self._running = False
      self._stopped = False
      self._synth_thread = None
      self._play_thread = None
      self._phrase_queue = queue.Queue()
      self._audio_queue = queue.Queue()
    with self._inflight_cond:
      self._phrases_in_flight = 0
      self._inflight_cond.notify_all()

  def _synth_loop(self) -> None:
    while True:
      item = self._phrase_queue.get()
      if item is _SHUTDOWN:
        break
      phrase = str(item)
      if self._stopped:
        self._mark_phrase_done()
        continue
      if self._on_phrase_begin is not None:
        self._on_phrase_begin(phrase)
      try:
        for chunk in self._tts.synthesize_stream(phrase):
          if self._stopped:
            self._tts.stop()
            break
          self._audio_queue.put((phrase, chunk))
      except Exception:
        logger.exception("TTS synthesis failed for phrase: %r", phrase)
      self._audio_queue.put((phrase, _PHRASE_END))

  def _play_loop(self) -> None:
    while True:
      phrase, chunk = self._audio_queue.get()
      if phrase is _SHUTDOWN:
        break
      if chunk is _PHRASE_END:
        if not self._stopped and self._on_phrase_end is not None:
          self._on_phrase_end(phrase)
        self._mark_phrase_done()
        continue
      if self._stopped:
        continue
      self._play_chunk(phrase, chunk)

  def _mark_phrase_done(self) -> None:
    with self._inflight_cond:
      self._phrases_in_flight = max(0, self._phrases_in_flight - 1)
      self._inflight_cond.notify_all()
