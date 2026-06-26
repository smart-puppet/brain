from __future__ import annotations

import logging
import queue
import re
import threading
import time
from typing import Callable

from puppet.core.types import Conversation
from puppet.llm.base import LlmBackend
from puppet.orchestrator.tts_pipeline import PhrasePlayback

logger = logging.getLogger(__name__)


class GenerationWorker:
  """Background LLM token stream with phrase-level TTS playback."""

  def __init__(
    self,
    llm: LlmBackend,
    *,
    phrase_delimiters: str,
    min_phrase_chars: int,
    min_first_phrase_chars: int = 8,
    first_phrase_max_wait_ms: int = 0,
    phrase_playback: PhrasePlayback,
  ) -> None:
    self._llm = llm
    self._phrase_delimiters = phrase_delimiters
    self._min_phrase_chars = min_phrase_chars
    self._min_first_phrase_chars = min_first_phrase_chars
    self._first_phrase_max_wait_s = max(0, first_phrase_max_wait_ms) / 1000.0
    self._phrase_playback = phrase_playback
    self._lock = threading.Lock()
    self._cancel = threading.Event()
    self._epoch = 0
    self._thread: threading.Thread | None = None

  @property
  def epoch(self) -> int:
    return self._epoch

  @property
  def active(self) -> bool:
    thread = self._thread
    return thread is not None and thread.is_alive()

  def _phrase_ready(self, buffer: str, *, first: bool) -> bool:
    min_chars = self._min_first_phrase_chars if first else self._min_phrase_chars
    if len(buffer) < min_chars:
      return False
    return bool(re.search(f"[{re.escape(self._phrase_delimiters)}]\\s*$", buffer))

  def _first_phrase_ready(self, buffer: str, *, first_token_at: float | None) -> bool:
    if self._phrase_ready(buffer, first=True):
      return True
    if self._first_phrase_max_wait_s <= 0 or first_token_at is None:
      return False
    if len(buffer) < self._min_first_phrase_chars:
      return False
    return time.monotonic() - first_token_at >= self._first_phrase_max_wait_s

  def start(
    self,
    conversation: Conversation,
    *,
    on_token: Callable[[str], None] | None = None,
    on_done: Callable[[str, int], None] | None = None,
  ) -> int:
    with self._lock:
      self.stop()
      self._epoch += 1
      epoch = self._epoch
      self._cancel.clear()
      self._thread = threading.Thread(
        target=self._run,
        args=(conversation.snapshot(), epoch, on_token, on_done),
        daemon=True,
        name="puppet-generation",
      )
      self._thread.start()
      return epoch

  def stop(self) -> None:
    self._cancel.set()
    self._llm.cancel()
    self._phrase_playback.stop()
    thread = self._thread
    if thread and thread.is_alive() and thread is not threading.current_thread():
      thread.join(timeout=2.0)
    self._thread = None

  def _produce_tokens(
    self,
    conversation: Conversation,
    token_queue: queue.Queue[str | None],
    epoch: int,
  ) -> None:
    try:
      for token in self._llm.stream_reply(conversation):
        if self._cancel.is_set() or epoch != self._epoch:
          break
        token_queue.put(token)
    except Exception:
      logger.exception("LLM generation failed")
    finally:
      token_queue.put(None)

  def _submit_phrase(self, text: str) -> None:
    self._phrase_playback.submit(text)

  def _run(
    self,
    snap,
    epoch: int,
    on_token: Callable[[str], None] | None,
    on_done: Callable[[str, int], None] | None,
  ) -> None:
    conversation = Conversation()
    conversation.restore(snap)
    token_buffer = ""
    full_reply: list[str] = []
    first_phrase = True
    first_token_at: float | None = None
    token_queue: queue.Queue[str | None] = queue.Queue()
    producer = threading.Thread(
      target=self._produce_tokens,
      args=(conversation, token_queue, epoch),
      daemon=True,
      name="puppet-llm-producer",
    )
    producer.start()

    try:
      while True:
        if self._cancel.is_set() or epoch != self._epoch:
          return
        try:
          token = token_queue.get(timeout=0.05)
        except queue.Empty:
          if not producer.is_alive():
            break
          if first_phrase and token_buffer and self._first_phrase_ready(
            token_buffer,
            first_token_at=first_token_at,
          ):
            self._submit_phrase(token_buffer)
            token_buffer = ""
            first_phrase = False
          continue
        if token is None:
          break
        if first_token_at is None:
          first_token_at = time.monotonic()
        token_buffer += token
        full_reply.append(token)
        if on_token:
          on_token(token)
        ready = (
          self._first_phrase_ready(token_buffer, first_token_at=first_token_at)
          if first_phrase
          else self._phrase_ready(token_buffer, first=False)
        )
        if not ready:
          continue
        self._submit_phrase(token_buffer)
        token_buffer = ""
        first_phrase = False

      if token_buffer and not self._cancel.is_set() and epoch == self._epoch:
        self._submit_phrase(token_buffer)
    finally:
      producer.join(timeout=2.0)
      if self._cancel.is_set() or epoch != self._epoch:
        self._phrase_playback.stop()
      else:
        self._phrase_playback.wait_done()
      if not self._cancel.is_set() and epoch == self._epoch and on_done:
        on_done("".join(full_reply).strip(), epoch)
