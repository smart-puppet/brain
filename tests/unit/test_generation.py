from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock

import numpy as np

from puppet.orchestrator.generation import GenerationWorker
from puppet.orchestrator.tts_pipeline import PhraseTtsPipeline
from puppet.tts.types import TtsChunk


def _worker(**kwargs: object) -> GenerationWorker:
  defaults = {
    "phrase_delimiters": ".?!\n",
    "min_phrase_chars": 8,
    "min_first_phrase_chars": 8,
    "first_phrase_max_wait_ms": 0,
    "phrase_playback": MagicMock(),
  }
  defaults.update(kwargs)
  return GenerationWorker(MagicMock(), **defaults)


def test_phrase_waits_for_sentence_end_not_comma() -> None:
  worker = _worker()
  assert not worker._phrase_ready("Kace est joyeux,", first=False)
  assert worker._phrase_ready("Kace est joyeux.", first=False)


def test_first_phrase_timeout_when_configured() -> None:
  worker = _worker(first_phrase_max_wait_ms=300)
  assert not worker._first_phrase_ready("Kace", first_token_at=0.0)
  assert worker._first_phrase_ready("Kace est", first_token_at=time.monotonic() - 0.35)


class SlowChunkTts:
  def __init__(self) -> None:
    self._stopped = False
    self.synth_started: list[str] = []

  def synthesize_stream(self, text: str):
    self.synth_started.append(text)
    yield TtsChunk(samples=np.zeros(1600, dtype=np.float32))

  def sample_rate(self) -> int:
    return 22050

  def stop(self) -> None:
    self._stopped = True


def test_defer_tts_does_not_submit() -> None:
  playback = MagicMock()
  worker = _worker(phrase_playback=playback, defer_tts=lambda: True)
  worker._submit_phrase("Je vois un lit a gauche.")
  playback.submit.assert_not_called()
  assert worker.suppressed_phrases == 0


def test_phrase_clean_strips_robot_tags() -> None:
  from puppet.play.actions import strip_robot_actions

  playback = MagicMock()
  worker = _worker(phrase_playback=playback, phrase_clean=strip_robot_actions)
  worker._submit_phrase("Okay! I will follow you. <<follow>>")
  playback.submit.assert_called_once()
  spoken = playback.submit.call_args[0][0]
  assert "<<" not in spoken
  assert "Okay" in spoken
  worker._submit_phrase("<<seek>>")
  assert playback.submit.call_count == 1


def test_pipeline_prefetches_next_phrase_while_playing() -> None:
  tts = SlowChunkTts()
  play_started = threading.Event()
  play_count = {"n": 0}

  def play_chunk(_phrase: str, _chunk: np.ndarray) -> None:
    play_count["n"] += 1
    if play_count["n"] == 1:
      play_started.set()
    time.sleep(0.12)

  pipeline = PhraseTtsPipeline(
    tts,  # type: ignore[arg-type]
    play_chunk=play_chunk,
    on_phrase_end=lambda _phrase: None,
  )
  pipeline.submit("First phrase.")
  assert play_started.wait(timeout=1.0)
  pipeline.submit("Second phrase.")
  deadline = time.monotonic() + 0.08
  while time.monotonic() < deadline and len(tts.synth_started) < 2:
    time.sleep(0.005)
  assert tts.synth_started == ["First phrase.", "Second phrase."]
  assert play_count["n"] == 1
  pipeline.wait_done(timeout=2.0)
  pipeline.stop()
