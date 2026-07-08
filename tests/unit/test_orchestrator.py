from __future__ import annotations

import time
from typing import Iterator

import numpy as np
import pytest

from puppet.core.audio.vad import PassthroughVad, VadEvent
from puppet.core.types import Conversation, PipelineState, TranscriptSegment
from puppet.llm.base import LlmBackend
from puppet.orchestrator.pipeline import Orchestrator
from puppet.stt.base import SttBackend
from puppet.tts.base import TtsBackend
from puppet.tts.types import TtsChunk


class FakeStt(SttBackend):
  def __init__(self) -> None:
    self._pending: TranscriptSegment | None = None
    self.fed_chunks = 0
    self.finalize_calls = 0
    self.reset_calls = 0

  def queue(self, segment: TranscriptSegment) -> None:
    self._pending = segment

  def feed(self, pcm: np.ndarray, sample_rate: int) -> TranscriptSegment | None:
    self.fed_chunks += 1
    seg = self._pending
    self._pending = None
    return seg

  def finalize(self) -> TranscriptSegment | None:
    self.finalize_calls += 1
    return None

  def reset(self) -> None:
    self.reset_calls += 1


class FakeLlm(LlmBackend):
  def __init__(self) -> None:
    pass

  def stream_reply(self, conversation: Conversation) -> Iterator[str]:
    yield "Hello there."

  def cancel(self) -> None:
    pass


class FakeTts(TtsBackend):
  def synthesize_stream(self, text: str) -> Iterator[TtsChunk]:
    yield TtsChunk(samples=np.zeros(100, dtype=np.float32))

  def sample_rate(self) -> int:
    return 22050

  def stop(self) -> None:
    pass


class FakePlayback:
  def __init__(self, *, busy: bool = False) -> None:
    self._busy = busy
    self.stop_calls = 0
    self.resume_calls = 0
    self.pause_calls = 0

  def submit(self, text: str) -> None:
    pass

  def wait_done(self) -> None:
    pass

  def stop(self) -> None:
    self.stop_calls += 1

  def pause(self) -> None:
    self.pause_calls += 1

  def resume(self) -> None:
    self.resume_calls += 1

  def is_busy(self) -> bool:
    return self._busy


class PlaybackProbe:
  def __init__(self) -> None:
    self.resume_calls = 0
    self.abort_calls = 0

  def resume(self) -> None:
    self.resume_calls += 1

  def abort(self) -> None:
    self.abort_calls += 1

  def reset_sample_clock(self) -> None:
    pass

  def warmup(self, silence_ms: int = 150) -> None:
    pass


class FakeVad(PassthroughVad):
  def __init__(self) -> None:
    self.reset_calls = 0

  def reset(self) -> None:
    self.reset_calls += 1


class StartOnceVad(PassthroughVad):
  def __init__(self) -> None:
    self._chunks = 0

  def feed(self, samples: np.ndarray) -> list[VadEvent]:
    self._chunks += 1
    if self._chunks < 2:
      return []
    if self._chunks == 2:
      return [VadEvent(kind="start")]
    return []


class GatedVad(PassthroughVad):
  def __init__(self) -> None:
    self._speech = False

  @property
  def is_speech(self) -> bool:
    return self._speech


def _base_cfg(**overrides: object) -> dict:
  cfg = {
    "audio": {"sample_rate": 16000, "channels": 1, "chunk_ms": 20},
    "puppet": {
      "barge_in_enabled": False,
      "min_phrase_chars": 1,
      "min_user_chars": 1,
      "stt_gap_ms": 50,
      "restart_on_partial": True,
    },
    "vad": {"enabled": False, "gate_stt": False},
    "stt": {"backend": "parakeet", "model_path": "x.gguf"},
    "llm": {"backend": "llama", "model_path": "x.gguf"},
    "tts": {"backend": "piper", "model_path": "x.onnx"},
  }
  if overrides:
    puppet = dict(cfg["puppet"])
    puppet.update({k: v for k, v in overrides.items() if k in puppet})
    cfg["puppet"] = puppet
  return cfg


def _with_fake_playback(orch: Orchestrator) -> Orchestrator:
  fake_playback = FakePlayback()
  orch._tts_pipeline = fake_playback  # type: ignore[assignment]
  orch._worker._phrase_playback = fake_playback
  return orch


@pytest.fixture
def orchestrator() -> Orchestrator:
  orch = Orchestrator(
    _base_cfg(),
    stt=FakeStt(),
    llm=FakeLlm(),
    tts=FakeTts(),
    vad=PassthroughVad(),
  )
  return _with_fake_playback(orch)


def _wait_for_generation(orch: Orchestrator, timeout: float = 2.0) -> None:
  deadline = time.monotonic() + timeout
  while orch._worker.active and time.monotonic() < deadline:
    time.sleep(0.01)


def test_eou_starts_streaming_generation(orchestrator: Orchestrator) -> None:
  orchestrator._set_state(PipelineState.LISTENING)
  orchestrator._on_transcript(TranscriptSegment(text="Hi", is_final=True))
  orchestrator._on_transcript(TranscriptSegment(text="", is_final=True, end_of_utterance=True))
  _wait_for_generation(orchestrator)
  assert orchestrator.conversation.messages[-2].content == "Hi"
  assert orchestrator.conversation.messages[-1].content == "Hello there."


def test_gap_starts_generation_without_eou(orchestrator: Orchestrator) -> None:
  orchestrator._set_state(PipelineState.LISTENING)
  orchestrator._on_transcript(TranscriptSegment(text="Hi"))
  orchestrator._last_stt_at = time.monotonic() - 1.0
  orchestrator._tick_gap()
  _wait_for_generation(orchestrator)
  assert orchestrator.conversation.messages[-1].role == "assistant"


def test_gap_waits_for_stt_tail(orchestrator: Orchestrator) -> None:
  orchestrator._set_state(PipelineState.LISTENING)
  orchestrator._on_transcript(TranscriptSegment(text="Je vais"))
  orchestrator._stt_tail_until = time.monotonic() + 0.5
  orchestrator._last_stt_at = time.monotonic() - 1.0
  orchestrator._tick_gap()
  assert not orchestrator._worker.active


def test_eou_during_stt_tail_is_deferred(orchestrator: Orchestrator) -> None:
  orchestrator._set_state(PipelineState.LISTENING)
  orchestrator._on_transcript(TranscriptSegment(text="Hi"))
  orchestrator._stt_tail_until = time.monotonic() + 0.5
  orchestrator._on_transcript(TranscriptSegment(text="", is_final=True, end_of_utterance=True))
  assert orchestrator._pending_stt_eou
  assert not orchestrator._worker.active


def test_partial_updates_draft(orchestrator: Orchestrator) -> None:
  orchestrator._on_stt_partial("hel")
  orchestrator._on_stt_partial("lo")
  assert orchestrator.conversation.draft_user == "hello"


def test_vad_during_speaking_does_not_reset_latency() -> None:
  cfg = _base_cfg()
  vad = StartOnceVad()
  orch = _with_fake_playback(
    Orchestrator(cfg, stt=FakeStt(), llm=FakeLlm(), tts=FakeTts(), vad=vad)
  )
  orch._set_state(PipelineState.SPEAKING)
  orch._latency._generation_start = 100.0
  orch._latency._vad_end = 90.0
  orch._handle_audio_chunk(np.zeros(320, dtype=np.float32), 16000)
  orch._handle_audio_chunk(np.zeros(320, dtype=np.float32), 16000)
  assert orch._latency._generation_start == 100.0
  assert orch._latency._vad_end == 90.0


def test_gap_waits_for_vad_silence() -> None:
  cfg = _base_cfg()
  cfg["vad"] = {"enabled": True, "gate_stt": True}
  gated_vad = GatedVad()
  orch = _with_fake_playback(
    Orchestrator(cfg, stt=FakeStt(), llm=FakeLlm(), tts=FakeTts(), vad=gated_vad)
  )
  orch._set_state(PipelineState.LISTENING)
  orch.conversation.draft_user = "Bonjour"
  orch._last_stt_at = time.monotonic() - 1.0
  gated_vad._speech = True
  orch._speech_active = True

  orch._tick_gap()

  assert not orch._worker.active

  orch._speech_active = False
  gated_vad._speech = False
  orch._tick_gap()

  assert orch._worker.active or orch.state == PipelineState.THINKING


def test_barge_in_only_during_speaking() -> None:
  cfg = _base_cfg()
  cfg["vad"] = {"enabled": True, "gate_stt": True, "barge_in": True}
  gated_vad = GatedVad()
  orch = _with_fake_playback(
    Orchestrator(cfg, stt=FakeStt(), llm=FakeLlm(), tts=FakeTts(), vad=gated_vad)
  )
  gated_vad._speech = True
  orch._set_state(PipelineState.THINKING)

  orch._handle_audio_chunk(np.zeros(320, dtype=np.float32), 16000)

  assert orch.state == PipelineState.THINKING


def test_cancel_reply_enters_echo_guard() -> None:
  cfg = _base_cfg()
  fake_playback = FakePlayback(busy=True)
  orch = Orchestrator(cfg, stt=FakeStt(), llm=FakeLlm(), tts=FakeTts())
  orch._tts_pipeline = fake_playback  # type: ignore[assignment]
  orch._worker._phrase_playback = fake_playback
  orch._recent_tts_phrases = ["Tu veux jouer ou discuter de quoi ?"]
  orch._reply_in_progress = True
  orch._set_state(PipelineState.SPEAKING)
  orch._cancel_reply()
  assert orch._await_fresh_speech
  assert orch._recent_tts_phrases == ["Tu veux jouer ou discuter de quoi ?"]
  orch._echo_quiet_until = 0.0
  orch._await_fresh_speech = False
  orch.conversation.draft_user = ""
  orch._on_stt_partial("discuté de quoi")
  assert orch.conversation.draft_user == "discuté de quoi"
  orch.conversation.draft_user = ""
  orch._recent_tts_phrases = ["Tu veux jouer ou discuter de quoi ?"]
  orch._mark_echo_risk(1.0)
  orch._await_fresh_speech = True
  orch._on_stt_partial("discuté de quoi")
  assert orch.conversation.draft_user == ""


def test_cancel_reply_stops_worker_and_clears_draft() -> None:
  cfg = _base_cfg()
  cfg["puppet"]["barge_in_enabled"] = True
  fake_playback = FakePlayback(busy=True)
  orch = Orchestrator(cfg, stt=FakeStt(), llm=FakeLlm(), tts=FakeTts())
  orch._tts_pipeline = fake_playback  # type: ignore[assignment]
  orch._worker._phrase_playback = fake_playback
  orch.conversation.draft_user = "Tell me a story"
  orch._reply_in_progress = True
  orch._set_state(PipelineState.SPEAKING)
  orch._cancel_reply()
  assert orch.state == PipelineState.LISTENING
  assert orch.conversation.draft_user == ""
  assert not orch._reply_in_progress
  assert fake_playback.stop_calls >= 1


def test_barge_in_cancel_on_sustained_clean_speech() -> None:
  cfg = _base_cfg()
  cfg["puppet"]["barge_in_enabled"] = True
  cfg["puppet"]["barge_in_clean_ms"] = 100
  cfg["puppet"]["barge_in_grace_ms"] = 0
  cfg["puppet"]["barge_in_clean_rms"] = 0.01
  fake_playback = FakePlayback(busy=True)
  orch = Orchestrator(cfg, stt=FakeStt(), llm=FakeLlm(), tts=FakeTts())
  orch._tts_pipeline = fake_playback  # type: ignore[assignment]
  orch._worker._phrase_playback = fake_playback
  orch._reply_in_progress = True
  orch._set_state(PipelineState.SPEAKING)
  orch._tts_playback_active = True
  orch._speaking_since = time.monotonic() - 10.0
  loud = np.full(320, 0.05, dtype=np.float32)
  orch._barge_clean_since = time.monotonic() - 0.2
  assert orch._tick_barge_in_cancel(loud)
  assert orch.state == PipelineState.LISTENING
  assert fake_playback.stop_calls >= 1


def test_fuzzy_tts_echo_detects_misheard_playback() -> None:
  cfg = _base_cfg()
  orch = _with_fake_playback(
    Orchestrator(cfg, stt=FakeStt(), llm=FakeLlm(), tts=FakeTts())
  )
  phrase = "Tu veux jouer ou discuter de quoi ?"
  orch._recent_tts_phrases = [phrase]
  orch._spoken_reply_corpus = phrase
  assert orch._text_looks_like_tts_echo("Joue où disputait de quoi?")


def test_fuzzy_tts_echo_detects_partial_phrase_bleed() -> None:
  cfg = _base_cfg()
  orch = _with_fake_playback(
    Orchestrator(cfg, stt=FakeStt(), llm=FakeLlm(), tts=FakeTts())
  )
  phrase = (
    "Dorothy a dit qu'elle avait le meilleur, "
    "puis Cha Cha a dit qu'elle avait le meilleur."
  )
  orch._recent_tts_phrases = [phrase]
  assert orch._text_looks_like_tts_echo("Puisach a dit qu'elle avait le meilleur")


def test_real_barge_in_not_classified_as_tts_echo() -> None:
  cfg = _base_cfg()
  orch = _with_fake_playback(
    Orchestrator(cfg, stt=FakeStt(), llm=FakeLlm(), tts=FakeTts())
  )
  orch._recent_tts_phrases = ["Tu veux jouer ou discuter de quoi ?"]
  assert not orch._text_looks_like_tts_echo("Comment tu t'appelles")


def test_barge_in_blocked_during_reply_grace() -> None:
  cfg = _base_cfg()
  orch = _with_fake_playback(
    Orchestrator(cfg, stt=FakeStt(), llm=FakeLlm(), tts=FakeTts())
  )
  orch._reply_in_progress = True
  orch._playback_started_at = time.monotonic()
  assert not orch._barge_in_allowed()


def test_barge_in_grace_covers_inter_phrase_gap() -> None:
  cfg = _base_cfg()
  cfg["puppet"]["barge_in_grace_ms"] = 1000
  orch = _with_fake_playback(
    Orchestrator(cfg, stt=FakeStt(), llm=FakeLlm(), tts=FakeTts())
  )
  orch._reply_in_progress = True
  orch._tts_playback_active = False
  orch._playback_started_at = time.monotonic() - 0.4
  assert not orch._barge_in_allowed()
  loud = np.full(320, 0.05, dtype=np.float32)
  orch._set_state(PipelineState.SPEAKING)
  orch._barge_clean_since = time.monotonic() - 0.5
  assert not orch._tick_barge_in_cancel(loud)


def test_barge_in_grace_is_per_reply_not_per_phrase() -> None:
  cfg = _base_cfg()
  cfg["puppet"]["barge_in_grace_ms"] = 1000
  orch = _with_fake_playback(
    Orchestrator(cfg, stt=FakeStt(), llm=FakeLlm(), tts=FakeTts())
  )
  orch._reply_in_progress = True
  orch._playback_started_at = time.monotonic() - 5.0
  orch._speaking_since = time.monotonic()
  assert orch._barge_in_allowed()


def test_phrase_end_keeps_playback_active_during_reply() -> None:
  cfg = _base_cfg()
  fake_playback = FakePlayback(busy=False)
  orch = Orchestrator(cfg, stt=FakeStt(), llm=FakeLlm(), tts=FakeTts())
  orch._tts_pipeline = fake_playback  # type: ignore[assignment]
  orch._reply_in_progress = True
  orch._tts_playback_active = True
  orch._on_tts_phrase_end("Oh oui !")
  assert orch._tts_playback_active


def test_echo_quiet_blocks_stt_drafts() -> None:
  cfg = _base_cfg()
  orch = _with_fake_playback(
    Orchestrator(cfg, stt=FakeStt(), llm=FakeLlm(), tts=FakeTts())
  )
  orch._set_state(PipelineState.LISTENING)
  orch._mark_echo_risk(1.0)
  orch._on_stt_partial("merci")
  assert orch.conversation.draft_user == ""


def test_unlock_fresh_speech_skips_stt_reset_while_user_speaks() -> None:
  cfg = _base_cfg()
  cfg["vad"] = {"enabled": True, "gate_stt": True}
  gated_vad = GatedVad()
  orch = _with_fake_playback(
    Orchestrator(cfg, stt=FakeStt(), llm=FakeLlm(), tts=FakeTts(), vad=gated_vad)
  )
  fake_stt = orch.stt
  assert isinstance(fake_stt, FakeStt)
  orch._set_state(PipelineState.LISTENING)
  orch._await_fresh_speech = True
  orch._echo_unlock_after = 0.0
  orch._echo_quiet_until = 0.0
  orch._speech_active = True
  resets_before = fake_stt.reset_calls

  orch._unlock_fresh_speech()

  assert not orch._await_fresh_speech
  assert fake_stt.reset_calls == resets_before


def test_is_stt_noise_tail() -> None:
  assert Orchestrator._is_stt_noise_tail("?")
  assert Orchestrator._is_stt_noise_tail(" . ")
  assert not Orchestrator._is_stt_noise_tail("oui")


def test_await_fresh_speech_ignores_partials_until_unlock() -> None:
  cfg = _base_cfg()
  cfg["vad"] = {"enabled": True, "gate_stt": True}
  gated_vad = GatedVad()
  orch = _with_fake_playback(
    Orchestrator(cfg, stt=FakeStt(), llm=FakeLlm(), tts=FakeTts(), vad=gated_vad)
  )
  orch._set_state(PipelineState.LISTENING)
  orch._await_fresh_speech = True
  orch._echo_unlock_after = time.monotonic() + 10.0
  orch._echo_quiet_until = 0.0
  orch.conversation.draft_user = "old question"

  orch._on_stt_partial("echo")
  assert orch.conversation.draft_user == "old question"

  gated_vad._speech = True
  orch._speech_active = True
  orch._echo_unlock_after = 0.0
  orch._unlock_fresh_speech()
  assert not orch._await_fresh_speech
  assert orch.conversation.draft_user == ""
  orch._on_transcript(TranscriptSegment(text="hello"))
  assert orch.conversation.draft_user == "hello"


def test_late_stt_partial_during_thinking_restarts_generation() -> None:
  cfg = _base_cfg()
  cfg["vad"] = {"enabled": True, "gate_stt": True}
  gated_vad = GatedVad()
  orch = _with_fake_playback(
    Orchestrator(cfg, stt=FakeStt(), llm=FakeLlm(), tts=FakeTts(), vad=gated_vad)
  )
  restarts: list[int] = []
  orch._restart_generation = lambda: restarts.append(1)  # type: ignore[method-assign]
  orch._generation_active = lambda: True  # type: ignore[method-assign]
  orch._set_state(PipelineState.THINKING)
  orch._speech_active = False
  gated_vad._speech = False
  orch.conversation.draft_user = "What is"
  orch._on_stt_partial(" the weather")
  assert restarts == [1]
  assert orch.conversation.draft_user == "What is the weather"


def test_stt_feeds_during_thinking_not_speaking() -> None:
  cfg = _base_cfg()
  cfg["vad"] = {"enabled": True, "gate_stt": True}
  gated_vad = GatedVad()
  orch = _with_fake_playback(
    Orchestrator(cfg, stt=FakeStt(), llm=FakeLlm(), tts=FakeTts(), vad=gated_vad)
  )
  fake_stt = orch.stt
  assert isinstance(fake_stt, FakeStt)

  gated_vad._speech = True
  orch._set_state(PipelineState.THINKING)
  orch._handle_audio_chunk(np.zeros(320, dtype=np.float32), 16000)
  assert fake_stt.fed_chunks == 1

  orch._set_state(PipelineState.SPEAKING)
  orch._handle_audio_chunk(np.zeros(320, dtype=np.float32), 16000)
  assert fake_stt.fed_chunks == 1


def test_listening_always_feeds_stt() -> None:
  cfg = _base_cfg()
  cfg["vad"] = {"enabled": True, "gate_stt": True}
  gated_vad = GatedVad()
  orch = _with_fake_playback(
    Orchestrator(cfg, stt=FakeStt(), llm=FakeLlm(), tts=FakeTts(), vad=gated_vad)
  )
  fake_stt = orch.stt
  assert isinstance(fake_stt, FakeStt)
  orch._set_state(PipelineState.LISTENING)

  gated_vad._speech = False
  orch._handle_audio_chunk(np.zeros(320, dtype=np.float32), 16000)
  assert fake_stt.fed_chunks == 1


def test_stt_tail_feeds_after_vad_end() -> None:
  cfg = _base_cfg()
  cfg["vad"] = {"enabled": True, "gate_stt": True}
  cfg["puppet"]["stt_tail_ms"] = 500
  gated_vad = GatedVad()
  orch = _with_fake_playback(
    Orchestrator(cfg, stt=FakeStt(), llm=FakeLlm(), tts=FakeTts(), vad=gated_vad)
  )
  fake_stt = orch.stt
  assert isinstance(fake_stt, FakeStt)
  orch._set_state(PipelineState.LISTENING)

  gated_vad._speech = True
  orch._handle_audio_chunk(np.zeros(320, dtype=np.float32), 16000)
  assert fake_stt.fed_chunks == 1

  gated_vad._speech = False
  orch._speech_active = False
  orch._stt_tail_until = time.monotonic() + 0.5
  orch._handle_audio_chunk(np.zeros(320, dtype=np.float32), 16000)
  assert fake_stt.fed_chunks == 2


def test_generation_done_finalizes_stt_and_starts_echo_guard(orchestrator: Orchestrator) -> None:
  orchestrator.conversation.draft_user = "Hi"
  fake_stt = orchestrator.stt
  assert isinstance(fake_stt, FakeStt)

  orchestrator._on_generation_done("Hello there.", epoch=0)

  assert fake_stt.finalize_calls == 1
  assert fake_stt.reset_calls == 2
  assert orchestrator._await_fresh_speech
  assert orchestrator.conversation.draft_user == ""
  assert orchestrator._echo_quiet_until > time.monotonic()


def test_generation_done_blocks_phantom_drafts() -> None:
  cfg = _base_cfg()
  orch = _with_fake_playback(
    Orchestrator(cfg, stt=FakeStt(), llm=FakeLlm(), tts=FakeTts())
  )
  orch._recent_tts_phrases = ["Tu vas aussi, hein ?"]
  orch.conversation.draft_user = "Bonjour"
  orch._on_generation_done("Bonjour ! Tu vas aussi, hein ?", epoch=0)
  orch._echo_unlock_after = 0.0

  orch._on_stt_partial("Hum.")
  assert orch.conversation.draft_user == ""

  orch._await_fresh_speech = False
  orch._echo_quiet_until = time.monotonic() + 1.0
  orch._recent_tts_phrases = ["Tu veux un câlin ou un jeu ?"]
  orch._on_stt_partial("Un jeu")
  assert orch.conversation.draft_user == ""


def test_draft_matching_recent_tts_is_rejected() -> None:
  cfg = _base_cfg()
  orch = _with_fake_playback(
    Orchestrator(cfg, stt=FakeStt(), llm=FakeLlm(), tts=FakeTts())
  )
  orch._set_state(PipelineState.LISTENING)
  orch._await_fresh_speech = True
  orch._recent_tts_phrases = ["Tu veux un câlin ou un jeu ?"]
  orch.conversation.draft_user = "Un jeu"
  assert not orch._can_start_generation()
  assert orch.conversation.draft_user == ""


def test_draft_matching_recent_tts_allowed_after_unlock() -> None:
  cfg = _base_cfg()
  orch = _with_fake_playback(
    Orchestrator(cfg, stt=FakeStt(), llm=FakeLlm(), tts=FakeTts())
  )
  orch._set_state(PipelineState.LISTENING)
  orch._recent_tts_phrases = ["Je vais bien, merci !"]
  orch.conversation.draft_user = "Je vais bien aussi"
  assert orch._can_start_generation()


def test_stt_tail_does_not_finalize_during_generation() -> None:
  cfg = _base_cfg()
  cfg["vad"] = {"enabled": True, "gate_stt": True}
  cfg["puppet"]["stt_tail_ms"] = 500
  gated_vad = GatedVad()
  orch = _with_fake_playback(
    Orchestrator(cfg, stt=FakeStt(), llm=FakeLlm(), tts=FakeTts(), vad=gated_vad)
  )
  fake_stt = orch.stt
  assert isinstance(fake_stt, FakeStt)
  orch._set_state(PipelineState.THINKING)
  orch._stt_tail_until = time.monotonic() - 0.01

  orch._tick_stt_tail()

  assert fake_stt.finalize_calls == 0


def test_vad_disabled_does_not_reset_stt_while_user_speaks() -> None:
  cfg = _base_cfg()
  cfg["vad"] = {"enabled": False, "gate_stt": False}
  cfg["audio"] = {"speech_rms_threshold": 0.01}
  orch = _with_fake_playback(
    Orchestrator(cfg, stt=FakeStt(), llm=FakeLlm(), tts=FakeTts())
  )
  fake_stt = orch.stt
  assert isinstance(fake_stt, FakeStt)
  orch._set_state(PipelineState.LISTENING)
  orch._enter_post_reply_listen()

  loud = np.full(320, 0.05, dtype=np.float32)
  orch._handle_audio_chunk(loud, 16000)
  assert fake_stt.reset_calls == 1


def test_respeaker_interrupt_feeds_stt_while_speaking() -> None:
  cfg = _base_cfg()
  orch = _with_fake_playback(
    Orchestrator(cfg, stt=FakeStt(), llm=FakeLlm(), tts=FakeTts())
  )
  fake_stt = orch.stt
  assert isinstance(fake_stt, FakeStt)
  orch._reply_in_progress = True
  orch._set_state(PipelineState.SPEAKING)
  orch._respeaker_interrupt_active = True
  orch._respeaker_interrupt_started_at = time.monotonic()

  orch._handle_audio_chunk(np.zeros(320, dtype=np.float32), 16000)

  assert fake_stt.fed_chunks == 1


def test_respeaker_interrupt_cancels_and_keeps_partial_assistant_context() -> None:
  cfg = _base_cfg()
  orch = _with_fake_playback(
    Orchestrator(cfg, stt=FakeStt(), llm=FakeLlm(), tts=FakeTts())
  )
  orch._reply_in_progress = True
  orch._respeaker_interrupt_active = True
  orch._current_reply_text = "Bonjour, je peux "
  orch._set_state(PipelineState.SPEAKING)

  orch._on_stt_partial("attends")

  assert orch.state == PipelineState.LISTENING
  assert not orch._reply_in_progress
  assert orch.conversation.messages
  assert orch.conversation.messages[-1].role == "assistant"
  assert orch.conversation.messages[-1].content == "Bonjour, je peux"
  assert orch.conversation.draft_user == "attends"


def test_respeaker_probe_pauses_and_resumes_tts_on_noise() -> None:
  cfg = _base_cfg()
  fake_playback = FakePlayback(busy=True)
  orch = Orchestrator(cfg, stt=FakeStt(), llm=FakeLlm(), tts=FakeTts())
  orch._tts_pipeline = fake_playback  # type: ignore[assignment]
  orch._worker._phrase_playback = fake_playback
  orch._reply_in_progress = True
  orch._set_state(PipelineState.SPEAKING)

  orch._pause_reply_for_interrupt_probe()
  assert fake_playback.pause_calls == 1
  assert orch._respeaker_interrupt_active

  orch._resume_reply_after_noise_probe()
  assert fake_playback.resume_calls == 1
  assert not orch._respeaker_interrupt_active


def test_phrase_begin_does_not_resume_playback_during_interrupt_probe() -> None:
  cfg = _base_cfg()
  orch = _with_fake_playback(
    Orchestrator(cfg, stt=FakeStt(), llm=FakeLlm(), tts=FakeTts())
  )
  probe = PlaybackProbe()
  orch._playback = probe  # type: ignore[assignment]
  orch._respeaker_interrupt_active = True

  orch._on_tts_phrase_begin("hello")

  assert probe.abort_calls == 1
  assert probe.resume_calls == 0


class ReadyPromptSpy:
  def __init__(self) -> None:
    self.submitted: list[str] = []

  def submit(self, text: str) -> None:
    self.submitted.append(text)

  def wait_done(self, timeout: float | None = None) -> None:
    del timeout

  def is_busy(self) -> bool:
    return False

  def stop(self) -> None:
    pass


def test_speak_ready_prompt_uses_language_profile() -> None:
  cfg = _base_cfg()
  cfg["language"] = {
    "active": "fr",
    "profiles": {"fr": {"ready_listen_prompt": "Je suis prêt !"}},
  }
  orch = Orchestrator(cfg, stt=FakeStt(), llm=FakeLlm(), tts=FakeTts(), vad=PassthroughVad())
  spy = ReadyPromptSpy()
  orch._tts_pipeline = spy  # type: ignore[assignment]
  orch._speak_ready_prompt()
  assert spy.submitted == ["Je suis prêt !"]
  assert orch._await_fresh_speech


def test_speak_ready_prompt_falls_back_when_field_missing() -> None:
  cfg = _base_cfg()
  cfg["language"] = {"active": "en", "profiles": {"en": {}}}
  orch = Orchestrator(cfg, stt=FakeStt(), llm=FakeLlm(), tts=FakeTts(), vad=PassthroughVad())
  spy = ReadyPromptSpy()
  orch._tts_pipeline = spy  # type: ignore[assignment]
  orch._speak_ready_prompt()
  assert spy.submitted == ["Hello! I'm Kace, and I'm ready to listen!"]

