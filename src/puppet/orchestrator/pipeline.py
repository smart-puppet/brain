from __future__ import annotations

import logging
import re
import time
from typing import Any

import numpy as np

from puppet.core.audio.capture import (
  AudioCapture,
  AudioPlayback,
)
from puppet.core.audio.pcm import prepend_lead_in_silence, rms_energy
from puppet.core.audio.vad import VoiceActivityDetector, create_vad
from puppet.core.audio.respeaker import RespeakerDoaMonitor, maybe_reset_respeaker_on_start
from puppet.core.config import get_ready_listen_prompt
from puppet.core.events import EventBus
from puppet.core.types import Conversation, PipelineState, TranscriptSegment
from puppet.llm.base import LlmBackend
from puppet.llm import create_llm
from puppet.orchestrator.generation import GenerationWorker
from puppet.orchestrator.tts_pipeline import PhraseTtsPipeline
from puppet.llm.perf import format_llama_perf, format_llama_perf_cli
from puppet.orchestrator.latency import TurnLatencyTracker, format_turn_latency_line
from puppet.orchestrator.trace import PipelineTracer
from puppet.stt.base import SttBackend
from puppet.stt import create_stt
from puppet.tts.base import TtsBackend
from puppet.tts import create_tts
from puppet.tts.types import MouthEvent, TtsChunk
from puppet.tts.alignments import fallback_fixed_flip_holds_ms, resolve_mouth_mode, shift_timeline
from puppet.hardware.mouth import MouthController, create_mouth

logger = logging.getLogger(__name__)


class Orchestrator:
  """Low-latency streaming pipeline: STT partials → gap-triggered LLM → phrase TTS."""

  def __init__(
    self,
    config: dict[str, Any],
    bus: EventBus | None = None,
    *,
    stt: SttBackend | None = None,
    llm: LlmBackend | None = None,
    tts: TtsBackend | None = None,
    vad: VoiceActivityDetector | None = None,
  ) -> None:
    self.config = config
    self.bus = bus or EventBus()
    self.state = PipelineState.IDLE
    self.conversation = Conversation()

    audio_cfg = config.get("audio", {})
    puppet_cfg = config.get("puppet", {})
    vad_cfg = config.get("vad", {})

    self.tts = tts or create_tts(config)
    # Load LLM before STT: parakeet on CUDA leaves too little VRAM for Ternary-Bonsai.
    self.llm = llm or create_llm(config)
    self.stt = stt or create_stt(config)
    self._vad = vad or create_vad(config)

    self._stt_rate = int(audio_cfg.get("sample_rate", 16000))
    # ReSpeaker-first default: keep continuous decode and avoid false interruptions.
    self._barge_in = bool(puppet_cfg.get("barge_in_enabled", False))
    self._gate_stt = bool(vad_cfg.get("gate_stt", True))
    self._vad_enabled = bool(vad_cfg.get("enabled", True))
    self._speech_active = False

    self._phrase_delimiters = puppet_cfg.get("phrase_delimiters", ".?!\n,")
    self._min_phrase_chars = int(puppet_cfg.get("min_phrase_chars", 8))
    self._min_first_phrase_chars = int(puppet_cfg.get("min_first_phrase_chars", puppet_cfg.get("min_phrase_chars", 8)))
    self._first_phrase_max_wait_ms = int(puppet_cfg.get("first_phrase_max_wait_ms", 0))
    self._min_user_chars = int(puppet_cfg.get("min_user_chars", 3))
    self._stt_gap_s = int(puppet_cfg.get("stt_gap_ms", 400)) / 1000.0
    self._stt_tail_s = int(puppet_cfg.get("stt_tail_ms", puppet_cfg.get("stt_gap_ms", 500))) / 1000.0
    self._stt_tail_until = 0.0
    self._pending_stt_eou = False
    self._restart_on_partial = bool(puppet_cfg.get("restart_on_partial", True))
    self._barge_in_cooldown_s = int(puppet_cfg.get("barge_in_cooldown_ms", 2500)) / 1000.0
    self._barge_in_grace_s = int(puppet_cfg.get("barge_in_grace_ms", 1200)) / 1000.0
    self._barge_in_clean_s = int(puppet_cfg.get("barge_in_clean_ms", 500)) / 1000.0
    self._barge_in_clean_rms = float(puppet_cfg.get("barge_in_clean_rms", 0.022))
    self._mic_speech_rms = float(audio_cfg.get("speech_rms_threshold", puppet_cfg.get("barge_in_clean_rms", 0.015)))
    self._last_mic_rms = 0.0
    self._barge_clean_since = 0.0
    self._echo_quiet_s = int(puppet_cfg.get("echo_quiet_ms", 2000)) / 1000.0
    self._post_reply_echo_s = int(
      puppet_cfg.get("post_reply_echo_ms", min(int(puppet_cfg.get("echo_quiet_ms", 2000)), 800))
    ) / 1000.0
    self._tts_echo_word_overlap = float(puppet_cfg.get("tts_echo_word_overlap", 0.45))
    self._tts_echo_trigram_overlap = float(puppet_cfg.get("tts_echo_trigram_overlap", 0.35))
    self._barge_in_cooldown_until = 0.0
    self._speaking_since = 0.0
    self._playback_started_at = 0.0
    self._await_fresh_speech = False
    self._echo_unlock_after = 0.0
    self._echo_quiet_until = 0.0
    self._await_fresh_since = 0.0
    self._reply_in_progress = False
    self._respeaker_interrupt_enabled = bool(
      audio_cfg.get("respeaker", {}).get("pause_tts_on_speech", True)
    )
    self._respeaker_interrupt_timeout_s = (
      max(100, int(audio_cfg.get("respeaker", {}).get("interrupt_timeout_ms", 900))) / 1000.0
    )
    self._respeaker_interrupt_min_chars = int(puppet_cfg.get("interrupt_min_chars", 2))
    self._respeaker_interrupt_active = False
    self._respeaker_interrupt_heard_text = False
    self._respeaker_interrupt_started_at = 0.0
    self._current_reply_text = ""

    stt_cfg = config.get("stt", {})
    self._stt_suspend_during_llm = bool(stt_cfg.get("suspend_during_llm", True))
    self._stt_suspended = False
    self._tts_playback_active = False

    mouth_cfg = puppet_cfg.get("mouth", {})
    audio_cfg = config.get("audio", {})
    delay_ms = mouth_cfg.get("playback_delay_ms")
    if delay_ms is None:
      device_frames = int(audio_cfg.get("output_frames_per_buffer", 4096))
      write_frames = int(audio_cfg.get("output_write_chunk_frames", 1024))
      self._mouth_playback_delay_samples = device_frames + write_frames
    else:
      self._mouth_playback_delay_samples = max(
        0,
        int(self.tts.sample_rate() * int(delay_ms) / 1000),
      )
    mouth_enabled = bool(mouth_cfg.get("enabled", False))
    self._mouth_mode = resolve_mouth_mode(config) if mouth_enabled else "word"
    if mouth_enabled and self._mouth_mode == "word":
      delay_ms_effective = int(
        round(self._mouth_playback_delay_samples * 1000 / max(1, self.tts.sample_rate()))
      )
      logger.info(
        "Mouth playback delay: %dms (%d samples) for %s sync",
        delay_ms_effective,
        self._mouth_playback_delay_samples,
        self._mouth_mode,
      )

    self._capture: AudioCapture | None = None
    self._playback: AudioPlayback | None = None
    self._playback_warmed = False
    self._last_stt_at = 0.0
    self._latency = TurnLatencyTracker()
    self._trace = PipelineTracer(self._latency)
    self._tts_logged_current = False
    self._mouth_reply_samples = 0
    self._mouth_sync_active = False
    self._recent_tts_phrases: list[str] = []
    self._spoken_reply_corpus = ""

    self._mouth: MouthController = create_mouth(
      config,
      sample_rate=self.tts.sample_rate(),
    )
    self._respeaker_doa = RespeakerDoaMonitor(config)
    if self._respeaker_doa.enabled:
      logger.info("ReSpeaker DoA debug enabled (puppet.respeaker logger)")
    self._mouth_fallback_flip_ms = int(
      mouth_cfg.get("fallback_flip_ms", mouth_cfg.get("stupid_flip_ms", 200))
    )

    self._tts_pipeline = PhraseTtsPipeline(
      self.tts,
      play_chunk=self._play_tts_chunk,
      on_phrase_begin=self._on_tts_phrase_begin,
      on_phrase_end=self._on_tts_phrase_end,
    )

    self._worker = GenerationWorker(
      self.llm,
      phrase_delimiters=self._phrase_delimiters,
      min_phrase_chars=self._min_phrase_chars,
      min_first_phrase_chars=self._min_first_phrase_chars,
      first_phrase_max_wait_ms=self._first_phrase_max_wait_ms,
      phrase_playback=self._tts_pipeline,
    )

  @staticmethod
  def _normalize_echo_text(text: str) -> str:
    return re.sub(r"[\W_]+", "", text.lower(), flags=re.UNICODE)

  @staticmethod
  def _echo_words(text: str) -> list[str]:
    return [w for w in re.findall(r"\w+", text.lower()) if len(w) > 1]

  @staticmethod
  def _words_fuzzy_match(a: str, b: str) -> bool:
    if a == b:
      return True
    short, long = (a, b) if len(a) <= len(b) else (b, a)
    return len(short) >= 4 and long.startswith(short)

  def _word_overlap_ratio(self, text: str, phrase: str) -> float:
    needle = self._echo_words(text)
    if not needle:
      return 1.0
    hay = self._echo_words(phrase)
    hits = sum(
      1 for w in needle if any(self._words_fuzzy_match(w, h) for h in hay)
    )
    return hits / len(needle)

  @staticmethod
  def _trigram_overlap(text: str, phrase: str) -> float:
    a = re.sub(r"[\W_]+", "", text.lower(), flags=re.UNICODE)
    b = re.sub(r"[\W_]+", "", phrase.lower(), flags=re.UNICODE)
    if len(a) < 3:
      return 0.0
    grams_a = {a[i : i + 3] for i in range(len(a) - 2)}
    grams_b = {b[i : i + 3] for i in range(len(b) - 2)}
    if not grams_a:
      return 0.0
    return len(grams_a & grams_b) / len(grams_a)

  def _matches_tts_phrase(self, text: str, phrase: str) -> bool:
    needle = self._normalize_echo_text(text)
    hay = self._normalize_echo_text(phrase)
    if not needle:
      return True
    if needle in hay or hay in needle:
      return True
    if self._word_overlap_ratio(text, phrase) >= self._tts_echo_word_overlap:
      return True
    if self._trigram_overlap(text, phrase) >= self._tts_echo_trigram_overlap:
      return True
    return False

  def _text_looks_like_tts_echo(self, text: str) -> bool:
    if self._spoken_reply_corpus and self._matches_tts_phrase(
      text, self._spoken_reply_corpus
    ):
      return True
    for phrase in self._recent_tts_phrases:
      if self._matches_tts_phrase(text, phrase):
        return True
    return False

  def _draft_looks_like_tts_echo(self, draft: str) -> bool:
    if not self._echo_suppresses_stt_draft():
      return False
    return self._text_looks_like_tts_echo(draft)

  def _reply_still_active(self) -> bool:
    return (
      self._reply_in_progress
      or self._worker.active
      or self._tts_pipeline.is_busy()
    )

  def _user_speaking_now(self) -> bool:
    if self._speech_active:
      return True
    if not self._vad_enabled:
      return self._mic_has_speech_energy()
    return self._vad.is_speech

  def _mic_has_speech_energy(self) -> bool:
    return self._last_mic_rms >= self._mic_speech_rms

  def _clear_fresh_speech_gate(self) -> None:
    """Open the mic after post-reply echo guard without clipping an active utterance."""
    self._await_fresh_speech = False
    self.conversation.draft_user = ""
    if not self._user_speaking_now():
      self.stt.reset()
    self._latency.clear_speech_window()
    if time.monotonic() >= self._echo_quiet_until:
      self._recent_tts_phrases.clear()
      self._spoken_reply_corpus = ""

  def _enter_post_reply_listen(self) -> None:
    """Block speaker bleed from being drafted until the user speaks again."""
    self._mark_echo_risk(duration_s=self._post_reply_echo_s)
    self._await_fresh_speech = True
    self._await_fresh_since = time.monotonic()
    self._echo_unlock_after = time.monotonic() + 0.4
    self.conversation.draft_user = ""
    self._last_stt_at = 0.0
    self._stt_tail_until = 0.0
    self.stt.reset()
    self._latency.clear_speech_window()

  def _speak_ready_prompt(self) -> None:
    """Say the localized ready-to-listen line once at streaming startup."""
    text = get_ready_listen_prompt(self.config)
    if not text:
      return
    logger.info("Speaking ready prompt")
    self._reply_in_progress = True
    self._recent_tts_phrases.append(text)
    self._spoken_reply_corpus = text
    self._tts_pipeline.submit(text)
    self._tts_pipeline.wait_done()
    self._reply_in_progress = False
    self._stop_tts_playback()
    self._set_state(PipelineState.LISTENING)
    self._enter_post_reply_listen()

  def _cancel_reply(self) -> None:
    """Hard-stop LLM + TTS playback and return to open listening."""
    if not self._reply_still_active():
      return
    logger.info("Barge-in — cancelling playback")
    self._reply_in_progress = False
    self._abort_playback()
    self._worker.stop()
    self._tts_pipeline.stop()
    self._stop_tts_playback()
    self._playback_started_at = 0.0
    self._barge_clean_since = 0.0
    self._set_state(PipelineState.LISTENING)
    self._enter_post_reply_listen()
    self._vad.reset()
    self._latency.reset()
    self._trace.reset()
    self.bus.emit("playback_cancelled")

  def _pause_reply_for_interrupt_probe(self) -> None:
    if self._respeaker_interrupt_active or not self._reply_still_active():
      return
    self._respeaker_interrupt_active = True
    self._respeaker_interrupt_heard_text = False
    self._respeaker_interrupt_started_at = time.monotonic()
    self._abort_playback()
    self._tts_pipeline.pause()
    self._tts_playback_active = False
    if self.state == PipelineState.SPEAKING:
      self._set_state(PipelineState.THINKING)
    logger.info("Speech detected during reply — pausing TTS pending STT confirmation")

  def _resume_reply_after_noise_probe(self) -> None:
    if not self._respeaker_interrupt_active:
      return
    self._respeaker_interrupt_active = False
    self._respeaker_interrupt_heard_text = False
    self._respeaker_interrupt_started_at = 0.0
    self._tts_pipeline.resume()
    if self._playback is not None:
      self._playback.resume()
    logger.info("Interrupt probe classified as noise — resuming reply playback")

  def _cancel_reply_for_user_interrupt(self) -> None:
    if not self._reply_still_active():
      self._resume_reply_after_noise_probe()
      return
    logger.info("User interruption confirmed — cancelling current generation")
    prefix = self._current_reply_text.strip()
    if prefix:
      self.conversation.add_assistant(prefix)
    self._reply_in_progress = False
    self._worker.stop()
    self._tts_pipeline.stop()
    self._stop_tts_playback()
    self._resume_stt_after_llm()
    self._playback_started_at = 0.0
    self._barge_clean_since = 0.0
    self._respeaker_interrupt_active = False
    self._respeaker_interrupt_heard_text = False
    self._respeaker_interrupt_started_at = 0.0
    self._set_state(PipelineState.LISTENING)
    self.bus.emit("playback_cancelled")

  def _set_state(self, state: PipelineState) -> None:
    prev = self.state
    self.state = state
    if state == PipelineState.LISTENING and prev != PipelineState.LISTENING:
      self._mouth.clear_sync()
      self._mouth.close_for_listen()
    self.bus.emit("state_changed", state=state)

  def _open_capture(self) -> None:
    audio_cfg = self.config.get("audio", {})
    if self._capture is None:
      device_index = audio_cfg.get("input_device")
      maybe_reset_respeaker_on_start(self.config, device_index=device_index)
      self._capture = AudioCapture(
        sample_rate=self._stt_rate,
        channels=int(audio_cfg.get("channels", 1)),
        chunk_ms=int(audio_cfg.get("chunk_ms", 20)),
        device_index=device_index,
      )

  def _open_playback(self) -> None:
    audio_cfg = self.config.get("audio", {})
    if self._playback is None:
      self._playback = AudioPlayback(
        sample_rate=self.tts.sample_rate(),
        channels=int(audio_cfg.get("channels", 1)),
        device_index=audio_cfg.get("output_device"),
        frames_per_buffer=int(audio_cfg.get("output_frames_per_buffer", 4096)),
        write_chunk_frames=int(audio_cfg.get("output_write_chunk_frames", 1024)),
        on_samples_committed=self._pump_mouth_timeline,
      )

  def _pump_mouth_timeline(self) -> None:
    if self._playback is not None and self._mouth_mode == "word":
      self._mouth.pump_timeline(self._playback)

  def _open_audio(self) -> None:
    self._open_capture()
    self._open_playback()

  def _warmup_playback(self) -> None:
    if self._playback is None or self._playback_warmed:
      return
    audio_cfg = self.config.get("audio", {})
    silence_ms = int(audio_cfg.get("playback_warmup_ms", 150))
    self._playback.warmup(silence_ms=silence_ms)
    self._playback_warmed = True

  def _abort_playback(self) -> None:
    if self._playback is not None:
      self._playback.abort()

  def _stop_tts_playback(self) -> None:
    self._tts_playback_active = False
    self._mouth_sync_active = False
    self._mouth.clear_sync()

  def _process_mic_chunk(self, mic: np.ndarray, sample_rate: int) -> TranscriptSegment | None:
    return self.stt.feed(mic, sample_rate)

  def _generation_active(self) -> bool:
    return self._worker.active or self.state in (PipelineState.THINKING, PipelineState.SPEAKING)

  def _barge_in_grace_active(self) -> bool:
    """Reply-level grace from first TTS playback (covers inter-phrase gaps)."""
    if not self._reply_still_active() or self._playback_started_at <= 0:
      return False
    return time.monotonic() - self._playback_started_at < self._barge_in_grace_s

  def _barge_in_allowed(self) -> bool:
    if time.monotonic() < self._barge_in_cooldown_until:
      return False
    if self._barge_in_grace_active():
      return False
    return True

  def _looks_like_barge_in(self, mic: np.ndarray) -> bool:
    """Detect user speech over playback without running STT."""
    threshold = self._barge_in_clean_rms
    mic_rms = rms_energy(mic)

    if self._tts_playback_active:
      return mic_rms >= threshold

    if self._reply_still_active():
      if self._user_speaking_now() and mic_rms >= threshold:
        return True
      return False

    if self.state == PipelineState.THINKING:
      vad_speech = self._vad_enabled and (self._speech_active or self._vad.is_speech)
      if vad_speech and mic_rms >= threshold * 0.5:
        return True
      return mic_rms >= threshold

    return mic_rms >= threshold

  def _tick_barge_in_cancel(self, mic: np.ndarray) -> bool:
    """Cancel playback when sustained user speech is seen on the mic."""
    if not self._barge_in or not self._reply_still_active():
      self._barge_clean_since = 0.0
      return False
    if self.state not in (PipelineState.SPEAKING, PipelineState.THINKING):
      self._barge_clean_since = 0.0
      return False
    if not self._barge_in_allowed():
      self._barge_clean_since = 0.0
      return False

    if not self._looks_like_barge_in(mic):
      self._barge_clean_since = 0.0
      return False

    now = time.monotonic()
    if self._barge_clean_since <= 0:
      self._barge_clean_since = now
      return False
    if now - self._barge_clean_since < self._barge_in_clean_s:
      return False

    self._barge_clean_since = 0.0
    self._barge_in_cooldown_until = now + self._barge_in_cooldown_s
    self._cancel_reply()
    return True

  def _mark_echo_risk(self, duration_s: float | None = None) -> None:
    quiet_s = self._echo_quiet_s if duration_s is None else duration_s
    until = time.monotonic() + quiet_s
    if until > self._echo_quiet_until:
      self._echo_quiet_until = until

  def _echo_suppresses_stt_draft(self) -> bool:
    if self.state != PipelineState.LISTENING:
      return False
    if self._await_fresh_speech:
      return True
    if self._tts_playback_active:
      return True
    return time.monotonic() < self._echo_quiet_until

  def _unlock_fresh_speech(self) -> None:
    """Resume STT after post-reply echo guard once VAD sees new speech."""
    if not self._await_fresh_speech or self.state != PipelineState.LISTENING:
      return
    if self._await_fresh_since > 0 and time.monotonic() - self._await_fresh_since > 15.0:
      logger.warning("Fresh-speech wait timed out — resuming capture")
      self._await_fresh_speech = False
      return
    if time.monotonic() < self._echo_unlock_after:
      return
    if self._tts_playback_active or time.monotonic() < self._echo_quiet_until:
      return
    if not self._vad_enabled:
      self._clear_fresh_speech_gate()
      return
    if self._user_speaking_now():
      self._clear_fresh_speech_gate()
      logger.info("Listening again — speak your request")

  def _handle_vad_events(self, mic: np.ndarray) -> bool:
    events = self._vad.feed(mic)
    for event in events:
      if event.kind == "start":
        self._speech_active = True
        if self.state == PipelineState.LISTENING:
          if self._await_fresh_speech:
            if self._tts_playback_active or time.monotonic() < self._echo_quiet_until:
              continue
            self._clear_fresh_speech_gate()
          # New user turn only — do not reset during THINKING/SPEAKING (echo triggers VAD).
          self._latency.reset()
          self._trace.reset()
        self.bus.emit("vad_start")
      elif event.kind == "end":
        self._speech_active = False
        self._stt_tail_until = time.monotonic() + self._stt_tail_s
        self._latency.mark_vad_end()
        self.bus.emit("vad_end")

    return False

  def _should_feed_stt(self) -> bool:
    if self._respeaker_interrupt_active:
      return True
    # Never transcribe during playback — open mic + speaker makes STT hear the bot.
    if self.state == PipelineState.SPEAKING:
      return False
    if self.state == PipelineState.LISTENING:
      return True
    if self.state == PipelineState.THINKING:
      if not self._gate_stt:
        return True
      return self._speech_active or self._vad.is_speech
    if not self._gate_stt:
      return True
    if self._stt_tail_until and time.monotonic() < self._stt_tail_until:
      return True
    return self._speech_active or self._vad.is_speech

  def _on_stt_partial(self, text: str) -> None:
    if self._echo_suppresses_stt_draft():
      return

    if text.strip():
      self._latency.mark_stt_partial()
    self.conversation.append_draft(text)
    self._last_stt_at = time.monotonic()
    self._trace.stt_partial(text, self.conversation.draft_user)
    self.bus.emit("transcript_partial", text=text, draft=self.conversation.draft_user)

    if self._respeaker_interrupt_active and text.strip():
      self._respeaker_interrupt_heard_text = True
      if len(self.conversation.draft_user.strip()) >= self._respeaker_interrupt_min_chars:
        self._cancel_reply_for_user_interrupt()
      return

    if self._restart_on_partial and self._generation_active():
      if self.state == PipelineState.SPEAKING:
        return
      logger.info("STT still updating draft — restarting LLM")
      self._restart_generation()

  def _user_speaking(self) -> bool:
    if self._speech_active:
      return True
    if not self._vad_enabled:
      return self._mic_has_speech_energy()
    return self._vad.is_speech

  def _stt_tail_blocking(self) -> bool:
    """True while post-VAD tail audio is still being fed to STT."""
    return self._stt_tail_until > 0 and time.monotonic() < self._stt_tail_until

  def _can_start_generation(self) -> bool:
    draft = self.conversation.draft_user.strip()
    if len(draft) < self._min_user_chars:
      return False
    if self._draft_looks_like_tts_echo(draft):
      logger.info("Ignoring STT draft that matches recent TTS: %r", draft)
      self.conversation.draft_user = ""
      self._last_stt_at = 0.0
      self.stt.reset()
      return False
    if self._user_speaking():
      return False
    return True

  def _on_stt_eou(self) -> None:
    if self._await_fresh_speech or self._echo_suppresses_stt_draft():
      return
    if self._stt_tail_blocking():
      self._pending_stt_eou = True
      return
    if self._can_start_generation():
      self._start_generation()

  def _on_transcript(self, segment: TranscriptSegment) -> None:
    if segment.text:
      self._on_stt_partial(segment.text)
    if segment.end_of_utterance:
      self._on_stt_eou()

  def _tick_stt_tail(self) -> None:
    if self._stt_tail_until <= 0:
      return
    if time.monotonic() < self._stt_tail_until:
      return
    self._stt_tail_until = 0.0
    # Do not tear down the STT stream mid-reply; barge-in still needs it.
    if self._generation_active():
      return
    final = self.stt.finalize()
    if final and final.text:
      self._on_transcript(final)
    elif final and final.end_of_utterance:
      self._on_stt_eou()
    if self._pending_stt_eou:
      self._pending_stt_eou = False
      self._on_stt_eou()
    else:
      self._tick_gap()

  def _tick_gap(self) -> None:
    if self.state != PipelineState.LISTENING:
      return
    if self._await_fresh_speech or self._echo_suppresses_stt_draft():
      return
    if self._stt_tail_blocking():
      return
    if self._generation_active():
      return
    if not self._can_start_generation():
      return
    if self._last_stt_at <= 0:
      return
    if time.monotonic() - self._last_stt_at < self._stt_gap_s:
      return
    self._start_generation()

  def _suspend_stt_for_llm(self) -> None:
    if not self._stt_suspend_during_llm or self._stt_suspended:
      return
    suspend = getattr(self.stt, "suspend", None)
    if suspend is None:
      return
    suspend()
    self._stt_suspended = True

  def _resume_stt_after_llm(self) -> None:
    if not self._stt_suspended:
      return
    resume = getattr(self.stt, "resume", None)
    if resume is not None:
      resume()
    self._stt_suspended = False

  def _start_generation(self) -> None:
    if self._await_fresh_speech:
      return
    if not self._can_start_generation():
      return
    # Trailing-audio feed after VAD end is only needed until generation starts.
    self._stt_tail_until = 0.0
    self._recent_tts_phrases.clear()
    self._spoken_reply_corpus = ""
    self._current_reply_text = ""
    self._reply_in_progress = True
    self._suspend_stt_for_llm()
    prompt = self.conversation.draft_user.strip()
    self._trace.llm_prompt(prompt)
    self._latency.mark_generation_start()
    self._set_state(PipelineState.THINKING)
    self._speaking_since = 0.0
    self._playback_started_at = 0.0
    epoch = self._worker.start(
      self.conversation,
      on_token=self._on_llm_token,
      on_done=self._on_generation_done,
    )
    self.bus.emit("generation_started", draft=self.conversation.draft_user, epoch=epoch)

  def _log_turn_summary(self) -> None:
    report = self._latency.report()
    if report is None:
      return
    perf = getattr(self.llm, "last_perf", None)
    llm_line = format_llama_perf_cli(perf) if perf is not None else None
    llm_wall_ms = None
    if perf is not None and (perf.prompt_ms or perf.generation_ms):
      llm_wall_ms = perf.prompt_ms + perf.generation_ms
    logger.info(
      "%s",
      format_turn_latency_line(report, llm_perf=llm_line, llm_wall_ms=llm_wall_ms),
    )
    if perf is not None:
      logger.debug("perf: %s", format_llama_perf(perf))

  def _on_llm_token(self, token: str) -> None:
    self._current_reply_text += token
    self._latency.mark_llm_token()
    self._resume_stt_after_llm()
    self._trace.llm_generating()
    self._set_state(PipelineState.SPEAKING)

  def _restart_generation(self) -> None:
    if not self._can_start_generation():
      return
    self._start_generation()

  def _on_generation_done(self, reply: str, epoch: int) -> None:
    if epoch != self._worker.epoch:
      return

    self._latency.mark_turn_end()
    self._log_turn_summary()
    self._resume_stt_after_llm()

    user_text = self.conversation.commit_draft()
    if reply:
      self.conversation.add_assistant(reply)
      self.bus.emit("assistant_reply", text=reply)

    if user_text:
      logger.info("User: %s", user_text)
    if reply:
      logger.info("Assistant: %s", reply)

    self._set_state(PipelineState.LISTENING)
    self._reply_in_progress = False
    self._current_reply_text = ""
    self._stop_tts_playback()
    self._end_stt_turn()
    self._enter_post_reply_listen()
    self.bus.emit("generation_done", epoch=epoch)

  @staticmethod
  def _is_stt_noise_tail(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
      return True
    return len(stripped) <= 2 and all(c in ".,?!…\"'«»" for c in stripped)

  def _end_stt_turn(self) -> None:
    """Flush and close the streaming STT session after a completed reply."""
    try:
      final = self.stt.finalize()
      if final and final.text.strip() and not self._is_stt_noise_tail(final.text):
        logger.debug("STT tail after turn (ignored): %r", final.text)
    except Exception:
      logger.exception("STT finalize failed after generation; forcing reset")
      self.stt.reset()
      return
    self.stt.reset()

  def _on_tts_phrase_begin(self, text: str) -> None:
    self._recent_tts_phrases.append(text)
    self._spoken_reply_corpus = f"{self._spoken_reply_corpus} {text}".strip()
    self._open_playback()
    self._warmup_playback()
    self._latency.mark_tts_phrase(text)
    if self._playback is not None:
      if self._respeaker_interrupt_active:
        # Keep hard-muted while probing whether detected speech is real user input.
        self._playback.abort()
      else:
        self._playback.resume()
      if not self._mouth_sync_active:
        self._playback.reset_sample_clock()
        self._mouth.clear_sync()
        self._mouth_reply_samples = 0
        self._mouth_sync_active = True
        self._mouth.on_reply_sync_start()
    self._tts_logged_current = False

  def _tts_lead_in_ms(self, phrase: str) -> int:
    tts_cfg = self.config.get("tts", {})
    lead_in_ms = int(tts_cfg.get("lead_in_ms", 80))
    max_words = int(tts_cfg.get("short_phrase_max_words", 2))
    short_ms = int(tts_cfg.get("short_phrase_lead_in_ms", 160))
    if len(phrase.split()) <= max_words:
      lead_in_ms = max(lead_in_ms, short_ms)
    return lead_in_ms

  def _play_tts_chunk(self, phrase: str, chunk: TtsChunk) -> None:
    if self.state == PipelineState.LISTENING and not self._reply_in_progress:
      return
    assert self._playback is not None
    first_chunk = not self._tts_logged_current
    lead_in_ms = 0
    samples = chunk.samples
    lead_in_samples = 0
    if first_chunk:
      lead_in_ms = self._tts_lead_in_ms(phrase)
      lead_in_samples = int(self.tts.sample_rate() * lead_in_ms / 1000)
      samples = prepend_lead_in_silence(
        samples,
        self.tts.sample_rate(),
        lead_in_ms,
      )
    if chunk.mouth_timeline and self._mouth_mode == "word" and int(samples.size) > 0:
      timeline = shift_timeline(
        chunk.mouth_timeline,
        self._mouth_reply_samples + lead_in_samples,
      )
      self._mouth.append_timeline(
        timeline,
        self._playback,
        playback_delay_samples=self._mouth_playback_delay_samples,
        source=self._mouth_mode,
      )
    elif self.tts.has_mouth_timeline and self._mouth_mode == "word" and int(samples.size) > 0:
      fallback = shift_timeline(
        [
          MouthEvent(0, True),
          MouthEvent(int(samples.size), False),
        ],
        self._mouth_reply_samples + lead_in_samples,
      )
      self._mouth.append_timeline(
        fallback,
        self._playback,
        playback_delay_samples=self._mouth_playback_delay_samples,
        source="whole-chunk",
      )
    elif self._mouth_mode == "fallback" and int(samples.size) > 0:
      holds = chunk.phoneme_hold_ms if self.tts.has_mouth_timeline else None
      hold_source = "phoneme"
      if not holds:
        holds = fallback_fixed_flip_holds_ms(
          int(samples.size),
          self.tts.sample_rate(),
          flip_ms=self._mouth_fallback_flip_ms,
        )
        hold_source = "fixed"
      self._mouth.append_fallback_durations(holds, source=hold_source)
    if first_chunk:
      self._tts_playback_active = True
      now = time.monotonic()
      self._speaking_since = now
      if self._playback_started_at <= 0:
        self._playback_started_at = now
      self._mark_echo_risk()
      if self.state != PipelineState.SPEAKING:
        self._set_state(PipelineState.SPEAKING)
    pcm = np.clip(samples * 32767.0, -32768, 32767).astype(np.int16).tobytes()
    self._mouth.on_chunk_play(int(samples.size), self.tts.sample_rate())
    self._playback.play_int16(pcm)
    self._mouth_reply_samples += int(samples.size)
    if first_chunk:
      self._latency.mark_speaker(lead_in_ms=lead_in_ms)
      self._trace.tts_playing(phrase)
      self._tts_logged_current = True

  def _on_tts_phrase_end(self, _phrase: str) -> None:
    self._mouth.on_phrase_end()
    if self._playback is not None:
      self._playback.flush()
      if self._mouth_mode == "word":
        self._mouth.pump_timeline(self._playback)
    if self._reply_still_active():
      return
    if not self._tts_pipeline.is_busy():
      self._stop_tts_playback()
      self._mark_echo_risk()

  def _should_process_stt(self) -> bool:
    if self._respeaker_interrupt_active:
      return True
    if self.state in (PipelineState.LISTENING, PipelineState.THINKING):
      return self._should_feed_stt()
    return False

  def _handle_audio_chunk(self, mic: np.ndarray, sample_rate: int) -> None:
    self._last_mic_rms = rms_energy(mic)
    self._handle_vad_events(mic)
    self._respeaker_doa.maybe_log(speech_active=self._user_speaking_now())
    self._unlock_fresh_speech()
    if (
      self._respeaker_interrupt_enabled
      and self._reply_still_active()
      and not self._respeaker_interrupt_active
      and self._user_speaking_now()
    ):
      self._pause_reply_for_interrupt_probe()
    if self._respeaker_interrupt_active:
      if self._reply_still_active() and self._user_speaking_now():
        self._respeaker_interrupt_started_at = time.monotonic()
      elif (
        not self._respeaker_interrupt_heard_text
        and time.monotonic() - self._respeaker_interrupt_started_at >= self._respeaker_interrupt_timeout_s
      ):
        self._resume_reply_after_noise_probe()

    if self._tick_barge_in_cancel(mic):
      return

    if self._should_process_stt():
      segment = self._process_mic_chunk(mic, sample_rate)
      if segment:
        self._on_transcript(segment)

    self._tick_gap()
    self._tick_stt_tail()

  def listen_once(self, duration_s: float = 5.0) -> None:
    self._open_capture()
    assert self._capture is not None
    self._set_state(PipelineState.LISTENING)
    deadline = time.monotonic() + duration_s
    while time.monotonic() < deadline:
      chunk = self._capture.read()
      self._handle_audio_chunk(chunk.samples, chunk.sample_rate)
    final = self.stt.finalize()
    if final:
      self._on_transcript(final)
    self._tick_gap()
    deadline = time.monotonic() + self._stt_gap_s + 0.5
    while self._worker.active and time.monotonic() < deadline:
      time.sleep(0.02)

  def run(self) -> None:
    self._open_capture()
    assert self._capture is not None
    self._set_state(PipelineState.LISTENING)
    logger.info("Puppet listening (streaming mode)")
    self._speak_ready_prompt()
    try:
      while True:
        chunk = self._capture.read()
        self._handle_audio_chunk(chunk.samples, chunk.sample_rate)
    except KeyboardInterrupt:
      logger.info("Shutting down")
    finally:
      self.close()

  def close(self) -> None:
    self._worker.stop()
    self._tts_pipeline.stop()
    self.stt.close()
    if hasattr(self.llm, "close"):
      self.llm.close()
    if self._capture:
      self._capture.close()
      self._capture = None
    if self._playback:
      self._playback.close()
      self._playback = None
    self._mouth.close()
    self._respeaker_doa.close()
