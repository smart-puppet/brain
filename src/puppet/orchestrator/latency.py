from __future__ import annotations

import time
from dataclasses import dataclass


def _ms(start: float, end: float) -> float:
  return max(0.0, (end - start) * 1000.0)


@dataclass(frozen=True)
class TurnLatencyReport:
  """Wall-clock pipeline phases for one completed turn."""

  gap_ms: float
  ttft_ms: float
  phrase_ms: float
  play_ms: float
  total_ms: float
  heard_lead_in_ms: float = 0.0

  @property
  def speech_to_speaker_ms(self) -> float:
    """Last STT text → first PCM sent to the speaker (includes TTS lead-in silence)."""
    return self.gap_ms + self.ttft_ms + self.phrase_ms

  @property
  def speech_to_heard_ms(self) -> float:
    """Last STT text → first audible speech (excludes configured TTS lead-in pad)."""
    return self.speech_to_speaker_ms - self.heard_lead_in_ms

  # Back-compat aliases used in tests/docs.
  @property
  def wait_ms(self) -> float:
    return self.gap_ms

  @property
  def buffer_ms(self) -> float:
    return self.phrase_ms

  @property
  def speech_to_audio_ms(self) -> float:
    return self.speech_to_speaker_ms


def _latency_bar(stt_ms: float, llm_ms: float, tts_ms: float, width: int = 21) -> str:
  total = stt_ms + llm_ms + tts_ms
  if total <= 0:
    return "?" * width
  segments = ((stt_ms, "█"), (llm_ms, "▓"), (tts_ms, "░"))
  counts = [int(round(width * ms / total)) if ms > 0 else 0 for ms, _ in segments]
  for i, (ms, _) in enumerate(segments):
    if ms > 0 and counts[i] == 0:
      counts[i] = 1
  while sum(counts) > width:
    counts[counts.index(max(counts))] -= 1
  while sum(counts) < width:
    idx = max(range(3), key=lambda i: segments[i][0])
    counts[idx] += 1
  return "".join(ch * n for (_, ch), n in zip(segments, counts))


def format_turn_latency_line(
  report: TurnLatencyReport,
  *,
  llm_perf: str | None = None,
  llm_wall_ms: float | None = None,
  bar_width: int = 21,
) -> str:
  """One-line latency bar plus optional LLM perf (same style as test_llm)."""
  bar = _latency_bar(
    report.gap_ms,
    report.ttft_ms + report.phrase_ms,
    report.play_ms,
    bar_width,
  )
  heard = report.speech_to_heard_ms
  speaker = report.speech_to_speaker_ms
  line = (
    f"latency heard {heard:.0f}ms (speaker {speaker:.0f}ms) [{bar}] "
    f"gap {report.gap_ms:.0f}ms | ttft {report.ttft_ms:.0f}ms | "
    f"phrase {report.phrase_ms:.0f}ms | play {report.play_ms:.0f}ms | "
    f"total {report.total_ms:.0f}ms"
  )
  if llm_wall_ms is not None:
    line = f"{line} | llm_wall {llm_wall_ms:.0f}ms"
  if llm_perf:
    line = f"{line}  |  {llm_perf}"
  return line


class TurnLatencyTracker:
  """Per-turn monotonic timestamps for pipeline trace timing."""

  def __init__(self) -> None:
    self.reset()

  def reset(self) -> None:
    now = time.monotonic()
    self._turn_start: float | None = now
    self._vad_end: float | None = None
    self._utterance_end: float | None = None
    self._first_stt_after_vad: float | None = None
    self._last_stt_partial_at: float | None = None
    self._generation_start: float | None = None
    self._first_llm_token: float | None = None
    self._first_tts_phrase: float | None = None
    self._first_speaker: float | None = None
    self._heard_lead_in_ms: float = 0.0
    self._turn_end: float | None = None

  def clear_speech_window(self) -> None:
    """Drop VAD/utterance anchors between turns (echo guard, fresh-speech unlock)."""
    self._vad_end = None
    self._utterance_end = None
    self._first_stt_after_vad = None

  def mark_vad_end(self) -> None:
    self._vad_end = time.monotonic()
    self._first_stt_after_vad = None

  def mark_stt_partial(self) -> None:
    """Call when STT emits non-empty text."""
    self._last_stt_partial_at = time.monotonic()

  def in_post_speech_window(self) -> bool:
    return self._vad_end is not None

  def ms_since_vad_end(self) -> float | None:
    if self._vad_end is None:
      return None
    return _ms(self._vad_end, time.monotonic())

  def ms_since_last_stt_partial(self) -> float | None:
    if self._last_stt_partial_at is None:
      return None
    return _ms(self._last_stt_partial_at, time.monotonic())

  def mark_first_stt_partial(self) -> float | None:
    """Ms since VAD silence onset for the first STT partial in this pause."""
    if self._vad_end is None or self._first_stt_after_vad is not None:
      return None
    now = time.monotonic()
    self._first_stt_after_vad = now
    return _ms(self._vad_end, now)

  def mark_generation_start(self) -> None:
    self._generation_start = time.monotonic()
    if self._last_stt_partial_at is not None:
      self._utterance_end = self._last_stt_partial_at
    elif self._vad_end is not None:
      self._utterance_end = self._vad_end

  def mark_llm_token(self) -> None:
    if self._first_llm_token is None:
      self._first_llm_token = time.monotonic()

  def mark_tts_phrase(self, _text: str) -> None:
    if self._first_tts_phrase is None:
      self._first_tts_phrase = time.monotonic()

  def mark_speaker(self, *, lead_in_ms: float = 0) -> None:
    if self._first_speaker is None:
      self._first_speaker = time.monotonic()
      self._heard_lead_in_ms = max(0.0, lead_in_ms)

  def _speech_origin(self) -> float | None:
    return self._utterance_end or self._vad_end or self._turn_start

  def ms_speech_end_to_first_audio(self) -> float | None:
    """Ms from last user words (or VAD end) to first speaker output."""
    origin = self._speech_origin()
    if origin is None:
      return None
    first = self._first_speaker or self._first_tts_phrase
    if first is None:
      return None
    return _ms(origin, first)

  def ms_speech_end_to_first_heard(self) -> float | None:
    total = self.ms_speech_end_to_first_audio()
    if total is None:
      return None
    return max(0.0, total - self._heard_lead_in_ms)

  def mark_turn_end(self) -> None:
    self._turn_end = time.monotonic()

  def report(self) -> TurnLatencyReport | None:
    """Build a phase breakdown after ``mark_turn_end()``."""
    if self._turn_end is None or self._generation_start is None:
      return None
    origin = self._speech_origin()
    if origin is None:
      return None
    first_audio = self._first_speaker or self._first_tts_phrase or self._first_llm_token
    if first_audio is None:
      first_audio = self._turn_end
    first_token = self._first_llm_token or first_audio
    gap_ms = _ms(origin, self._generation_start)
    ttft_ms = _ms(self._generation_start, first_token)
    phrase_ms = _ms(first_token, first_audio)
    play_ms = _ms(first_audio, self._turn_end)
    total_ms = _ms(origin, self._turn_end)
    return TurnLatencyReport(
      gap_ms=gap_ms,
      ttft_ms=ttft_ms,
      phrase_ms=phrase_ms,
      play_ms=play_ms,
      total_ms=total_ms,
      heard_lead_in_ms=self._heard_lead_in_ms,
    )

  def ms_since_generation_start(self) -> float | None:
    if self._generation_start is None:
      return None
    return _ms(self._generation_start, time.monotonic())

  def ms_since_first_llm_token(self) -> float | None:
    if self._first_llm_token is None:
      return None
    return _ms(self._first_llm_token, time.monotonic())
