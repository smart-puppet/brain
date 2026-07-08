from __future__ import annotations

import logging

from puppet.orchestrator.latency import TurnLatencyTracker

logger = logging.getLogger("puppet.trace")

_CLIP = 120


def _clip(text: str, limit: int = _CLIP) -> str:
  s = " ".join(text.split())
  if len(s) <= limit:
    return s
  return s[: limit - 1] + "…"


def _fmt_ms(ms: float | None) -> str:
  if ms is None or ms < 1:
    return ""
  return f" ({ms:.0f}ms)"


class PipelineTracer:
  """Synthetic per-turn pipeline milestones (enable at DEBUG on ``puppet.trace``)."""

  def __init__(self, latency: TurnLatencyTracker) -> None:
    self._latency = latency
    self._llm_generating_logged = False
    self._tts_playing_logged = False

  def reset(self) -> None:
    self._llm_generating_logged = False
    self._tts_playing_logged = False

  def stt_partial(self, fragment: str, draft: str) -> None:
    if not draft.strip():
      return
    if not logger.isEnabledFor(logging.DEBUG):
      return
    since_vad_end = self._latency.mark_first_stt_partial() if fragment.strip() else None
    if since_vad_end is not None:
      logger.debug("stt  %s (%dms since speech end)", _clip(draft), round(since_vad_end))
    else:
      logger.debug("stt  %s", _clip(draft))

  def stt_decoded(self, draft: str) -> None:
    self.stt_partial("", draft)

  def llm_prompt(self, prompt: str) -> None:
    if not logger.isEnabledFor(logging.DEBUG):
      return
    self._llm_generating_logged = False
    logger.debug('llm  → %s', _clip(prompt))

  def llm_generating(self) -> None:
    if self._llm_generating_logged or not logger.isEnabledFor(logging.DEBUG):
      return
    self._llm_generating_logged = True
    logger.debug("llm  generating%s", _fmt_ms(self._latency.ms_since_generation_start()))

  def tts_playing(self, phrase: str) -> None:
    if not phrase.strip() or not logger.isEnabledFor(logging.DEBUG):
      return
    first = not self._tts_playing_logged
    if first:
      self._tts_playing_logged = True
      heard = self._latency.ms_speech_end_to_first_heard()
      suffix = _fmt_ms(heard) if heard is not None else ""
    else:
      suffix = ""
    logger.debug('tts  playing %s%s', _clip(phrase), suffix)
