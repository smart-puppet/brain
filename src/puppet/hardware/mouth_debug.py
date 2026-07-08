from __future__ import annotations

import logging
from typing import Any, Sequence

from puppet.tts.types import MouthEvent, WordCue

logger = logging.getLogger("puppet.mouth")


def configure_mouth_logging(config: dict[str, Any]) -> None:
  """Enable standalone ``puppet.mouth`` DEBUG output when ``puppet.mouth.debug`` is true."""
  mouth_cfg = config.get("puppet", {}).get("mouth", {})
  if not mouth_cfg.get("debug", False):
    return
  mouth_logger = logging.getLogger("puppet.mouth")
  mouth_logger.setLevel(logging.DEBUG)
  if mouth_logger.handlers:
    return
  handler = logging.StreamHandler()
  handler.setLevel(logging.DEBUG)
  root = logging.getLogger()
  if root.handlers and root.handlers[0].formatter is not None:
    handler.setFormatter(root.handlers[0].formatter)
  else:
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
  mouth_logger.addHandler(handler)
  mouth_logger.propagate = False


def _ms(samples: int, sample_rate: int) -> str:
  if sample_rate <= 0:
    return f"{samples} smp"
  return f"{samples} smp ({samples * 1000 / sample_rate:.0f}ms)"


class MouthDebug:
  def __init__(self, *, enabled: bool, sample_rate: int) -> None:
    self._enabled = enabled
    self._sample_rate = sample_rate

  def _log(self, msg: str, *args: object) -> None:
    if self._enabled and logger.isEnabledFor(logging.DEBUG):
      logger.debug(msg, *args)

  def reply_sync_start(self) -> None:
    self._log("reply sync start (sample clock reset)")

  def timeline_scheduled(
    self,
    events: Sequence[MouthEvent],
    *,
    reply_sample: int,
    playback_delay_samples: int,
    generation: int,
    source: str,
  ) -> None:
    if not events:
      return
    first = events[0].sample_offset
    last = events[-1].sample_offset
    opens = sum(1 for e in events if e.open)
    self._log(
      "schedule %s %d events (gen=%d reply@%s delay+%s span %s..%s, %d open)",
      source,
      len(events),
      generation,
      _ms(reply_sample, self._sample_rate),
      _ms(playback_delay_samples, self._sample_rate),
      _ms(first, self._sample_rate),
      _ms(last, self._sample_rate),
      opens,
    )
    for idx, event in enumerate(events):
      state = "OPEN" if event.open else "CLOSED"
      self._log(
        "  [%d] %s @ %s",
        idx,
        state,
        _ms(int(event.sample_offset) + playback_delay_samples, self._sample_rate),
      )

  def wait_aborted(self, *, target_samples: int, generation: int) -> None:
    self._log(
      "wait aborted @ %s (gen=%d)",
      _ms(target_samples, self._sample_rate),
      generation,
    )

  def generation_cancelled(self, generation: int) -> None:
    self._log("generation cancelled (gen=%d)", generation)

  def word_chunk_scheduled(
    self,
    cues: Sequence[WordCue],
    *,
    generation: int,
    chunk_id: int,
  ) -> None:
    if not cues:
      return
    preview = ", ".join(f"{c.start_ms}-{c.end_ms}ms" for c in cues[:6])
    if len(cues) > 6:
      preview += f", … (+{len(cues) - 6})"
    self._log("word chunk %d schedule %d cues (gen=%d): %s", chunk_id, len(cues), generation, preview)

  def timeline_skipped_stale(
    self,
    *,
    target_samples: int,
    playback_samples: int,
    generation: int,
  ) -> None:
    self._log(
      "skip stale event @ %s (playback %s, gen=%d)",
      _ms(target_samples, self._sample_rate),
      _ms(playback_samples, self._sample_rate),
      generation,
    )

  def timeline_late_catchup(
    self,
    *,
    target_samples: int,
    playback_samples: int,
    generation: int,
    open_mouth: bool,
  ) -> None:
    state = "OPEN" if open_mouth else "CLOSED"
    self._log(
      "late catch-up %s @ %s (playback %s, gen=%d)",
      state,
      _ms(target_samples, self._sample_rate),
      _ms(playback_samples, self._sample_rate),
      generation,
    )

  def phrase_chunk_open(self, generation: int) -> None:
    self._log("phrase chunk → OPEN (gen=%d)", generation)

  def phrase_end_close(self, generation: int) -> None:
    self._log("phrase end → CLOSED (gen=%d)", generation)

  def fallback_start(self, generation: int, flip_s: float) -> None:
    self._log("fallback flap start (gen=%d, min %.0fms)", generation, flip_s * 1000)

  def fallback_durations(
    self,
    generation: int,
    durations_ms: Sequence[int],
    *,
    source: str = "phoneme",
    min_ms: int = 200,
  ) -> None:
    if not durations_ms:
      return
    preview = ", ".join(str(ms) for ms in durations_ms[:8])
    if len(durations_ms) > 8:
      preview += f", … (+{len(durations_ms) - 8})"
    above_floor = sum(1 for ms in durations_ms if ms > min_ms)
    self._log(
      "fallback schedule %d holds (%s, gen=%d, %d–%dms, %d above %dms floor): %s",
      len(durations_ms),
      source,
      generation,
      min(durations_ms),
      max(durations_ms),
      above_floor,
      min_ms,
      preview,
    )

  def servo(
    self,
    *,
    open_mouth: bool,
    angle: float,
    target_samples: int,
    playback_samples: int,
    generation: int,
    source: str = "phoneme",
  ) -> None:
    state = "OPEN" if open_mouth else "CLOSED"
    self._log(
      "servo %s %.0f° [%s] @ playback %s (target %s, gen=%d)",
      state,
      angle,
      source,
      _ms(playback_samples, self._sample_rate),
      _ms(target_samples, self._sample_rate),
      generation,
    )

  def reset(self, generation: int) -> None:
    self._log("reset → closed (gen=%d)", generation)
