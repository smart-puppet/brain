from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal, Protocol, Sequence

from piper.const import BOS, EOS, PAD

from puppet.tts.types import MouthEvent, WordCue

logger = logging.getLogger(__name__)

# Word / phrase breaks in Piper alignments (not per-phoneme jaw motion).
_WORD_BREAK_PHONEMES = frozenset({BOS, EOS, PAD, " ", "$", "^"})

# Punctuation / breaks — jaw stays closed.
_SILENCE_PHONEMES = _WORD_BREAK_PHONEMES | frozenset(
  {
    ".",
    ",",
    ";",
    ":",
    "!",
    "?",
    "-",
    "…",
    '"',
    "'",
    "«",
    "»",
    "(",
    ")",
    "[",
    "]",
  }
)


class PhonemeAlignmentLike(Protocol):
  phoneme: str
  num_samples: int


MouthMode = Literal["word", "fallback"]


def model_has_phoneme_duration(model_path: str | Path) -> bool:
  return "phoneme_duration" in Path(model_path).name


def mouth_phoneme_timeline_available(config: dict) -> bool:
  mouth_cfg = config.get("puppet", {}).get("mouth", {})
  if not mouth_cfg.get("enabled", False):
    return False
  tts_cfg = config.get("tts", {})
  model_path = tts_cfg.get("model_path")
  if not model_path:
    return False
  aligned = resolve_alignment_model_path(str(model_path), prefer_alignments=True)
  return model_has_phoneme_duration(aligned)


def resolve_mouth_mode(config: dict) -> MouthMode:
  """Pick word sync or fallback flap; word requires a phoneme_duration ONNX."""
  mouth_cfg = config.get("puppet", {}).get("mouth", {})
  configured = str(mouth_cfg.get("mode", "word")).lower()
  if configured in ("fallback", "stupid"):
    return "fallback"
  if configured not in ("word",):
    raise ValueError(
      f"puppet.mouth.mode must be 'word' or 'fallback', got {configured!r}"
    )
  if mouth_phoneme_timeline_available(config):
    return "word"
  flip_ms = int(mouth_cfg.get("fallback_flip_ms", mouth_cfg.get("stupid_flip_ms", 200)))
  logger.info(
    "No phoneme_duration ONNX for TTS model; using fallback mouth mode (%dms flap)",
    flip_ms,
  )
  return "fallback"


def resolve_alignment_model_path(model_path: str | Path, *, prefer_alignments: bool) -> str:
  path = Path(model_path)
  if model_has_phoneme_duration(path):
    return str(path)
  if not prefer_alignments:
    return str(path)
  if not path.name.endswith(".onnx"):
    return str(path)
  aligned = path.with_name(path.name.replace(".onnx", ".phoneme_duration.onnx"))
  if aligned.is_file():
    return str(aligned)
  return str(path)


def phoneme_opens_mouth(phoneme: str) -> bool:
  return phoneme not in _SILENCE_PHONEMES


@dataclass(frozen=True)
class MouthTiming:
  """Coalesce phoneme edges so a hobby servo can keep up (binary open/closed only)."""

  min_close_ms: int = 180
  min_open_ms: int = 100
  servo_move_ms: int = 200
  fallback_flip_ms: int = 200
  word_min_gap_ms: int = 80

  @classmethod
  def from_config(cls, config: dict) -> MouthTiming:
    mouth_cfg = config.get("puppet", {}).get("mouth", {})
    servo_move_ms = int(mouth_cfg.get("servo_move_ms", 200))
    return cls(
      min_close_ms=int(mouth_cfg.get("min_close_ms", 180)),
      min_open_ms=int(mouth_cfg.get("min_open_ms", 100)),
      servo_move_ms=servo_move_ms,
      fallback_flip_ms=int(mouth_cfg.get("fallback_flip_ms", mouth_cfg.get("stupid_flip_ms", 200))),
      word_min_gap_ms=int(mouth_cfg.get("word_min_gap_ms", servo_move_ms)),
    )


def _ms_to_samples(ms: int, sample_rate: int) -> int:
  return max(0, int(sample_rate * ms / 1000))


def _segments_from_events(
  events: Sequence[MouthEvent],
  total_samples: int,
) -> list[tuple[int, int, bool]]:
  if total_samples <= 0:
    return []
  if not events:
    return [(0, total_samples, False)]
  segments: list[tuple[int, int, bool]] = []
  state = False
  pos = 0
  for event in events:
    offset = int(event.sample_offset)
    if offset > pos:
      segments.append((pos, offset, state))
    state = bool(event.open)
    pos = offset
  if pos < total_samples:
    segments.append((pos, total_samples, state))
  return segments


def _merge_adjacent_segments(
  segments: Sequence[tuple[int, int, bool]],
) -> list[tuple[int, int, bool]]:
  merged: list[tuple[int, int, bool]] = []
  for start, end, open_state in segments:
    if merged and merged[-1][2] == open_state:
      prev_start, _, _ = merged[-1]
      merged[-1] = (prev_start, end, open_state)
    else:
      merged.append((start, end, open_state))
  return merged


def simplify_timeline_for_servo(
  events: Sequence[MouthEvent],
  audio_samples: int,
  sample_rate: int,
  timing: MouthTiming,
) -> list[MouthEvent]:
  """Drop brief toggles and space moves for slow servos."""
  if not events or audio_samples <= 0 or sample_rate <= 0:
    return list(events)

  min_close = _ms_to_samples(timing.min_close_ms, sample_rate)
  min_open = _ms_to_samples(timing.min_open_ms, sample_rate)
  min_move = _ms_to_samples(timing.servo_move_ms, sample_rate)

  segments = _segments_from_events(events, audio_samples)

  # Keep jaw open through short closed gaps (consonants, punctuation).
  coalesced: list[tuple[int, int, bool]] = []
  for start, end, open_state in segments:
    duration = end - start
    if not open_state and duration < min_close:
      open_state = True
    coalesced.append((start, end, open_state))
  segments = _merge_adjacent_segments(coalesced)

  # Drop brief open blips between closed regions.
  coalesced = []
  for start, end, open_state in segments:
    duration = end - start
    if open_state and duration < min_open:
      open_state = False
    coalesced.append((start, end, open_state))
  segments = _merge_adjacent_segments(coalesced)

  simplified: list[MouthEvent] = []
  last_at = -min_move
  prev_open: bool | None = None
  for start, end, open_state in segments:
    if open_state == prev_open:
      continue
    at = max(start, last_at + min_move)
    if at >= end:
      continue
    simplified.append(MouthEvent(sample_offset=at, open=open_state))
    last_at = at
    prev_open = open_state

  if simplified and simplified[-1].open:
    close_at = max(audio_samples, last_at + min_move)
    if close_at > simplified[-1].sample_offset:
      simplified.append(MouthEvent(sample_offset=close_at, open=False))

  return simplified


MouthGranularity = Literal["phoneme", "word"]


def timeline_for_audio_chunk(
  *,
  alignments: Sequence[PhonemeAlignmentLike] | None,
  audio_samples: int,
  sample_rate: int = 22050,
  timing: MouthTiming | None = None,
  granularity: MouthGranularity = "phoneme",
) -> list[MouthEvent]:
  timing = timing or MouthTiming()
  if alignments:
    if granularity == "word":
      events = word_alignments_to_timeline(
        alignments,
        sample_rate=sample_rate,
        min_open_ms=timing.min_open_ms,
        min_gap_ms=timing.word_min_gap_ms,
      )
      simplify_timing = replace(timing, min_close_ms=0, min_open_ms=0, servo_move_ms=0)
    else:
      events = phoneme_alignments_to_timeline(alignments)
      simplify_timing = timing
    aligned_samples = sum(int(a.num_samples) for a in alignments)
    total = max(audio_samples, aligned_samples)
    if aligned_samples < audio_samples and events:
      last_open = events[-1].open
      if last_open:
        events.append(MouthEvent(sample_offset=aligned_samples, open=False))
    return simplify_timeline_for_servo(events, total, sample_rate, simplify_timing)
  if audio_samples <= 0:
    return []
  raw = [MouthEvent(0, True), MouthEvent(audio_samples, False)]
  return simplify_timeline_for_servo(raw, audio_samples, sample_rate, timing)


def phoneme_alignments_to_timeline(
  alignments: Sequence[PhonemeAlignmentLike],
) -> list[MouthEvent]:
  events: list[MouthEvent] = []
  offset = 0
  last_open: bool | None = None
  for align in alignments:
    open_mouth = phoneme_opens_mouth(align.phoneme)
    if last_open is None:
      if open_mouth:
        events.append(MouthEvent(sample_offset=offset, open=True))
      last_open = open_mouth
    elif open_mouth != last_open:
      events.append(MouthEvent(sample_offset=offset, open=open_mouth))
      last_open = open_mouth
    offset += int(align.num_samples)
  return events


def _word_spans_from_alignments(
  alignments: Sequence[PhonemeAlignmentLike],
) -> list[tuple[int, int]]:
  spans: list[tuple[int, int]] = []
  offset = 0
  word_start: int | None = None
  for align in alignments:
    phoneme = align.phoneme
    if phoneme in _WORD_BREAK_PHONEMES:
      if word_start is not None:
        spans.append((word_start, offset))
        word_start = None
    elif phoneme_opens_mouth(phoneme) and word_start is None:
      word_start = offset
    offset += int(align.num_samples)
  if word_start is not None:
    spans.append((word_start, offset))
  return spans


def word_cues_from_alignments(
  alignments: Sequence[PhonemeAlignmentLike],
  *,
  sample_rate: int = 22050,
  min_open_ms: int = 100,
  min_gap_ms: int = 200,
) -> list[WordCue]:
  """Word open/close windows in milliseconds from chunk audio start."""
  if sample_rate <= 0:
    return []
  min_open_samples = _ms_to_samples(min_open_ms, sample_rate)
  min_gap_samples = _ms_to_samples(min_gap_ms, sample_rate)
  cues: list[WordCue] = []
  last_close_sample: int | None = None
  for start, end in _word_spans_from_alignments(alignments):
    if end - start < min_open_samples:
      continue
    if last_close_sample is not None and start - last_close_sample < min_gap_samples:
      continue
    cues.append(
      WordCue(
        start_ms=int(round(start * 1000 / sample_rate)),
        end_ms=int(round(end * 1000 / sample_rate)),
      )
    )
    last_close_sample = end
  return cues


def shift_word_cues(cues: Sequence[WordCue], offset_ms: int) -> list[WordCue]:
  if offset_ms <= 0:
    return list(cues)
  return [
    WordCue(start_ms=int(cue.start_ms) + offset_ms, end_ms=int(cue.end_ms) + offset_ms)
    for cue in cues
  ]


def word_alignments_to_timeline(
  alignments: Sequence[PhonemeAlignmentLike],
  *,
  sample_rate: int = 22050,
  min_open_ms: int = 100,
  min_gap_ms: int = 200,
) -> list[MouthEvent]:
  """Open at each word start, close when the word ends (split on Piper word boundaries)."""
  events: list[MouthEvent] = []
  for cue in word_cues_from_alignments(
    alignments,
    sample_rate=sample_rate,
    min_open_ms=min_open_ms,
    min_gap_ms=min_gap_ms,
  ):
    events.append(MouthEvent(sample_offset=_ms_to_samples(cue.start_ms, sample_rate), open=True))
    events.append(MouthEvent(sample_offset=_ms_to_samples(cue.end_ms, sample_rate), open=False))
  return events


def fallback_fixed_flip_holds_ms(
  audio_samples: int,
  sample_rate: int,
  *,
  flip_ms: int,
) -> list[int]:
  """Fixed-interval flips covering the whole chunk (when no phoneme_duration model)."""
  if audio_samples <= 0 or sample_rate <= 0:
    return []
  flip_ms = max(1, int(flip_ms))
  chunk_ms = int(round(audio_samples * 1000 / sample_rate))
  count = max(1, (chunk_ms + flip_ms - 1) // flip_ms)
  return [flip_ms] * count


def phoneme_hold_durations_ms(
  alignments: Sequence[PhonemeAlignmentLike],
  sample_rate: int,
  *,
  min_ms: int,
) -> list[int]:
  """Per-phoneme hold times for fallback mouth mode (each at least ``min_ms``)."""
  if sample_rate <= 0:
    return []
  floor_ms = max(1, int(min_ms))
  durations: list[int] = []
  for align in alignments:
    n = int(align.num_samples)
    if n <= 0:
      continue
    ms = int(round(n * 1000 / sample_rate))
    durations.append(max(ms, floor_ms))
  return durations


def shift_timeline(events: Sequence[MouthEvent], sample_offset: int) -> list[MouthEvent]:
  if sample_offset <= 0:
    return list(events)
  return [
    MouthEvent(sample_offset=int(event.sample_offset) + sample_offset, open=event.open)
    for event in events
  ]
