from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock

from puppet.tts.alignments import (
  model_has_phoneme_duration,
  phoneme_alignments_to_timeline,
  phoneme_hold_durations_ms,
  phoneme_opens_mouth,
  resolve_alignment_model_path,
  fallback_fixed_flip_holds_ms,
  resolve_mouth_mode,
  word_alignments_to_timeline,
  word_cues_from_alignments,
)
from puppet.tts.types import MouthEvent, WordCue


@dataclass
class _Align:
  phoneme: str
  num_samples: int


def test_model_has_phoneme_duration() -> None:
  assert model_has_phoneme_duration("de_DE-elmo-medium-v1.phoneme_duration.onnx")
  assert not model_has_phoneme_duration("de_DE-elmo-medium-v1.onnx")


def test_resolve_alignment_model_path(tmp_path: Path) -> None:
  base = tmp_path / "voice.onnx"
  aligned = tmp_path / "voice.phoneme_duration.onnx"
  base.write_bytes(b"x")
  aligned.write_bytes(b"x")

  assert resolve_alignment_model_path(base, prefer_alignments=True) == str(aligned)
  assert resolve_alignment_model_path(base, prefer_alignments=False) == str(base)
  assert resolve_alignment_model_path(aligned, prefer_alignments=False) == str(aligned)


def test_phoneme_opens_mouth() -> None:
  assert not phoneme_opens_mouth("^")
  assert not phoneme_opens_mouth(" ")
  assert phoneme_opens_mouth("a")


def test_phoneme_hold_durations_ms_floors_short_phonemes() -> None:
  alignments = [
    _Align("a", 1000),
    _Align(" ", 500),
    _Align("b", 6000),
  ]
  assert phoneme_hold_durations_ms(alignments, 22050, min_ms=200) == [200, 200, 272]


def test_word_cues_from_alignments() -> None:
  sr = 1000
  alignments = [
    _Align("^", 10),
    _Align("a", 120),
    _Align("b", 130),
    _Align(" ", 30),
    _Align("c", 120),
    _Align("d", 130),
    _Align("$", 10),
  ]
  cues = word_cues_from_alignments(
    alignments, sample_rate=sr, min_open_ms=100, min_gap_ms=0
  )
  assert cues == [WordCue(10, 260), WordCue(290, 540)]


def test_word_alignments_to_timeline_splits_on_spaces() -> None:
  sr = 1000
  alignments = [
    _Align("^", 10),
    _Align("ɪ", 120),
    _Align("c", 80),
    _Align("h", 50),
    _Align(" ", 20),
    _Align("b", 100),
    _Align("ɪ", 80),
    _Align("n", 120),
    _Align("$", 10),
  ]
  events = word_alignments_to_timeline(
    alignments, sample_rate=sr, min_open_ms=100, min_gap_ms=0
  )
  assert events == [
    MouthEvent(10, True),
    MouthEvent(260, False),
    MouthEvent(280, True),
    MouthEvent(580, False),
  ]


def test_word_alignments_skips_words_shorter_than_min_open() -> None:
  sr = 1000
  alignments = [
    _Align("^", 10),
    _Align("a", 50),
    _Align(" ", 20),
    _Align("h", 150),
    _Align("i", 80),
    _Align("$", 10),
  ]
  events = word_alignments_to_timeline(
    alignments, sample_rate=sr, min_open_ms=100, min_gap_ms=0
  )
  assert events == [
    MouthEvent(80, True),
    MouthEvent(310, False),
  ]


def test_word_alignments_skips_when_next_open_too_soon_after_close() -> None:
  sr = 1000
  alignments = [
    _Align("^", 10),
    _Align("a", 120),
    _Align("b", 130),
    _Align(" ", 30),
    _Align("c", 120),
    _Align("d", 130),
    _Align("$", 10),
  ]
  events = word_alignments_to_timeline(
    alignments, sample_rate=sr, min_open_ms=100, min_gap_ms=100
  )
  assert events == [
    MouthEvent(10, True),
    MouthEvent(260, False),
  ]


def test_timeline_for_audio_chunk_word_granularity() -> None:
  from puppet.tts.alignments import MouthTiming, timeline_for_audio_chunk

  alignments = [
    _Align("^", 100),
    _Align("h", 2205),
    _Align("i", 2205),
    _Align(" ", 100),
    _Align("b", 2205),
    _Align("y", 2205),
    _Align("$", 50),
  ]
  events = timeline_for_audio_chunk(
    alignments=alignments,
    audio_samples=9070,
    sample_rate=22050,
    timing=MouthTiming(servo_move_ms=0, min_open_ms=100, word_min_gap_ms=0),
    granularity="word",
  )
  opens = [e for e in events if e.open]
  assert len(opens) == 2
  assert opens[0].sample_offset == 110
  assert opens[1].sample_offset == 4608


def test_fallback_fixed_flip_holds_ms_covers_chunk() -> None:
  assert fallback_fixed_flip_holds_ms(22050, 22050, flip_ms=200) == [200] * 5
  assert fallback_fixed_flip_holds_ms(44100, 22050, flip_ms=200) == [200] * 10
  assert fallback_fixed_flip_holds_ms(5000, 22050, flip_ms=200) == [200, 200]


def test_resolve_mouth_mode_uses_word_when_phoneme_model_exists(tmp_path: Path) -> None:
  model = tmp_path / "voice.phoneme_duration.onnx"
  model.write_bytes(b"x")
  cfg = {
    "puppet": {"mouth": {"enabled": True, "mode": "word"}},
    "tts": {"model_path": str(model)},
  }
  assert resolve_mouth_mode(cfg) == "word"


def test_resolve_mouth_mode_falls_back_without_phoneme_model(tmp_path: Path) -> None:
  model = tmp_path / "voice.onnx"
  model.write_bytes(b"x")
  cfg = {
    "puppet": {"mouth": {"enabled": True, "mode": "word"}},
    "tts": {"model_path": str(model)},
  }
  assert resolve_mouth_mode(cfg) == "fallback"


def test_shift_timeline() -> None:
  from puppet.tts.alignments import shift_timeline

  events = [MouthEvent(100, True), MouthEvent(300, False)]
  assert shift_timeline(events, 500) == [
    MouthEvent(600, True),
    MouthEvent(800, False),
  ]


def test_timeline_for_audio_chunk_fallback() -> None:
  from puppet.tts.alignments import MouthTiming, timeline_for_audio_chunk

  events = timeline_for_audio_chunk(
    alignments=None,
    audio_samples=1000,
    sample_rate=22050,
    timing=MouthTiming(min_close_ms=0, min_open_ms=0, servo_move_ms=0),
  )
  assert events == [MouthEvent(0, True), MouthEvent(1000, False)]


def test_simplify_timeline_merges_rapid_phoneme_edges() -> None:
  from puppet.tts.alignments import MouthTiming, simplify_timeline_for_servo

  raw = [
    MouthEvent(1000, True),
    MouthEvent(1500, False),
    MouthEvent(2000, True),
    MouthEvent(2500, False),
    MouthEvent(3000, True),
    MouthEvent(10000, False),
  ]
  timing = MouthTiming(min_close_ms=180, min_open_ms=100, servo_move_ms=200)
  simplified = simplify_timeline_for_servo(raw, 12000, 22050, timing)
  assert len(simplified) <= 3
  assert simplified[0].open is True


def test_phoneme_alignments_to_timeline() -> None:
  alignments = [
    _Align("^", 100),
    _Align("h", 200),
    _Align("a", 300),
    _Align(" ", 50),
    _Align("l", 250),
    _Align("$", 100),
  ]
  events = phoneme_alignments_to_timeline(alignments)
  assert events == [
    MouthEvent(100, True),
    MouthEvent(600, False),
    MouthEvent(650, True),
    MouthEvent(900, False),
  ]


def test_piper_tts_enables_alignments_when_mouth_enabled(monkeypatch, tmp_path: Path) -> None:
  model = tmp_path / "voice.phoneme_duration.onnx"
  config = tmp_path / "voice.phoneme_duration.onnx.json"
  model.write_bytes(b"")
  config.write_text(
    '{"audio": {"sample_rate": 22050}, "phoneme_id_map": {}, "num_symbols": 1}',
    encoding="utf-8",
  )

  class FakeVoice:
    session = MagicMock()
    session.get_outputs.return_value = [MagicMock(), MagicMock()]
    config = MagicMock(sample_rate=22050)

    def synthesize(self, text, include_alignments=False):
      del text
      chunk = MagicMock()
      chunk.audio_float_array = __import__("numpy").zeros(10, dtype="float32")
      chunk.phoneme_alignments = None
      yield chunk

  monkeypatch.setattr("puppet.tts.piper._load_piper_voice", lambda **kwargs: FakeVoice())
  from puppet.tts.piper import PiperTts

  tts = PiperTts(str(model), include_alignments=True)
  assert tts.has_mouth_timeline
