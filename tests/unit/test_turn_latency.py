from __future__ import annotations

import pytest

from puppet.orchestrator.latency import (
  TurnLatencyReport,
  TurnLatencyTracker,
  format_turn_latency_line,
)


def test_turn_latency_report_phases() -> None:
  tracker = TurnLatencyTracker()
  tracker._turn_start = 0.0
  tracker._vad_end = 1.0
  tracker._utterance_end = 1.2
  tracker._generation_start = 1.4
  tracker._first_llm_token = 1.9
  tracker._first_tts_phrase = 2.0
  tracker._first_speaker = 2.1
  tracker._heard_lead_in_ms = 80.0
  tracker._output_latency_ms = 20.0
  tracker._turn_end = 2.5

  report = tracker.report()
  assert report is not None
  assert report.gap_ms == pytest.approx(400.0)
  assert report.ttft_ms == pytest.approx(500.0)
  assert report.phrase_ms == pytest.approx(200.0)
  assert report.speech_to_speaker_ms == pytest.approx(1100.0)
  assert report.tts_ms == pytest.approx(300.0)
  assert report.speech_to_heard_ms == pytest.approx(1200.0)
  assert report.play_ms == pytest.approx(400.0)
  assert report.total_ms == pytest.approx(1500.0)


def test_format_turn_latency_line_bar_matches_headline() -> None:
  report = TurnLatencyReport(
    gap_ms=400,
    ttft_ms=500,
    phrase_ms=200,
    play_ms=4000,
    total_ms=5100,
    heard_lead_in_ms=80,
    output_latency_ms=20,
  )
  line = format_turn_latency_line(
    report,
    llm_wall_ms=2353.0,
    llm_perf="ctx 100/8192 tok | Prompt: 200.0 t/s | Generation: 15.0 t/s",
  )
  assert line.startswith("latency 1200ms [")
  assert "wait 400ms" in line
  assert "llm 500ms" in line
  assert "tts 300ms" in line
  assert "play" not in line
  assert "llm_wall 2353ms" in line
  assert "Generation: 15.0 t/s" in line
  # Playback length must not dominate the bar (would be all ░ if included).
  bar = line[line.index("[") + 1 : line.index("]")]
  assert bar.count("░") < len(bar) / 2


def test_heard_uses_vad_end_not_last_stt() -> None:
  tracker = TurnLatencyTracker()
  tracker._vad_end = 1.0
  tracker._utterance_end = 1.5
  tracker._generation_start = 1.6
  tracker._first_llm_token = 1.7
  tracker._first_speaker = 1.8
  tracker._turn_end = 2.0
  report = tracker.report()
  assert report is not None
  assert report.gap_ms == pytest.approx(600.0)
  assert report.speech_to_heard_ms == pytest.approx(800.0)
