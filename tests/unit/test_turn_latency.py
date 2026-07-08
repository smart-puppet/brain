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
  tracker._turn_end = 2.5

  report = tracker.report()
  assert report is not None
  assert report.gap_ms == pytest.approx(200.0)
  assert report.ttft_ms == pytest.approx(500.0)
  assert report.phrase_ms == pytest.approx(200.0)
  assert report.speech_to_speaker_ms == pytest.approx(900.0)
  assert report.speech_to_heard_ms == pytest.approx(820.0)
  assert report.play_ms == pytest.approx(400.0)
  assert report.total_ms == pytest.approx(1300.0)


def test_format_turn_latency_line_includes_bar_and_perf() -> None:
  report = TurnLatencyReport(
    gap_ms=300,
    ttft_ms=250,
    phrase_ms=250,
    play_ms=200,
    total_ms=1000,
    heard_lead_in_ms=50,
  )
  line = format_turn_latency_line(
    report,
    llm_wall_ms=2353.0,
    llm_perf="ctx 100/8192 tok | Prompt: 200.0 t/s | Generation: 15.0 t/s",
  )
  assert "heard 750ms" in line
  assert "speaker 800ms" in line
  assert line.startswith("latency heard")
  assert "gap 300ms" in line
  assert "ttft 250ms" in line
  assert "phrase 250ms" in line
  assert "play 200ms" in line
  assert "llm_wall 2353ms" in line
  assert "Generation: 15.0 t/s" in line
