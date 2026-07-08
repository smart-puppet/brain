from __future__ import annotations

import pytest

from puppet.orchestrator.latency import TurnLatencyTracker


def test_ms_speech_end_to_first_audio() -> None:
  tracker = TurnLatencyTracker()
  tracker._vad_end = 1.0
  tracker._first_speaker = 2.1
  assert tracker.ms_speech_end_to_first_audio() == pytest.approx(1100.0)


def test_turn_latency_tracker_ms_since_generation_start(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  times = iter([0.0, 1.0, 1.4])
  monkeypatch.setattr("puppet.orchestrator.latency.time.monotonic", lambda: next(times))

  tracker = TurnLatencyTracker()
  assert tracker.ms_since_generation_start() is None
  tracker.mark_generation_start()
  assert tracker.ms_since_generation_start() == pytest.approx(400.0)


def test_turn_latency_tracker_first_stt_after_vad_end(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  times = iter([0.0, 1.0, 1.42, 2.0])
  monkeypatch.setattr("puppet.orchestrator.latency.time.monotonic", lambda: next(times))

  tracker = TurnLatencyTracker()
  assert tracker.mark_first_stt_partial() is None
  tracker.mark_vad_end()
  assert tracker.mark_first_stt_partial() == pytest.approx(420.0)
  assert tracker.mark_first_stt_partial() is None


def test_turn_latency_tracker_each_vad_end_can_measure_again(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  times = iter([0.0, 1.0, 1.2, 1.5, 1.9])
  monkeypatch.setattr("puppet.orchestrator.latency.time.monotonic", lambda: next(times))

  tracker = TurnLatencyTracker()
  tracker.mark_vad_end()
  assert tracker.mark_first_stt_partial() == pytest.approx(200.0)
  tracker.mark_vad_end()
  assert tracker.mark_first_stt_partial() == pytest.approx(400.0)


def test_turn_latency_tracker_ms_since_last_stt_partial(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  times = iter([0.0, 1.0, 1.25])
  monkeypatch.setattr("puppet.orchestrator.latency.time.monotonic", lambda: next(times))

  tracker = TurnLatencyTracker()
  assert tracker.ms_since_last_stt_partial() is None
  tracker.mark_stt_partial()
  assert tracker.ms_since_last_stt_partial() == pytest.approx(250.0)
