from __future__ import annotations

import logging

import pytest

from puppet.orchestrator.latency import TurnLatencyTracker
from puppet.orchestrator.trace import PipelineTracer


def test_pipeline_trace_milestones(caplog: pytest.LogCaptureFixture) -> None:
  latency = TurnLatencyTracker()
  trace = PipelineTracer(latency)

  with caplog.at_level(logging.DEBUG, logger="puppet.trace"):
    trace.stt_partial("hello", "hello world")
    latency.mark_generation_start()
    trace.llm_prompt("hello world")
    latency.mark_llm_token()
    trace.llm_generating()
    trace.tts_playing("Hi there.")

  messages = [r.message for r in caplog.records if r.name == "puppet.trace"]
  assert messages == [
    "stt  hello world",
    "llm  → hello world",
    "llm  generating",
    "tts  playing Hi there.",
  ]


def test_stt_partial_logs_vad_end_latency(
  caplog: pytest.LogCaptureFixture,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  times = iter([0.0, 1.0, 1.42, 2.0])
  monkeypatch.setattr("puppet.orchestrator.latency.time.monotonic", lambda: next(times))

  latency = TurnLatencyTracker()
  trace = PipelineTracer(latency)
  latency.mark_vad_end()

  with caplog.at_level(logging.DEBUG, logger="puppet.trace"):
    trace.stt_partial("hel", "hel")
    trace.stt_partial("lo", "hello")

  messages = [r.message for r in caplog.records if r.name == "puppet.trace"]
  assert messages[0] == "stt  hel (420ms since speech end)"
  assert messages[1] == "stt  hello"
