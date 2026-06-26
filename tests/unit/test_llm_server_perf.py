from __future__ import annotations

import pytest

from puppet.llm.perf import LlamaPerfStats, format_llama_perf_cli


def test_server_timings_to_perf() -> None:
  stats = LlamaPerfStats.from_server(
    timings={
      "cache_n": 120,
      "prompt_n": 200,
      "prompt_ms": 950.0,
      "predicted_n": 42,
      "predicted_ms": 2900.0,
    },
    usage={
      "prompt_tokens": 200,
      "completion_tokens": 42,
      "total_tokens": 242,
      "prompt_tokens_details": {"cached_tokens": 120},
    },
    n_ctx=8192,
    ttft_ms=480.0,
    wall_ms=3500.0,
  )
  assert stats.prompt_tokens == 200
  assert stats.generation_tokens == 42
  assert stats.cache_tokens == 120
  assert stats.context_tokens == 242
  assert stats.n_ctx == 8192
  assert stats.context_pct == pytest.approx(100.0 * 242 / 8192)
  cli = format_llama_perf_cli(stats)
  assert "ctx 242/8192 tok" in cli
  assert "Prompt:" in cli
  assert "Generation:" in cli
