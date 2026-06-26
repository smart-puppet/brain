from __future__ import annotations

from puppet.llm.perf import LlamaPerfStats, format_llama_perf


def test_format_llama_perf_prompt_and_gen() -> None:
  stats = LlamaPerfStats(
    prompt_tokens=100,
    prompt_ms=500.0,
    generation_tokens=50,
    generation_ms=1000.0,
  )
  assert stats.prompt_tps == 200.0
  assert stats.generation_tps == 50.0
  text = format_llama_perf(stats)
  assert "prompt 100 tok @ 200.00 t/s" in text
  assert "gen 50 tok @ 50.00 t/s" in text


def test_format_llama_perf_empty() -> None:
  stats = LlamaPerfStats(prompt_tokens=0, prompt_ms=0, generation_tokens=0, generation_ms=0)
  assert format_llama_perf(stats) == "no tokens"
