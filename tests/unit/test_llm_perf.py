from __future__ import annotations

import pytest

from puppet.llm.perf import LlamaPerfStats, format_llama_perf, kv_context_tokens_from_ctx, read_llama_perf


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


def test_kv_context_tokens_from_ctx_uses_seq_pos_max(monkeypatch) -> None:
  pytest.importorskip("llama_cpp")
  import llama_cpp

  monkeypatch.setattr(llama_cpp, "llama_get_memory", lambda ctx: object())
  monkeypatch.setattr(llama_cpp, "llama_memory_seq_pos_max", lambda mem, seq: 65)

  assert kv_context_tokens_from_ctx(object()) == 66


def test_kv_context_tokens_from_ctx_empty_cache(monkeypatch) -> None:
  pytest.importorskip("llama_cpp")
  import llama_cpp

  monkeypatch.setattr(llama_cpp, "llama_get_memory", lambda ctx: object())
  monkeypatch.setattr(llama_cpp, "llama_memory_seq_pos_max", lambda mem, seq: -1)

  assert kv_context_tokens_from_ctx(object()) is None


def test_read_llama_perf_prefers_kv_fill(monkeypatch) -> None:
  pytest.importorskip("llama_cpp")
  import llama_cpp

  class Raw:
    n_p_eval = 12
    n_eval = 24
    t_p_eval_ms = 100.0
    t_eval_ms = 500.0

  model = type("M", (), {"ctx": object()})()
  monkeypatch.setattr(llama_cpp, "llama_perf_context", lambda ctx: Raw())
  monkeypatch.setattr(llama_cpp, "llama_n_ctx", lambda ctx: 8192)
  monkeypatch.setattr(llama_cpp, "llama_get_memory", lambda ctx: object())
  monkeypatch.setattr(llama_cpp, "llama_memory_seq_pos_max", lambda mem, seq: 499)

  stats = read_llama_perf(model, n_ctx=8192)
  assert stats.prompt_tokens == 12
  assert stats.generation_tokens == 24
  assert stats.context_tokens == 500
