from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class LlamaPerfStats:
  """Snapshot from llama.cpp perf counters."""

  prompt_tokens: int
  prompt_ms: float
  generation_tokens: int
  generation_ms: float
  n_ctx: int | None = None
  context_tokens: int | None = None

  @property
  def prompt_tps(self) -> float | None:
    if self.prompt_tokens <= 0 or self.prompt_ms <= 0:
      return None
    return self.prompt_tokens / (self.prompt_ms / 1000.0)

  @property
  def generation_tps(self) -> float | None:
    if self.generation_tokens <= 0 or self.generation_ms <= 0:
      return None
    return self.generation_tokens / (self.generation_ms / 1000.0)

  @property
  def context_pct(self) -> float | None:
    if self.n_ctx is None or self.n_ctx <= 0 or self.context_tokens is None:
      return None
    return 100.0 * self.context_tokens / self.n_ctx


def reset_llama_perf(llama_model: object) -> None:
  """Reset llama.cpp perf counters on a ``llama_cpp.Llama`` instance."""
  import llama_cpp

  ctx = getattr(llama_model, "ctx", None)
  if ctx is None:
    return
  llama_cpp.llama_perf_context_reset(ctx)


def kv_context_tokens_from_ctx(ctx: object, *, seq_id: int = 0) -> int | None:
  """Tokens currently stored in the llama.cpp KV cache (sequence ``seq_id``)."""
  import llama_cpp

  try:
    mem = llama_cpp.llama_get_memory(ctx)
    if mem is None:
      return None
    pos_max = int(llama_cpp.llama_memory_seq_pos_max(mem, seq_id))
  except (AttributeError, TypeError, ValueError):
    return None
  if pos_max < 0:
    return None
  return pos_max + 1


def read_llama_perf(llama_model: object, *, n_ctx: int | None = None) -> LlamaPerfStats:
  """Read llama.cpp perf counters from a ``llama_cpp.Llama`` instance."""
  import llama_cpp

  ctx = getattr(llama_model, "ctx", None)
  if ctx is None:
    raise RuntimeError("not a llama_cpp.Llama model")
  raw: Any = llama_cpp.llama_perf_context(ctx)
  prompt_tokens = int(raw.n_p_eval)
  generation_tokens = int(raw.n_eval)
  ctx_limit = n_ctx
  if ctx_limit is None:
    try:
      ctx_limit = int(llama_cpp.llama_n_ctx(ctx))
    except (AttributeError, TypeError, ValueError):
      ctx_limit = None
  context_tokens = kv_context_tokens_from_ctx(ctx)
  if context_tokens is None and (prompt_tokens or generation_tokens):
    context_tokens = prompt_tokens + generation_tokens
  return LlamaPerfStats(
    prompt_tokens=prompt_tokens,
    prompt_ms=float(raw.t_p_eval_ms),
    generation_tokens=generation_tokens,
    generation_ms=float(raw.t_eval_ms),
    n_ctx=ctx_limit,
    context_tokens=context_tokens,
  )


def format_context_usage(stats: LlamaPerfStats) -> str | None:
  if stats.context_tokens is None:
    return None
  if stats.n_ctx is not None and stats.n_ctx > 0:
    pct = stats.context_pct
    if pct is not None:
      pct_s = f" ({pct:.1f}%)" if pct < 10 else f" ({pct:.0f}%)"
    else:
      pct_s = ""
    return f"ctx {stats.context_tokens}/{stats.n_ctx} tok{pct_s}"
  return f"ctx {stats.context_tokens} tok"


def format_llama_perf(stats: LlamaPerfStats) -> str:
  parts: list[str] = []
  ctx_line = format_context_usage(stats)
  if ctx_line:
    parts.append(
      f"{ctx_line} (prompt {stats.prompt_tokens} tok + gen {stats.generation_tokens} tok)"
    )
  if stats.prompt_tps is not None:
    parts.append(
      f"prompt {stats.prompt_tokens} tok @ {stats.prompt_tps:.2f} t/s "
      f"({stats.prompt_ms:.0f} ms)"
    )
  if stats.generation_tps is not None:
    parts.append(
      f"gen {stats.generation_tokens} tok @ {stats.generation_tps:.2f} t/s "
      f"({stats.generation_ms:.0f} ms)"
    )
  return ", ".join(parts) if parts else "no tokens"


def format_llama_perf_cli(stats: LlamaPerfStats) -> str:
  """Format like llama-cli: ``Prompt: 212.2 t/s | Generation: 14.4 t/s``."""
  parts: list[str] = []
  ctx_line = format_context_usage(stats)
  if ctx_line:
    parts.append(ctx_line)
  if stats.prompt_tps is not None:
    parts.append(f"Prompt: {stats.prompt_tps:.1f} t/s")
  if stats.generation_tps is not None:
    parts.append(f"Generation: {stats.generation_tps:.1f} t/s")
  line = " | ".join(parts) if parts else "no tokens"
  extras: list[str] = []
  if stats.prompt_tokens or stats.generation_tokens:
    extras.append(f"prompt {stats.prompt_tokens} tok + gen {stats.generation_tokens} tok")
  if extras:
    line = f"{line} ({', '.join(extras)})"
  return line
