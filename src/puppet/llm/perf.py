from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class LlamaPerfStats:
  """Snapshot from llama.cpp perf counters or llama-server timings/usage."""

  prompt_tokens: int
  prompt_ms: float
  generation_tokens: int
  generation_ms: float
  cache_tokens: int = 0
  n_ctx: int | None = None
  context_tokens: int | None = None
  ttft_ms: float | None = None
  wall_ms: float | None = None

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

  @classmethod
  def from_server_timings(
    cls,
    timings: dict[str, float | int],
    *,
    usage: dict[str, Any] | None = None,
    n_ctx: int | None = None,
    ttft_ms: float | None = None,
    wall_ms: float | None = None,
  ) -> LlamaPerfStats:
    return cls.from_server(
      timings=timings,
      usage=usage,
      n_ctx=n_ctx,
      ttft_ms=ttft_ms,
      wall_ms=wall_ms,
    )

  @classmethod
  def from_server(
    cls,
    *,
    timings: dict[str, float | int] | None = None,
    usage: dict[str, Any] | None = None,
    n_ctx: int | None = None,
    ttft_ms: float | None = None,
    wall_ms: float | None = None,
  ) -> LlamaPerfStats:
    timings = timings or {}
    usage = usage or {}
    prompt_tokens = int(timings.get("prompt_n") or usage.get("prompt_tokens") or 0)
    generation_tokens = int(
      timings.get("predicted_n") or usage.get("completion_tokens") or 0
    )
    cache_tokens = max(0, int(timings.get("cache_n") or 0))
    details = usage.get("prompt_tokens_details") or {}
    if isinstance(details, dict):
      cache_tokens = max(cache_tokens, int(details.get("cached_tokens") or 0))
    context_tokens = int(usage.get("total_tokens") or 0)
    if context_tokens <= 0 and (prompt_tokens or generation_tokens):
      context_tokens = prompt_tokens + generation_tokens
    return cls(
      prompt_tokens=prompt_tokens,
      prompt_ms=float(timings.get("prompt_ms") or 0.0),
      generation_tokens=generation_tokens,
      generation_ms=float(timings.get("predicted_ms") or 0.0),
      cache_tokens=cache_tokens,
      n_ctx=n_ctx,
      context_tokens=context_tokens or None,
      ttft_ms=ttft_ms,
      wall_ms=wall_ms,
    )


def reset_llama_perf(llama_model: object) -> None:
  """Reset llama.cpp perf counters on a ``llama_cpp.Llama`` instance."""
  import llama_cpp

  ctx = getattr(llama_model, "ctx", None)
  if ctx is None:
    return
  llama_cpp.llama_perf_context_reset(ctx)


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
  context_tokens = prompt_tokens + generation_tokens if (prompt_tokens or generation_tokens) else None
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
    pct_s = f" ({pct:.0f}%)" if pct is not None else ""
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
  if stats.cache_tokens > 0:
    parts.append(f"cache {stats.cache_tokens} tok reused")
  if stats.ttft_ms is not None:
    parts.append(f"ttft {stats.ttft_ms:.0f} ms")
  if stats.wall_ms is not None:
    parts.append(f"wall {stats.wall_ms:.0f} ms")
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
  if stats.cache_tokens > 0:
    extras.append(f"cache {stats.cache_tokens} tok")
  if stats.ttft_ms is not None:
    extras.append(f"ttft {stats.ttft_ms:.0f} ms")
  if extras:
    line = f"{line} ({', '.join(extras)})"
  return line
