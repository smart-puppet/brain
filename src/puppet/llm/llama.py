from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Iterator

from puppet.core.types import Conversation
from puppet.llm.base import LlmBackend
from puppet.llm.binding import validate_llama_binding
from puppet.llm.perf import (
  LlamaPerfStats,
  format_llama_perf,
  read_llama_perf,
  reset_llama_perf,
)

logger = logging.getLogger(__name__)

_TERNARY_HINT = (
  "Ternary-Bonsai Q2_0 needs llama-cpp-python built from the PrismML fork. "
  "Run: ./scripts/build_llama_prism.sh and set llm.binding: prism in config/llm.yaml "
  "(see docs/deployment.md). If test_llm.py works but puppet fails, parakeet may "
  "have loaded on CUDA first and exhausted GPU VRAM."
)


def _resolve_model_path(model_path: str) -> Path:
  path = Path(model_path).expanduser()
  if path.is_file():
    return path.resolve()
  return path.resolve()


def _model_load_hint(path: Path) -> str | None:
  name = path.name
  if "ternary-bonsai" in name.lower() or "q2_0" in name.lower():
    return _TERNARY_HINT
  if not path.is_file():
    return f"Model file not found: {path}"
  return None


def _load_llama(model_path: str, **kwargs: Any) -> Any:
  from llama_cpp import Llama

  path = _resolve_model_path(model_path)
  hint = _model_load_hint(path)
  try:
    return Llama(model_path=str(path), verbose=False, **kwargs)
  except ValueError as exc:
    if hint:
      raise RuntimeError(f"Failed to load LLM {path.name}: {hint}") from exc
    raise RuntimeError(f"Failed to load LLM from {path}: {exc}") from exc


def _resolve_ggml_type(value: str | int) -> int:
  """Map a llama.cpp type name (e.g. ``q4_0``) to ``llama_cpp.GGML_TYPE_*``."""
  if isinstance(value, int):
    return value
  try:
    import llama_cpp
  except ImportError as exc:
    raise RuntimeError("llama-cpp-python not installed") from exc

  key = f"GGML_TYPE_{str(value).upper().replace('-', '_')}"
  try:
    return int(getattr(llama_cpp, key))
  except AttributeError as exc:
    raise ValueError(f"Unknown GGML cache type: {value!r}") from exc


class LlamaLlm(LlmBackend):
  """LLM via llama-cpp-python with token streaming."""

  def __init__(
    self,
    model_path: str,
    *,
    n_ctx: int = 8096,
    n_gpu_layers: int = -1,
    n_batch: int = 256,
    n_threads: int = 0,
    type_k: str | int = "q4_0",
    type_v: str | int = "q4_0",
    flash_attn: bool = True,
    temperature: float = 0.7,
    top_p: float = 0.95,
    top_k: int = 40,
    presence_penalty: float = 0.0,
    max_tokens: int = 512,
    system_prompt: str = "",
  ) -> None:
    try:
      import llama_cpp  # noqa: F401
    except ImportError as exc:
      raise RuntimeError(
        "llama-cpp-python not installed. pip install llama-cpp-python"
      ) from exc

    self._max_tokens = max_tokens
    self._temperature = temperature
    self._top_p = float(top_p)
    self._top_k = int(top_k)
    self._presence_penalty = float(presence_penalty)
    self._system_prompt = system_prompt.strip()
    self._n_ctx = n_ctx
    self._cancelled = False
    self._last_perf: LlamaPerfStats | None = None
    self._vision_hint_fn = None
    self._vision_refresh_fn = None
    llama_kwargs: dict[str, Any] = dict(
      n_ctx=n_ctx,
      n_gpu_layers=n_gpu_layers,
      n_batch=n_batch,
      type_k=_resolve_ggml_type(type_k),
      type_v=_resolve_ggml_type(type_v),
      flash_attn=flash_attn,
    )
    if n_threads > 0:
      llama_kwargs["n_threads"] = n_threads
    self._llm = _load_llama(model_path, **llama_kwargs)
    thread_note = str(n_threads) if n_threads > 0 else "default"
    logger.info(
      "Loaded LLM: %s (n_ctx=%d, n_batch=%d, n_threads=%s, type_k=%s, type_v=%s, flash_attn=%s)",
      model_path,
      n_ctx,
      n_batch,
      thread_note,
      type_k,
      type_v,
      flash_attn,
    )

  def warmup(self, *, max_tokens: int = 8, prompt: str = "Hi", stream: bool = True) -> None:
    """Run a tiny completion so the first real reply avoids cold-start latency."""
    messages: list[dict[str, str]] = []
    if self._system_prompt:
      messages.append({"role": "system", "content": self._system_prompt})
    messages.append({"role": "user", "content": prompt})
    started = time.monotonic()
    try:
      if stream:
        stream_out = self._llm.create_chat_completion(
          messages=messages,
          stream=True,
          temperature=0.0,
          max_tokens=max(1, max_tokens),
        )
        for _ in stream_out:
          pass
      else:
        self._llm.create_chat_completion(
          messages=messages,
          stream=False,
          temperature=0.0,
          max_tokens=max(1, max_tokens),
        )
    except RuntimeError as exc:
      logger.warning("LLM warmup failed: %s", exc)
      return
    elapsed_ms = (time.monotonic() - started) * 1000.0
    mode = "stream" if stream else "batch"
    logger.info(
      "LLM warmup complete (%.0f ms, %s, max_tokens=%d)",
      elapsed_ms,
      mode,
      max_tokens,
    )

  @property
  def last_perf(self) -> LlamaPerfStats | None:
    return self._last_perf

  def set_vision_hint_fn(self, fn) -> None:
    """Optional callable returning a short vision context line for the system prompt."""
    self._vision_hint_fn = fn

  def set_vision_refresh_fn(self, fn) -> None:
    """Optional callable invoked before each reply to request a fresh eyes capture."""
    self._vision_refresh_fn = fn

  def _system_with_vision(self) -> str:
    base = self._system_prompt
    if not self._vision_hint_fn:
      return base
    try:
      line = (self._vision_hint_fn() or "").strip()
    except Exception:
      return base
    if not line:
      return base
    vision_block = (
      "Private robot state. Never copy or read it aloud. "
      "Never output CameraJSON, SEEING, PATH, RANGES, Vision, or meter readings. "
      "Object names in CameraJSON are always English (YOLO labels). "
      "When you speak, translate every object name into the child's spoken language "
      "(e.g. French: bed→lit, chair→chaise, plant→plante; German: bed→Bett). "
      "If objects is non-empty and they asked what you see, answer in your own short kid words. "
      "If they asked what you see and objects is empty or stale_s is large, say you are looking "
      "and add <<look>> as a hidden last line. "
      "To roll after you speak, add one hidden last line: <<follow>> or <<seek>> or <<stop>> or <<back>>. "
      "<<seek>> means you look for the child; you cannot hide. "
      "Do not count to ten yourself; a separate voice does that, then you search the room. "
      "If BodyStatus already says following or searching, do not add <<follow>> or <<seek>> again. "
      "If they ask to stop, stay, or stand still, you MUST add <<stop>>. "
      "No tag for stories or normal chat. Never say the tags out loud.\n"
      f"{line}"
    )
    if base:
      return f"{base}\n\n{vision_block}"
    return vision_block

  def _build_messages(self, conversation: Conversation) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    system = self._system_with_vision()
    if system:
      messages.append({"role": "system", "content": system})
    for msg in conversation.prompt_messages():
      messages.append({"role": msg.role, "content": msg.content})
    return messages

  def stream_reply(self, conversation: Conversation) -> Iterator[str]:
    self._cancelled = False
    self._last_perf = None
    if self._vision_refresh_fn is not None:
      try:
        self._vision_refresh_fn()
      except Exception as exc:  # noqa: BLE001
        logger.warning("Vision refresh before reply failed: %s", exc)
    messages = self._build_messages(conversation)
    reset_llama_perf(self._llm)
    trace_logger = logging.getLogger("puppet.trace")
    try:
      stream = self._llm.create_chat_completion(
        messages=messages,
        stream=True,
        temperature=self._temperature,
        top_p=self._top_p,
        top_k=self._top_k,
        presence_penalty=self._presence_penalty,
        max_tokens=self._max_tokens,
      )
      for chunk in stream:
        if self._cancelled:
          break
        delta = chunk["choices"][0]["delta"]
        token = delta.get("content") or ""
        if token:
          yield token
    except RuntimeError as exc:
      if "llama_decode" in str(exc):
        logger.error(
          "LLM decode failed (often GPU OOM on Jetson with concurrent STT). "
          "Try lowering llm.n_ctx or n_batch."
        )
      raise
    finally:
      try:
        self._last_perf = read_llama_perf(self._llm, n_ctx=self._n_ctx)
      except RuntimeError:
        self._last_perf = None
      if self._last_perf is not None and trace_logger.isEnabledFor(logging.DEBUG):
        trace_logger.debug("llm  perf %s", format_llama_perf(self._last_perf))

  def cancel(self) -> None:
    self._cancelled = True


def create_llm(config: dict[str, Any]) -> LlmBackend:
  llm_cfg = config.get("llm", {})
  backend = llm_cfg.get("backend", "llama")
  if backend != "llama":
    raise ValueError(f"Unsupported LLM backend: {backend!r} (only 'llama' is supported)")
  validate_llama_binding(config)
  llm = LlamaLlm(
    model_path=llm_cfg["model_path"],
    n_ctx=llm_cfg.get("n_ctx", 8096),
    n_gpu_layers=llm_cfg.get("n_gpu_layers", -1),
    n_batch=llm_cfg.get("n_batch", 256),
    n_threads=int(llm_cfg.get("n_threads", 0)),
    type_k=llm_cfg.get("type_k", "q4_0"),
    type_v=llm_cfg.get("type_v", "q4_0"),
    flash_attn=bool(llm_cfg.get("flash_attn", True)),
    temperature=llm_cfg.get("temperature", 0.7),
    top_p=float(llm_cfg.get("top_p", 0.95)),
    top_k=int(llm_cfg.get("top_k", 40)),
    presence_penalty=float(llm_cfg.get("presence_penalty", 0.0)),
    max_tokens=llm_cfg.get("max_tokens", 512),
    system_prompt=llm_cfg.get("system_prompt", ""),
  )
  if llm_cfg.get("warmup", True):
    llm.warmup(
      max_tokens=int(llm_cfg.get("warmup_max_tokens", 8)),
      prompt=str(llm_cfg.get("warmup_prompt", "Hi")),
      stream=bool(llm_cfg.get("warmup_stream", True)),
    )
  return llm
