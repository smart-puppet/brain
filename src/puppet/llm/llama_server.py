from __future__ import annotations

import http.client
import json
import logging
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Iterator

from puppet.core.types import Conversation, TurnSnapshot
from puppet.llm.base import LlmBackend
from puppet.llm.perf import LlamaPerfStats

logger = logging.getLogger(__name__)


def _abs_path(path: str) -> Path:
  candidate = Path(path).expanduser()
  if candidate.is_file():
    return candidate.resolve()
  if candidate.is_absolute():
    return candidate
  return (Path.cwd() / candidate).resolve()


class LlamaServerLlm(LlmBackend):
  """LLM via a local llama-server process (Prism fork / Ternary-Bonsai)."""

  def __init__(
    self,
    model_path: str,
    *,
    server_bin: str,
    host: str = "127.0.0.1",
    port: int = 8081,
    n_ctx: int = 8192,
    n_gpu_layers: int = -1,
    n_parallel: int = 1,
    n_batch: int = 256,
    n_threads: int = 0,
    type_k: str = "q4_0",
    type_v: str = "q4_0",
    flash_attn: bool = True,
    temperature: float = 0.7,
    max_tokens: int = 1024,
    system_prompt: str = "",
    startup_timeout_s: float = 120.0,
    manage_process: bool = True,
    id_slot: int = 0,
    cache_prompt: bool = True,
    prefill_at_generation: bool = False,
  ) -> None:
    self._model_path = _abs_path(model_path)
    self._server_bin = _abs_path(server_bin)
    self._host = host
    self._port = int(port)
    self._n_ctx = int(n_ctx)
    self._n_gpu_layers = int(n_gpu_layers)
    self._n_parallel = int(n_parallel)
    self._n_batch = int(n_batch)
    self._n_threads = int(n_threads)
    self._type_k = type_k
    self._type_v = type_v
    self._flash_attn = flash_attn
    self._temperature = temperature
    self._max_tokens = max_tokens
    self._system_prompt = system_prompt.strip()
    self._startup_timeout_s = startup_timeout_s
    self._manage_process = manage_process
    self._id_slot = int(id_slot)
    self._cache_prompt = cache_prompt
    self._prefill_at_generation = prefill_at_generation
    self._cancelled = False
    self._last_perf: LlamaPerfStats | None = None
    self._proc: subprocess.Popen[str] | None = None
    self._stderr_thread: threading.Thread | None = None
    self._prefill_lock = threading.Lock()
    self._prefill_cancel = threading.Event()
    self._prefill_thread: threading.Thread | None = None
    self._prefilled_draft: str | None = None
    self._prefill_inflight_draft: str | None = None

    if not self._model_path.is_file():
      raise FileNotFoundError(f"LLM model not found: {self._model_path}")
    if not self._server_bin.is_file():
      raise FileNotFoundError(f"llama-server binary not found: {self._server_bin}")

    if self._manage_process and not self._health_ok():
      self._start_server()
    elif not self._health_ok():
      raise RuntimeError(
        f"llama-server is not reachable at {self._base_url()}. "
        "Start it manually or set llm.manage_process: true."
      )
    logger.info(
      "LLM server ready: %s model=%s ctx=%d slot=%d",
      self._base_url(),
      self._model_path.name,
      self._n_ctx,
      self._id_slot,
    )

  @property
  def base_url(self) -> str:
    return self._base_url()

  @property
  def last_perf(self) -> LlamaPerfStats | None:
    return self._last_perf

  @property
  def prefilled_draft(self) -> str | None:
    return self._prefilled_draft

  def _base_url(self) -> str:
    return f"http://{self._host}:{self._port}"

  def _health_ok(self) -> bool:
    try:
      conn = http.client.HTTPConnection(self._host, self._port, timeout=1.0)
      conn.request("GET", "/health")
      resp = conn.getresponse()
      ok = resp.status == 200
      resp.read()
      conn.close()
      return ok
    except (OSError, TimeoutError):
      return False

  def _drain_server_stderr(self) -> None:
    proc = self._proc
    if proc is None or proc.stderr is None:
      return
    log = logging.getLogger("puppet.llm.server")
    try:
      for line in proc.stderr:
        text = line.rstrip()
        if text and log.isEnabledFor(logging.DEBUG):
          log.debug("%s", text)
    except (OSError, ValueError):
      pass

  def _start_server(self) -> None:
    args = [
      str(self._server_bin),
      "-m",
      str(self._model_path),
      "-c",
      str(self._n_ctx),
      "--host",
      self._host,
      "--port",
      str(self._port),
      "-ngl",
      str(self._n_gpu_layers),
      "-ctk",
      self._type_k,
      "-ctv",
      self._type_v,
      "-fa",
      "on" if self._flash_attn else "off",
      "-np",
      str(self._n_parallel),
      "-b",
      str(self._n_batch),
    ]
    if self._n_threads > 0:
      args.extend(["-t", str(self._n_threads)])
    logger.info("Starting llama-server: %s", " ".join(args))
    self._proc = subprocess.Popen(
      args,
      stdout=subprocess.DEVNULL,
      stderr=subprocess.PIPE,
      text=True,
      bufsize=1,
    )
    self._stderr_thread = threading.Thread(
      target=self._drain_server_stderr,
      daemon=True,
      name="llama-server-stderr",
    )
    self._stderr_thread.start()
    deadline = time.monotonic() + self._startup_timeout_s
    while time.monotonic() < deadline:
      if self._proc.poll() is not None:
        err = (self._proc.stderr.read() if self._proc.stderr else "")[:2000]
        raise RuntimeError(f"llama-server exited early:\n{err}")
      if self._health_ok():
        return
      time.sleep(0.5)
    raise TimeoutError(
      f"llama-server did not become ready within {self._startup_timeout_s:.0f}s"
    )

  def warmup(
    self,
    *,
    max_tokens: int = 8,
    prompt: str = "Hi",
    stream: bool = True,
  ) -> None:
    """Run a tiny streamed completion so the first real reply avoids cold-start latency."""
    messages: list[dict[str, str]] = []
    if self._system_prompt:
      messages.append({"role": "system", "content": self._system_prompt})
    messages.append({"role": "user", "content": prompt})
    started = time.monotonic()
    saved_perf = self._last_perf
    self._last_perf = None
    try:
      if stream:
        result = self._chat_once(messages, max_tokens=max(1, max_tokens), stream=True)
        assert not isinstance(result, dict)
        for _ in result:
          pass
      else:
        self._chat_once(messages, max_tokens=max(1, max_tokens), stream=False)
    except RuntimeError as exc:
      logger.warning("LLM warmup failed: %s", exc)
      self._last_perf = saved_perf
      return
    finally:
      self._last_perf = saved_perf

    elapsed_ms = (time.monotonic() - started) * 1000.0
    mode = "stream" if stream else "batch"
    logger.info(
      "LLM warmup complete (%.0f ms, %s, max_tokens=%d)",
      elapsed_ms,
      mode,
      max_tokens,
    )

  def _build_messages(self, conversation: Conversation) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    if self._system_prompt:
      messages.append({"role": "system", "content": self._system_prompt})
    for msg in conversation.prompt_messages():
      messages.append({"role": msg.role, "content": msg.content})
    return messages

  def _chat_payload(
    self,
    messages: list[dict[str, str]],
    *,
    max_tokens: int | None,
    stream: bool,
  ) -> dict[str, Any]:
    return {
      "messages": messages,
      "stream": stream,
      "stream_options": {"include_usage": True},
      "temperature": self._temperature,
      "max_tokens": max_tokens if max_tokens is not None else self._max_tokens,
      "cache_prompt": self._cache_prompt,
      "id_slot": self._id_slot,
    }

  def _chat_once(
    self,
    messages: list[dict[str, str]],
    *,
    max_tokens: int | None = None,
    stream: bool = True,
  ) -> Iterator[str] | dict[str, Any]:
    payload = self._chat_payload(messages, max_tokens=max_tokens, stream=stream)
    body = json.dumps(payload).encode("utf-8")
    trace = logging.getLogger("puppet.trace")
    if trace.isEnabledFor(logging.DEBUG):
      trace.debug(
        "llm-server POST /v1/chat/completions stream=%s slot=%d msgs=%d bytes=%d",
        stream,
        self._id_slot,
        len(messages),
        len(body),
      )

    if not stream:
      return self._request_json(body)

    started = time.monotonic()
    first_token_at: float | None = None
    timings: dict[str, float | int] | None = None
    usage: dict[str, Any] | None = None
    conn = http.client.HTTPConnection(self._host, self._port, timeout=300.0)
    try:
      conn.request(
        "POST",
        "/v1/chat/completions",
        body=body,
        headers={"Content-Type": "application/json"},
      )
      resp = conn.getresponse()
      if resp.status >= 400:
        err_body = resp.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"llama-server HTTP {resp.status}: {err_body[:500]}")

      while True:
        if self._cancelled:
          break
        raw_line = resp.readline()
        if not raw_line:
          break
        line = raw_line.decode("utf-8", errors="replace").strip()
        if not line or not line.startswith("data:"):
          continue
        chunk = line[5:].strip()
        if chunk == "[DONE]":
          break
        try:
          event = json.loads(chunk)
        except json.JSONDecodeError:
          continue

        if "timings" in event:
          timings = event["timings"]
        if "usage" in event:
          usage = event["usage"]

        choices = event.get("choices") or []
        if not choices:
          continue
        delta = choices[0].get("delta") or {}
        token = delta.get("content") or ""
        if not token:
          continue
        if first_token_at is None:
          first_token_at = time.monotonic()
        yield token
    finally:
      conn.close()

    ttft_ms = (first_token_at - started) * 1000.0 if first_token_at is not None else None
    wall_ms = (time.monotonic() - started) * 1000.0
    if timings or usage:
      self._last_perf = LlamaPerfStats.from_server(
        timings=timings,
        usage=usage,
        n_ctx=self._n_ctx,
        ttft_ms=ttft_ms,
        wall_ms=wall_ms,
      )
    elif first_token_at is not None:
      self._last_perf = LlamaPerfStats(
        prompt_tokens=0,
        prompt_ms=0.0,
        generation_tokens=0,
        generation_ms=0.0,
        ttft_ms=(first_token_at - started) * 1000.0,
        wall_ms=(time.monotonic() - started) * 1000.0,
      )

  def _request_json(self, body: bytes) -> dict[str, Any]:
    conn = http.client.HTTPConnection(self._host, self._port, timeout=300.0)
    try:
      conn.request(
        "POST",
        "/v1/chat/completions",
        body=body,
        headers={"Content-Type": "application/json"},
      )
      resp = conn.getresponse()
      raw = resp.read().decode("utf-8")
      if resp.status >= 400:
        raise RuntimeError(f"llama-server HTTP {resp.status}: {raw[:500]}")
      data = json.loads(raw)
      if "timings" in data or "usage" in data:
        self._last_perf = LlamaPerfStats.from_server(
          timings=data.get("timings"),
          usage=data.get("usage"),
          n_ctx=self._n_ctx,
        )
      return data
    finally:
      conn.close()

  def schedule_prefill(self, conversation: Conversation) -> None:
    """Warm the server KV cache for the current draft without generating tokens."""
    draft = conversation.draft_user.strip()
    if not draft:
      return
    with self._prefill_lock:
      if draft == self._prefilled_draft or draft == self._prefill_inflight_draft:
        return
      self._cancel_prefill_locked(wait=False)
      snap = conversation.snapshot()
      self._prefill_cancel = threading.Event()
      cancel = self._prefill_cancel
      self._prefill_inflight_draft = draft
      self._prefill_thread = threading.Thread(
        target=self._run_prefill,
        args=(snap, draft, cancel),
        daemon=True,
        name="llm-prefill",
      )
      self._prefill_thread.start()

  def ensure_prefill(self, conversation: Conversation) -> None:
    """Block until the current draft is prefilled (used right before generation)."""
    self.cancel_prefill(wait=True)
    draft = conversation.draft_user.strip()
    if not draft or draft == self._prefilled_draft:
      return
    started = time.monotonic()
    if self._prefill_blocking(draft, conversation.snapshot()):
      elapsed_ms = (time.monotonic() - started) * 1000.0
      logger.debug("LLM prefill sync (%.0f ms, %d chars)", elapsed_ms, len(draft))

  def cancel_prefill(self, *, wait: bool = True) -> None:
    with self._prefill_lock:
      self._cancel_prefill_locked(wait=wait)

  def _cancel_prefill_locked(self, *, wait: bool) -> None:
    self._prefill_cancel.set()
    thread = self._prefill_thread
    if wait and thread is not None and thread.is_alive() and thread is not threading.current_thread():
      thread.join(timeout=5.0)
    self._prefill_thread = None
    self._prefill_inflight_draft = None

  def _run_prefill(self, snap: TurnSnapshot, draft: str, cancel: threading.Event) -> None:
    if cancel.is_set():
      return
    ok = self._prefill_blocking(draft, snap, cancel=cancel)
    if ok:
      logger.debug("LLM prefill done (%d chars)", len(draft))

  def _prefill_blocking(
    self,
    draft: str,
    snap: TurnSnapshot,
    *,
    cancel: threading.Event | None = None,
  ) -> bool:
    if cancel is not None and cancel.is_set():
      return False
    conversation = Conversation()
    conversation.restore(snap)
    if conversation.draft_user.strip() != draft:
      return False
    messages = self._build_messages(conversation)
    saved_perf = self._last_perf
    self._last_perf = None
    try:
      self._chat_once(messages, max_tokens=0, stream=False)
    except RuntimeError as exc:
      if cancel is None or not cancel.is_set():
        logger.debug("LLM prefill failed: %s", exc)
      return False
    finally:
      self._last_perf = saved_perf
      with self._prefill_lock:
        if self._prefill_inflight_draft == draft:
          self._prefill_inflight_draft = None
    if cancel is not None and cancel.is_set():
      return False
    self._prefilled_draft = draft
    return True

  def stream_reply(self, conversation: Conversation) -> Iterator[str]:
    self._cancelled = False
    self._last_perf = None
    if self._prefill_at_generation:
      self.ensure_prefill(conversation)
    messages = self._build_messages(conversation)
    stream = self._chat_once(messages, stream=True)
    assert not isinstance(stream, dict)
    yield from stream
    self._prefilled_draft = None

  def cancel(self) -> None:
    self._cancelled = True
    self.cancel_prefill(wait=False)

  def close(self) -> None:
    if self._proc is None:
      return
    if self._proc.poll() is None:
      self._proc.terminate()
      try:
        self._proc.wait(timeout=5.0)
      except subprocess.TimeoutExpired:
        self._proc.kill()
    thread = self._stderr_thread
    if thread is not None and thread.is_alive():
      thread.join(timeout=1.0)
    self._stderr_thread = None
    self._proc = None

  def __del__(self) -> None:
    try:
      self.close()
    except Exception:
      pass


def create_llama_server(config: dict[str, Any]) -> LlamaServerLlm:
  llm_cfg = config.get("llm", {})
  default_bin = "../PrismML-Eng/Jun12-2026/build/bin/llama-server"
  llm = LlamaServerLlm(
    model_path=llm_cfg["model_path"],
    server_bin=llm_cfg.get("server_bin", default_bin),
    host=llm_cfg.get("host", "127.0.0.1"),
    port=int(llm_cfg.get("port", 8081)),
    n_ctx=int(llm_cfg.get("n_ctx", 8192)),
    n_gpu_layers=int(llm_cfg.get("n_gpu_layers", -1)),
    n_parallel=int(llm_cfg.get("n_parallel", 1)),
    n_batch=int(llm_cfg.get("n_batch", 256)),
    n_threads=int(llm_cfg.get("n_threads", 0)),
    type_k=str(llm_cfg.get("type_k", "q4_0")),
    type_v=str(llm_cfg.get("type_v", "q4_0")),
    flash_attn=bool(llm_cfg.get("flash_attn", True)),
    temperature=float(llm_cfg.get("temperature", 0.7)),
    max_tokens=int(llm_cfg.get("max_tokens", 1024)),
    system_prompt=llm_cfg.get("system_prompt", ""),
    startup_timeout_s=float(llm_cfg.get("startup_timeout_s", 120.0)),
    manage_process=bool(llm_cfg.get("manage_process", True)),
    id_slot=int(llm_cfg.get("id_slot", 0)),
    cache_prompt=bool(llm_cfg.get("cache_prompt", True)),
    prefill_at_generation=bool(llm_cfg.get("prefill_at_generation", False)),
  )
  if llm_cfg.get("warmup", True):
    llm.warmup(
      max_tokens=int(llm_cfg.get("warmup_max_tokens", 8)),
      prompt=str(llm_cfg.get("warmup_prompt", "Hi")),
      stream=bool(llm_cfg.get("warmup_stream", True)),
    )
  return llm
