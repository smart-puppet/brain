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

# Frozen across turns so llama.cpp can reuse the KV prefix. Mutable CameraJSON
# / BodyStatus are appended only to the current user message.
# One locale only: mixing EN/FR/DE in this block pulls Gemma into the wrong language.
_VISION_BY_LANG: dict[str, str] = {
  "en": (
    "Private robot state may appear at the end of the last user message "
    "(BodyStatus / CameraJSON). Never copy or read it aloud. "
    "Never output CameraJSON, SEEING, PATH, RANGES, Vision, or meter readings. "
    "Object names in CameraJSON are always English YOLO labels. "
    "When you speak, use ordinary English (potted plant → plant). "
    "Always speak at least one short sentence. Never reply with only a tag. "
    "If they asked what you see (including now / again / and there), answer from CameraJSON "
    "objects in your own words — never copy JSON aloud. "
    "If they asked what you see and there is no CameraJSON, say you are looking "
    "and add <<look>> as a hidden last line. Do not invent or repeat old objects. "
    "Do not add <<look>> for greetings, thanks, or small talk. "
    "Only add <<follow>> when they asked you to follow, come closer, come here, or come to them. "
    "Only add <<seek>> when they asked to play hide-and-seek. "
    "Only add <<back>> when they asked you to reverse. "
    "<<seek>> means you look for them; you cannot hide. "
    "Do not count to ten yourself; a separate voice does that, then you search the room. "
    "If BodyStatus already says following or searching, do not add <<follow>> or <<seek>> again. "
    "Only add <<stop>> or <<idle>> when they asked to stop, stay, or stand still — never on "
    "chat like super, ok, or hello. "
    "No tag for stories or normal chat. Never say the tags out loud."
  ),
  "fr": (
    "L'état privé du robot peut apparaître à la fin du dernier message de l'enfant "
    "(BodyStatus / CameraJSON). Ne le copie jamais et ne le lis jamais à voix haute. "
    "N'écris jamais CameraJSON, SEEING, PATH, RANGES, Vision, ni des mesures en mètres. "
    "Les noms d'objets dans CameraJSON sont toujours en anglais (labels YOLO). "
    "Quand tu parles, traduis chaque nom en français (bed→lit, chair→chaise, "
    "plant→plante, person→personne). "
    "Dis toujours au moins une phrase courte à l'enfant. Ne réponds jamais avec seulement une balise. "
    "S'il a demandé ce que tu vois (y compris maintenant, encore, et là), réponds à partir des "
    "objets CameraJSON avec tes propres mots d'enfant — ne copie jamais le JSON. "
    "S'il a demandé ce que tu vois et qu'il n'y a pas de CameraJSON, dis que tu regardes "
    "et ajoute <<look>> en dernière ligne cachée. N'invente pas et ne répète pas de vieux objets. "
    "N'ajoute pas <<look>> pour un bonjour, un merci, ou du bavardage. "
    "N'ajoute <<follow>> que s'il t'a demandé de le suivre, de venir, ou de te rapprocher. "
    "N'ajoute <<seek>> que s'il a demandé à jouer à cache-cache. "
    "N'ajoute <<back>> que s'il t'a demandé de reculer. "
    "<<seek>> veut dire que TU cherches l'enfant ; tu ne peux pas te cacher. "
    "Ne compte pas jusqu'à dix toi-même ; une autre voix le fait, puis tu cherches dans la pièce. "
    "Si BodyStatus dit déjà que tu suis ou que tu cherches, n'ajoute pas <<follow>> ni <<seek>>. "
    "N'ajoute <<stop>> ou <<idle>> que s'il t'a demandé de t'arrêter, de rester, "
    "ou de ne pas bouger — jamais sur super, ok, bonjour, ou allô. "
    "Pas de balise pour les histoires ou le chat normal. Ne dis jamais les balises à voix haute."
  ),
  "de": (
    "Privater Roboterzustand kann am Ende der letzten Kind-Nachricht stehen "
    "(BodyStatus / CameraJSON). Nie vorlesen, nie kopieren. "
    "Nie CameraJSON, SEEING, PATH, RANGES, Vision oder Meterzahlen ausgeben. "
    "Objektnamen in CameraJSON sind immer englische YOLO-Labels. "
    "Beim Sprechen ins Deutsche übersetzen (bed→Bett, chair→Stuhl, plant→Pflanze, person→Person). "
    "Immer mindestens einen kurzen Satz zum Kind sagen. Nie nur ein Tag antworten. "
    "Wenn gefragt wird, was du siehst (auch jetzt, nochmal, und da), aus CameraJSON-Objekten "
    "mit eigenen Kinderwörtern antworten — JSON nie vorlesen. "
    "Wenn gefragt wird, was du siehst, und kein CameraJSON da ist, sag dass du schaust "
    "und füge <<look>> als versteckte letzte Zeile hinzu. Keine alten Objekte erfinden oder wiederholen. "
    "Kein <<look>> bei Hallo, Danke oder Smalltalk. "
    "<<follow>> nur wenn sie dich bitten zu folgen, näher zu kommen, herzukommen. "
    "<<seek>> nur beim Versteckspiel. "
    "<<back>> nur wenn sie dich bitten rückwärts zu fahren. "
    "<<seek>> heißt: DU suchst das Kind; du kannst dich nicht verstecken. "
    "Nicht selbst bis zehn zählen; eine andere Stimme zählt, dann suchst du im Zimmer. "
    "Wenn BodyStatus schon sagt dass du folgst oder suchst, kein <<follow>> oder <<seek>> nochmal. "
    "<<stop>> oder <<idle>> nur wenn sie dich bitten zu stoppen, zu bleiben, stillzustehen "
    "— nie bei super, ok, hallo. "
    "Kein Tag für Geschichten oder normales Quatschen. Tags nie laut sagen."
  ),
}


def vision_instructions(lang: str) -> str:
  key = (lang or "en").strip().lower()[:2]
  return _VISION_BY_LANG.get(key) or _VISION_BY_LANG["en"]


_VISION_INSTRUCTIONS = _VISION_BY_LANG["en"]

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
    frequency_penalty: float = 0.0,
    repeat_penalty: float = 1.0,
    max_tokens: int = 512,
    system_prompt: str = "",
    language: str = "en",
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
    self._frequency_penalty = float(frequency_penalty)
    self._repeat_penalty = float(repeat_penalty)
    self._system_prompt = system_prompt.strip()
    self._vision_instructions = vision_instructions(language)
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
    system = self._frozen_system()
    if system:
      messages.append({"role": "system", "content": system})
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
    """Optional callable returning BodyStatus / CameraJSON for the current user turn."""
    self._vision_hint_fn = fn

  def set_vision_refresh_fn(self, fn) -> None:
    """Optional callable invoked before each reply to request a fresh eyes capture."""
    self._vision_refresh_fn = fn

  def _frozen_system(self) -> str:
    """Persona + static robot rules. Must not change between turns (KV prefix)."""
    rules = getattr(self, "_vision_instructions", None) or vision_instructions("en")
    base = self._system_prompt
    if base:
      return f"{base}\n\n{rules}"
    return rules

  def _mutable_robot_state(self) -> str:
    if not self._vision_hint_fn:
      return ""
    try:
      return (self._vision_hint_fn() or "").strip()
    except Exception:
      return ""

  def _build_messages(self, conversation: Conversation) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    system = self._frozen_system()
    if system:
      messages.append({"role": "system", "content": system})
    for msg in conversation.prompt_messages():
      messages.append({"role": msg.role, "content": msg.content})
    state = self._mutable_robot_state()
    if not state:
      return messages
    blob = f"Private: {state}"
    for i in range(len(messages) - 1, -1, -1):
      if messages[i]["role"] == "user":
        messages[i] = {
          "role": "user",
          "content": f"{messages[i]['content']}\n\n{blob}",
        }
        break
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
        frequency_penalty=self._frequency_penalty,
        repeat_penalty=self._repeat_penalty,
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
  lang = str((config.get("language") or {}).get("active") or "en")
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
    frequency_penalty=float(llm_cfg.get("frequency_penalty", 0.0)),
    repeat_penalty=float(llm_cfg.get("repeat_penalty", 1.0)),
    max_tokens=llm_cfg.get("max_tokens", 512),
    system_prompt=llm_cfg.get("system_prompt", ""),
    language=lang,
  )
  if llm_cfg.get("warmup", True):
    llm.warmup(
      max_tokens=int(llm_cfg.get("warmup_max_tokens", 8)),
      prompt=str(llm_cfg.get("warmup_prompt", "Hi")),
      stream=bool(llm_cfg.get("warmup_stream", True)),
    )
  return llm
