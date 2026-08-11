"""Subscribe to robot/nav/scene and request captures on robot/nav/capture."""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
import uuid
from typing import Any, Optional

logger = logging.getLogger(__name__)

_VISION_DUMP_RE = re.compile(
  r"(?i)(?:\bvision\s*:|\bseeing\s*:|\bpath\s*:|\branges\s*:|\bcamerajson\b|"
  r"\bnearby things\b|\bnearest object\b|\bpath/floor\b|\bopen ranges\b|"
  r"\bprivate (?:sensor|camera)\b|m to the left|m to the right|\|\s*PATH|\|\s*RANGES)"
)


def looks_like_vision_dump(text: str) -> bool:
  """True if model output is parroting camera/system vision notes."""
  if not text or not text.strip():
    return False
  if _VISION_DUMP_RE.search(text):
    return True
  # Compact JSON camera blob echoed back
  if '"objects"' in text and ("\"path\"" in text or "side" in text):
    return True
  if text.count("|") >= 2 and ("m to the" in text.lower() or "~" in text):
    return True
  return False


class SceneIngest:
  """Caches eyes traversability scenes and can request a fresh capture."""

  def __init__(
    self,
    *,
    broker: str = "127.0.0.1",
    port: int = 1883,
    topic: str = "robot/nav/scene",
    capture_topic: str = "robot/nav/capture",
    min_interval_s: float = 1.0,
    capture_timeout_s: float = 60.0,
    capture_view: str = "traverse",
  ) -> None:
    self.broker = broker
    self.port = port
    self.topic = topic
    self.capture_topic = capture_topic
    self.min_interval_s = min_interval_s
    self.capture_timeout_s = capture_timeout_s
    self.capture_view = capture_view if capture_view in ("boxes", "traverse") else "traverse"
    self._lock = threading.Lock()
    self._hint = ""
    self._objects: list[dict[str, Any]] = []
    self._scene: dict[str, Any] = {}
    self._ts = 0.0
    self._client = None
    self._error: Optional[str] = None
    self._scene_event = threading.Event()

  def start(self) -> None:
    try:
      import paho.mqtt.client as mqtt
    except ImportError:
      self._error = "paho-mqtt not installed"
      logger.warning(self._error)
      return
    client = mqtt.Client(
      mqtt.CallbackAPIVersion.VERSION2,
      client_id=f"puppet_scene_{os.getpid()}",
    )
    client.on_connect = self._on_connect
    client.on_message = self._on_message
    try:
      client.connect(self.broker, self.port, keepalive=30)
      client.loop_start()
      self._client = client
      logger.info(
        "Vision MQTT subscribed to %s @ %s:%s (capture=%s)",
        self.topic,
        self.broker,
        self.port,
        self.capture_topic,
      )
    except Exception as exc:  # noqa: BLE001
      self._error = str(exc)
      logger.warning("Vision MQTT connect failed: %s", exc)

  def stop(self) -> None:
    if self._client is not None:
      self._client.loop_stop()
      self._client.disconnect()
      self._client = None

  def _on_connect(self, client, userdata, flags, reason_code, properties=None):
    client.subscribe(self.topic)

  def _on_message(self, client, userdata, msg):
    try:
      payload = json.loads(msg.payload.decode("utf-8"))
    except Exception:
      return
    if not isinstance(payload, dict):
      return
    now = time.time()
    with self._lock:
      self._scene = payload
      self._hint = str(payload.get("hint") or "")
      objs = payload.get("objects") or []
      if isinstance(objs, list):
        self._objects = objs[:8]
      self._ts = now
    self._scene_event.set()

  def request_capture(
    self,
    *,
    view: Optional[str] = None,
    timeout_s: Optional[float] = None,
  ) -> dict[str, Any]:
    """Publish robot/nav/capture and wait for a matching robot/nav/scene."""
    if self._client is None:
      return {"ok": False, "error": self._error or "mqtt not connected"}
    use_view = view or self.capture_view
    if use_view not in ("boxes", "traverse"):
      use_view = "traverse"
    timeout = float(timeout_s if timeout_s is not None else self.capture_timeout_s)
    req_id = uuid.uuid4().hex
    body = {"req_id": req_id, "view": use_view, "timeout_s": timeout}
    try:
      self._client.publish(self.capture_topic, json.dumps(body), qos=1)
    except Exception as exc:  # noqa: BLE001
      return {"ok": False, "error": str(exc), "req_id": req_id}

    deadline = time.time() + timeout
    while time.time() < deadline:
      with self._lock:
        scene = dict(self._scene)
      if scene.get("req_id") == req_id:
        return {"ok": True, **scene}
      remaining = deadline - time.time()
      if remaining <= 0:
        break
      self._scene_event.clear()
      self._scene_event.wait(timeout=min(0.25, remaining))

    with self._lock:
      scene = dict(self._scene)
    if scene.get("req_id") == req_id:
      return {"ok": True, **scene}
    return {
      "ok": False,
      "error": "capture timeout (is eyes listening on robot/nav/capture?)",
      "req_id": req_id,
    }

  def context_line(self) -> str:
    """Compact private camera JSON for the system prompt (not for speaking)."""
    with self._lock:
      objects = list(self._objects)
      hint = self._hint
      age = time.time() - self._ts

    # Prefer objects; omit path when objects exist so the LLM does not fixate on floor.
    payload: dict[str, Any] = {
      "objects": [
        {
          "name": str(o.get("label") or "thing").replace("_", " "),
          "side": o.get("bearing") or "center",
          "m": o.get("dist_m"),
        }
        for o in objects[:6]
      ],
    }
    if not objects:
      payload["path"] = hint or None
    if age > 5.0:
      payload["stale_s"] = int(age)
    return "CameraJSON: " + json.dumps(payload, ensure_ascii=True, separators=(",", ":"))

  def has_objects(self) -> bool:
    with self._lock:
      return bool(self._objects)

  def object_mention_tokens(self) -> list[str]:
    """Tokens that should appear in a good spoken answer about current objects."""
    with self._lock:
      objects = list(self._objects)
    tokens: list[str] = []
    aliases = {
      "potted plant": ["plant", "plante", "pflanze", "pot", "flower", "fleur"],
      "person": ["person", "personne", "people", "gens", "somebody", "quelqu"],
      "vase": ["vase"],
      "chair": ["chair", "chaise", "stuhl"],
      "couch": ["couch", "sofa", "canape"],
      "bottle": ["bottle", "bouteille", "flasche"],
      "cup": ["cup", "tasse", "becher"],
      "book": ["book", "livre", "buch"],
      "tv": ["tv", "television", "tele"],
      "laptop": ["laptop", "ordinateur", "computer"],
    }
    for o in objects:
      label = str(o.get("label") or "").replace("_", " ").lower()
      if not label:
        continue
      tokens.append(label)
      tokens.extend(label.split())
      tokens.extend(aliases.get(label, []))
    # unique, skip tiny tokens
    out: list[str] = []
    for t in tokens:
      t = t.lower().strip()
      if len(t) >= 3 and t not in out:
        out.append(t)
    return out

  def reply_mentions_objects(self, reply: str) -> bool:
    if not reply or not self.has_objects():
      return False
    text = reply.lower()
    return any(tok in text for tok in self.object_mention_tokens())

  def spoken_glimpse(self, lang: str = "en") -> str:
    """Kid-friendly spoken line from the latest objects (never raw CameraJSON)."""
    with self._lock:
      objects = list(self._objects)
      age = time.time() - self._ts

    lang = (lang or "en").lower()[:2]
    if age > 8.0 and not objects:
      return {
        "en": "I cannot see clearly right now.",
        "fr": "Je ne vois pas bien pour l'instant.",
        "de": "Ich kann gerade nicht gut sehen.",
      }.get(lang, "I cannot see clearly right now.")

    if not objects:
      return {
        "en": "I do not notice any special objects right now.",
        "fr": "Je ne remarque pas d'objet special pour l'instant.",
        "de": "Ich sehe gerade keine besonderen Dinge.",
      }.get(lang, "I do not notice any special objects right now.")

    parts: list[str] = []
    for o in objects[:3]:
      name = str(o.get("label") or "thing").replace("_", " ")
      name = {
        "potted plant": {"en": "plant", "fr": "plante", "de": "Pflanze"},
        "person": {"en": "person", "fr": "personne", "de": "Person"},
        "vase": {"en": "vase", "fr": "vase", "de": "Vase"},
        "chair": {"en": "chair", "fr": "chaise", "de": "Stuhl"},
      }.get(name, {}).get(lang, name)
      side = str(o.get("bearing") or "center")
      side_word = {
        "left": {"en": "on the left", "fr": "a gauche", "de": "links"},
        "right": {"en": "on the right", "fr": "a droite", "de": "rechts"},
        "center": {"en": "in front of me", "fr": "devant moi", "de": "vor mir"},
      }.get(side, {}).get(lang, side)
      parts.append(f"{name} {side_word}")

    if lang == "fr":
      def _fr_np(part: str) -> str:
        # part is like "plante a gauche"
        noun = part.split()[0]
        article = "un" if noun in ("vase",) else "une"
        return f"{article} {part}"

      if len(parts) == 1:
        return f"Je vois {_fr_np(parts[0])}!"
      return "Je vois " + ", ".join(_fr_np(p) for p in parts[:-1]) + f" et {_fr_np(parts[-1])}!"
    if lang == "de":
      if len(parts) == 1:
        return f"Ich sehe {parts[0]}!"
      return "Ich sehe " + ", ".join(parts[:-1]) + f" und {parts[-1]}!"
    if len(parts) == 1:
      return f"I see a {parts[0]}!"
    return "I see " + ", ".join(parts[:-1]) + f", and {parts[-1]}!"
