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

# English YOLO/COCO labels → spoken words per language.
_LABEL_I18N: dict[str, dict[str, str]] = {
  "person": {"en": "person", "fr": "personne", "de": "Person"},
  "bicycle": {"en": "bike", "fr": "velo", "de": "Fahrrad"},
  "car": {"en": "car", "fr": "voiture", "de": "Auto"},
  "motorcycle": {"en": "motorcycle", "fr": "moto", "de": "Motorrad"},
  "airplane": {"en": "plane", "fr": "avion", "de": "Flugzeug"},
  "bus": {"en": "bus", "fr": "bus", "de": "Bus"},
  "train": {"en": "train", "fr": "train", "de": "Zug"},
  "truck": {"en": "truck", "fr": "camion", "de": "LKW"},
  "boat": {"en": "boat", "fr": "bateau", "de": "Boot"},
  "traffic light": {"en": "traffic light", "fr": "feu", "de": "Ampel"},
  "fire hydrant": {"en": "hydrant", "fr": "borne", "de": "Hydrant"},
  "stop sign": {"en": "stop sign", "fr": "stop", "de": "Stoppschild"},
  "parking meter": {"en": "parking meter", "fr": "parcometre", "de": "Parkuhr"},
  "bench": {"en": "bench", "fr": "banc", "de": "Bank"},
  "bird": {"en": "bird", "fr": "oiseau", "de": "Vogel"},
  "cat": {"en": "cat", "fr": "chat", "de": "Katze"},
  "dog": {"en": "dog", "fr": "chien", "de": "Hund"},
  "horse": {"en": "horse", "fr": "cheval", "de": "Pferd"},
  "sheep": {"en": "sheep", "fr": "mouton", "de": "Schaf"},
  "cow": {"en": "cow", "fr": "vache", "de": "Kuh"},
  "elephant": {"en": "elephant", "fr": "elephant", "de": "Elefant"},
  "bear": {"en": "bear", "fr": "ours", "de": "Bar"},
  "zebra": {"en": "zebra", "fr": "zebre", "de": "Zebra"},
  "giraffe": {"en": "giraffe", "fr": "girafe", "de": "Giraffe"},
  "backpack": {"en": "backpack", "fr": "sac", "de": "Rucksack"},
  "umbrella": {"en": "umbrella", "fr": "parapluie", "de": "Schirm"},
  "handbag": {"en": "bag", "fr": "sac", "de": "Tasche"},
  "tie": {"en": "tie", "fr": "cravate", "de": "Krawatte"},
  "suitcase": {"en": "suitcase", "fr": "valise", "de": "Koffer"},
  "frisbee": {"en": "frisbee", "fr": "frisbee", "de": "Frisbee"},
  "skis": {"en": "skis", "fr": "skis", "de": "Ski"},
  "snowboard": {"en": "snowboard", "fr": "snowboard", "de": "Snowboard"},
  "sports ball": {"en": "ball", "fr": "balle", "de": "Ball"},
  "kite": {"en": "kite", "fr": "cerf-volant", "de": "Drachen"},
  "baseball bat": {"en": "bat", "fr": "batte", "de": "Schager"},
  "baseball glove": {"en": "glove", "fr": "gant", "de": "Handschuh"},
  "skateboard": {"en": "skateboard", "fr": "skate", "de": "Skateboard"},
  "surfboard": {"en": "surfboard", "fr": "surf", "de": "Surfbrett"},
  "tennis racket": {"en": "racket", "fr": "raquette", "de": "Schager"},
  "bottle": {"en": "bottle", "fr": "bouteille", "de": "Flasche"},
  "wine glass": {"en": "glass", "fr": "verre", "de": "Glas"},
  "cup": {"en": "cup", "fr": "tasse", "de": "Tasse"},
  "fork": {"en": "fork", "fr": "fourchette", "de": "Gabel"},
  "knife": {"en": "knife", "fr": "couteau", "de": "Messer"},
  "spoon": {"en": "spoon", "fr": "cuillere", "de": "Loffel"},
  "bowl": {"en": "bowl", "fr": "bol", "de": "Schussel"},
  "banana": {"en": "banana", "fr": "banane", "de": "Banane"},
  "apple": {"en": "apple", "fr": "pomme", "de": "Apfel"},
  "sandwich": {"en": "sandwich", "fr": "sandwich", "de": "Sandwich"},
  "orange": {"en": "orange", "fr": "orange", "de": "Orange"},
  "broccoli": {"en": "broccoli", "fr": "brocoli", "de": "Brokkoli"},
  "carrot": {"en": "carrot", "fr": "carotte", "de": "Karotte"},
  "hot dog": {"en": "hot dog", "fr": "hot-dog", "de": "Hotdog"},
  "pizza": {"en": "pizza", "fr": "pizza", "de": "Pizza"},
  "donut": {"en": "donut", "fr": "donut", "de": "Donut"},
  "cake": {"en": "cake", "fr": "gateau", "de": "Kuchen"},
  "chair": {"en": "chair", "fr": "chaise", "de": "Stuhl"},
  "couch": {"en": "couch", "fr": "canape", "de": "Sofa"},
  "potted plant": {"en": "plant", "fr": "plante", "de": "Pflanze"},
  "bed": {"en": "bed", "fr": "lit", "de": "Bett"},
  "dining table": {"en": "table", "fr": "table", "de": "Tisch"},
  "toilet": {"en": "toilet", "fr": "toilettes", "de": "Toilette"},
  "tv": {"en": "TV", "fr": "tele", "de": "Fernseher"},
  "laptop": {"en": "laptop", "fr": "ordinateur", "de": "Laptop"},
  "mouse": {"en": "mouse", "fr": "souris", "de": "Maus"},
  "remote": {"en": "remote", "fr": "telecommande", "de": "Fernbedienung"},
  "keyboard": {"en": "keyboard", "fr": "clavier", "de": "Tastatur"},
  "cell phone": {"en": "phone", "fr": "telephone", "de": "Handy"},
  "microwave": {"en": "microwave", "fr": "micro-ondes", "de": "Mikrowelle"},
  "oven": {"en": "oven", "fr": "four", "de": "Ofen"},
  "toaster": {"en": "toaster", "fr": "grille-pain", "de": "Toaster"},
  "sink": {"en": "sink", "fr": "evier", "de": "Spule"},
  "refrigerator": {"en": "fridge", "fr": "frigo", "de": "Kuhlschrank"},
  "book": {"en": "book", "fr": "livre", "de": "Buch"},
  "clock": {"en": "clock", "fr": "horloge", "de": "Uhr"},
  "vase": {"en": "vase", "fr": "vase", "de": "Vase"},
  "scissors": {"en": "scissors", "fr": "ciseaux", "de": "Schere"},
  "teddy bear": {"en": "teddy", "fr": "ours en peluche", "de": "Teddy"},
  "hair drier": {"en": "hair dryer", "fr": "seche-cheveux", "de": "Fohn"},
  "toothbrush": {"en": "toothbrush", "fr": "brosse a dents", "de": "Zahnbürste"},
}

# French: masculine nouns → "un"; others → "une" (kid-simple).
_FR_MASC = frozenset(
  {
    "velo",
    "avion",
    "bus",
    "train",
    "camion",
    "bateau",
    "feu",
    "stop",
    "banc",
    "oiseau",
    "chat",
    "chien",
    "cheval",
    "mouton",
    "elephant",
    "ours",
    "zebre",
    "sac",
    "parapluie",
    "frisbee",
    "skate",
    "surf",
    "verre",
    "bol",
    "sandwich",
    "brocoli",
    "hot-dog",
    "donut",
    "gateau",
    "canape",
    "lit",
    "vase",
    "ordinateur",
    "telephone",
    "four",
    "frigo",
    "livre",
    "ours en peluche",
  }
)

_VISION_NEED_RE = re.compile(
  r"(?i)(?:"
  r"what\s+do\s+you\s+see|what\s+can\s+you\s+see|do\s+you\s+see|"
  r"what('?s|\s+is)\s+(?:in\s+front|ahead|there|that)|"
  r"look\s+(?:around|at|ahead)|can\s+you\s+look|show\s+me\s+what|"
  r"is\s+there\s+(?:a|an|any)|are\s+there\s+(?:any|a)|"
  r"where\s+is\s+(?:the|a|an)|where\s+are\s+(?:the|my)|"
  r"in\s+front\s+of\s+you|around\s+you|"
  r"qu['’ ]?est[- ]?ce\s+que\s+tu\s+vois|que\s+vois[- ]?tu|"
  r"tu\s+vois|vois[- ]?tu|regarde|devant\s+toi|"
  r"y\s+a[- ]?t[- ]?il|est[- ]?ce\s+qu['’]?\s*il\s+y\s+a|"
  r"ou\s+(?:est|sont)|qu['’]?est[- ]?ce\s+qu['’]?\s*il\s+y\s+a|"
  r"was\s+siehst|was\s+kannst\s+du\s+sehen|siehst\s+du|"
  r"schau(?:e)?\s|vor\s+dir|gibt\s+es|"
  r"wo\s+ist|wo\s+sind"
  r")"
)


def needs_vision_capture(text: str) -> bool:
  """True when the user utterance likely needs a fresh camera capture."""
  t = (text or "").strip()
  if len(t) < 3:
    return False
  return bool(_VISION_NEED_RE.search(t))


def translate_label(label: str, lang: str = "en") -> str:
  """Map English YOLO/COCO label to the spoken language."""
  key = str(label or "thing").replace("_", " ").strip().lower()
  lang = (lang or "en").lower()[:2]
  row = _LABEL_I18N.get(key)
  if row:
    return row.get(lang) or row.get("en") or key
  return key


def looks_like_vision_dump(text: str) -> bool:
  """True if model output is parroting camera/system vision notes."""
  if not text or not text.strip():
    return False
  if _VISION_DUMP_RE.search(text):
    return True
  if '"objects"' in text and ("\"path\"" in text or "side" in text):
    return True
  if text.count("|") >= 2 and ("m to the" in text.lower() or "~" in text):
    return True
  return False


_VISION_QUESTION_RE = re.compile(
  r"(?i)(?:what\s+(?:do\s+you|can\s+you)?\s*see|do\s+you\s+see|"
  r"look\s+around|what.?s\s+(?:in\s+front|there)|"
  r"que\s+(?:vois|voyais)|tu\s+vois|vous\s+voyez|"
  r"qu.?est[- ]ce\s+que\s+tu\s+vois|regarde|"
  r"was\s+siehst|schau\s+mal)"
)
_VISION_FOLLOWUP_RE = re.compile(
  r"(?iu)^\s*(?:"
  r"(?:et\s+)?maintenant|"
  r"(?:and\s+)?now|"
  r"(?:und\s+)?jetzt|"
  r"encore|again|"
  r"et\s+l[aà]"
  r")[\s.!?]*$"
)


def looks_like_vision_question(text: str) -> bool:
  """True when the child asked what is in front of the camera."""
  return bool(_VISION_QUESTION_RE.search(text or ""))


def looks_like_vision_followup(text: str) -> bool:
  """True for a short 'and now?' after a see-question (needs a fresh capture)."""
  return bool(_VISION_FOLLOWUP_RE.match((text or "").strip()))


def looks_like_looking_bridge(text: str) -> bool:
  """True when the spoken line is 'I am looking', not a description of objects."""
  return bool(
    re.search(
      r"(?i)\b(je\s+regarde|i(?:'m|\s+am)\s+looking|ich\s+(?:schaue|gucke|sehe\s+nach))\b",
      text or "",
    )
  )


def should_force_object_glimpse(
  *,
  looked: bool,
  inject_context: bool,
  has_objects: bool,
  mentions_objects: bool,
  vision_dump: bool,
  suppressed_phrases: bool,
  vision_question: bool,
  motion: bool,
) -> bool:
  """Replace a reply with a spoken object list only for look / see turns.

  Motion replies (reverse, follow) must not be overwritten just because
  CameraJSON was injected and the model did not name YOLO labels.
  A looking-bridge ('Je regarde...') plus <<look>> is the intended first
  sentence — do not replace it with a canned YOLO dump.
  """
  if motion:
    return False
  if looked:
    # Spoken line is 'I am looking' + <<look>>; the fresh scene is for the next sentence.
    return False
  if not inject_context:
    return False
  if vision_dump or suppressed_phrases:
    return True
  if has_objects and not mentions_objects and vision_question:
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
    name: str = "vision",
  ) -> None:
    self.broker = broker
    self.port = port
    self.topic = topic
    self.capture_topic = capture_topic
    self.min_interval_s = min_interval_s
    self.capture_timeout_s = capture_timeout_s
    self.capture_view = capture_view if capture_view in ("boxes", "traverse") else "traverse"
    self.name = (name or "vision").strip() or "vision"
    self._lock = threading.Lock()
    self._hint = ""
    self._objects: list[dict[str, Any]] = []
    self._scene: dict[str, Any] = {}
    self._ts = 0.0
    self._client = None
    self._error: Optional[str] = None
    self._scene_event = threading.Event()
    self._inject_context = False
    self._from_look = False
    self._pending_req_id: Optional[str] = None

  def start(self) -> None:
    try:
      import paho.mqtt.client as mqtt
    except ImportError:
      self._error = "paho-mqtt not installed"
      logger.warning(self._error)
      return
    client = mqtt.Client(
      mqtt.CallbackAPIVersion.VERSION2,
      client_id=f"puppet_scene_{self.name}_{os.getpid()}",
    )
    client.on_connect = self._on_connect
    client.on_message = self._on_message
    try:
      client.connect(self.broker, self.port, keepalive=30)
      client.loop_start()
      self._client = client
      logger.info(
        "Vision MQTT [%s] subscribed to %s @ %s:%s (capture=%s)",
        self.name,
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

  def set_inject_context(self, enabled: bool) -> None:
    """Only inject CameraJSON into the LLM when a capture was requested for this turn."""
    self._inject_context = bool(enabled)

  @property
  def inject_context(self) -> bool:
    return self._inject_context

  def _on_connect(self, client, userdata, flags, reason_code, properties=None):
    client.subscribe(self.topic)

  def _on_message(self, client, userdata, msg):
    try:
      payload = json.loads(msg.payload.decode("utf-8"))
    except Exception:
      return
    if not isinstance(payload, dict):
      return
    req = payload.get("req_id")
    with self._lock:
      pending = self._pending_req_id
    # Vision cache is for <<look>> answers. Play/nav frames must not wipe a look
    # or Gemma will be told to look again on ordinary chat.
    if self.name == "vision":
      if pending:
        if req != pending:
          return
      else:
        return
    now = time.time()
    with self._lock:
      self._scene = payload
      self._hint = str(payload.get("hint") or "")
      objs = payload.get("objects") or []
      if isinstance(objs, list):
        self._objects = objs[:8]
      self._ts = now
      self._from_look = False
    self._scene_event.set()

  def apply_scene(
    self, scene: dict[str, Any], *, age_s: float = 0.0, from_look: bool = False
  ) -> None:
    """Update the cache from a scene dict (e.g. copied from the play ingest)."""
    now = time.time() - max(0.0, float(age_s))
    objs = scene.get("objects") or []
    with self._lock:
      self._scene = dict(scene)
      self._hint = str(scene.get("hint") or "")
      if isinstance(objs, list):
        self._objects = objs[:8]
      self._ts = now
      self._from_look = bool(from_look)

  def latest_scene(self) -> dict[str, Any]:
    with self._lock:
      return dict(self._scene)

  def scene_age_s(self) -> float:
    with self._lock:
      if self._ts <= 0:
        return 9999.0
      return time.time() - self._ts

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
    self._scene_event.clear()
    with self._lock:
      self._pending_req_id = req_id
    try:
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
        self._scene_event.wait(timeout=min(0.25, remaining))
        self._scene_event.clear()

      with self._lock:
        scene = dict(self._scene)
      if scene.get("req_id") == req_id:
        return {"ok": True, **scene}
      return {
        "ok": False,
        "error": "capture timeout (is eyes listening on robot/nav/capture?)",
        "req_id": req_id,
      }
    finally:
      with self._lock:
        self._pending_req_id = None

  def from_look(self) -> bool:
    with self._lock:
      return bool(self._from_look)

  def context_line(self) -> str:
    """Compact private camera JSON for the system prompt (not for speaking)."""
    if not self._inject_context:
      return ""
    with self._lock:
      objects = list(self._objects)
      hint = self._hint
      age = time.time() - self._ts
      from_look = self._from_look

    # No CameraJSON on ordinary chat. Gemma looks only if they asked what it sees.
    if (not from_look) or age > 8.0:
      return ""

    # Prefer objects; omit path when objects exist so the LLM does not fixate on floor.
    # Object names stay English (YOLO); the model must translate when speaking.
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

  def object_mention_tokens(self, lang: str = "en") -> list[str]:
    """Tokens that should appear in a good spoken answer about current objects."""
    with self._lock:
      objects = list(self._objects)
    tokens: list[str] = []
    for o in objects:
      label = str(o.get("label") or "").replace("_", " ").lower()
      if not label:
        continue
      tokens.append(label)
      tokens.extend(label.split())
      for code in ("en", "fr", "de"):
        translated = translate_label(label, code)
        tokens.append(translated)
        tokens.extend(translated.replace("-", " ").split())
      # Also accept spoken-lang form specifically
      tokens.append(translate_label(label, lang))
    out: list[str] = []
    for t in tokens:
      t = t.lower().strip()
      if len(t) >= 3 and t not in out:
        out.append(t)
    return out

  def reply_mentions_objects(self, reply: str, lang: str = "en") -> bool:
    if not reply or not self.has_objects():
      return False
    text = reply.lower()
    return any(tok in text for tok in self.object_mention_tokens(lang))

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
      name = translate_label(str(o.get("label") or "thing"), lang)
      side = str(o.get("bearing") or "center")
      side_word = {
        "left": {"en": "on the left", "fr": "a gauche", "de": "links"},
        "right": {"en": "on the right", "fr": "a droite", "de": "rechts"},
        "center": {"en": "in front of me", "fr": "devant moi", "de": "vor mir"},
      }.get(side, {}).get(lang, side)
      parts.append((name, side_word))

    if lang == "fr":
      def _fr_np(name: str, side_word: str) -> str:
        head = name.split()[0]
        article = "un" if name in _FR_MASC or head in _FR_MASC else "une"
        return f"{article} {name} {side_word}"

      np = [_fr_np(n, s) for n, s in parts]
      if len(np) == 1:
        return f"Je vois {np[0]}!"
      return "Je vois " + ", ".join(np[:-1]) + f" et {np[-1]}!"
    if lang == "de":
      spoken = [f"{n} {s}" for n, s in parts]
      if len(spoken) == 1:
        return f"Ich sehe {spoken[0]}!"
      return "Ich sehe " + ", ".join(spoken[:-1]) + f" und {spoken[-1]}!"
    spoken = [f"{n} {s}" for n, s in parts]
    if len(spoken) == 1:
      return f"I see a {spoken[0]}!"
    return "I see " + ", ".join(f"a {p}" for p in spoken[:-1]) + f", and a {spoken[-1]}!"
