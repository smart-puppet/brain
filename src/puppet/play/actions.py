"""Parse hidden robot commands from LLM replies (not from user regex)."""

from __future__ import annotations

import re

_TAG_RE = re.compile(
  r"<<\s*(follow|seek|stop|idle|look|see|capture|"
  r"back|reverse|backward|recule|"
  r"forward|avance|vorwaerts)\s*>>",
  re.IGNORECASE,
)
_LINE_RE = re.compile(
  r"(?im)^\s*(?:ACTION|ACT)\s*:\s*"
  r"(follow|seek|stop|idle|look|see|capture|"
  r"back|reverse|backward|recule|"
  r"forward|avance|vorwaerts)\s*$"
)

_NORM = {
  "follow": "follow",
  "seek": "seek",
  "stop": "idle",
  "idle": "idle",
  "look": "look",
  "see": "look",
  "capture": "look",
  "back": "back",
  "reverse": "back",
  "backward": "back",
  "recule": "back",
  "forward": "forward",
  "avance": "forward",
  "vorwaerts": "forward",
}

# Private robot state is appended to the user turn. q8 KV cache copies it
# faithfully, so strip it from anything Piper might speak.
_PRIVATE_SUFFIX_RE = re.compile(
  r"(?is)(?:\n|\s)*\b(?:Private:\s*)?(?:BodyStatus|CameraJSON)\b.*\Z"
)
_PRIVATE_MOTION_LINE_RE = re.compile(r"(?im)^\s*Motion\s*=\s*(?:on|off)\.?\s*$")
_PRIVATE_PHRASE_RE = re.compile(
  r"(?i)(?:\bBodyStatus\b|\bCameraJSON\b|\bPrivate:\s*|\bMotion\s*=\s*(?:on|off)|"
  r"(?:i(?:'m|\s+am)|you(?:'re|\s+are))\s+(?:already\s+)?standing still|"
  r"^standing still\b)"
)
_STANDING_STILL_SENTENCE_RE = re.compile(
  r"(?is)(?:^|[.!?]\s+)*(?:i(?:'m|\s+am)|you(?:'re|\s+are))\s+"
  r"(?:already\s+)?standing still\b[^.!?]*[.!?]*"
)

# Gemma often EOS after the canned motion sentence and never writes the tag.
# Honor that spoken line from the prompt — not the child's words.
_SPOKEN_FORWARD_RE = re.compile(
  r"(?iu)j['’]avance un peu|roll forward a little|fahre ein stück vor"
)
_SPOKEN_BACK_RE = re.compile(
  r"(?iu)je recule un peu|roll backward a little|fahre ein stück rückwärts"
)


def looks_like_private_state(text: str) -> bool:
  """True if a TTS phrase is parroting BodyStatus / CameraJSON / Motion=."""
  return bool(_PRIVATE_PHRASE_RE.search(text or ""))


def strip_private_robot_state(text: str) -> str:
  """Drop leaked BodyStatus / CameraJSON lines from spoken LLM text."""
  if not text:
    return ""
  spoken = _PRIVATE_SUFFIX_RE.sub("", text)
  spoken = _PRIVATE_MOTION_LINE_RE.sub("", spoken)
  spoken = _STANDING_STILL_SENTENCE_RE.sub("", spoken)
  spoken = re.sub(r"\n{3,}", "\n\n", spoken)
  return spoken.strip()


def _norm(raw: str) -> str:
  return _NORM.get(raw.strip().lower(), "")


def parse_robot_actions(text: str) -> tuple[str, list[str]]:
  """Return (spoken_text, actions) where actions are follow|seek|idle|look|back|forward."""
  if not text:
    return "", []
  actions: list[str] = []
  for match in _TAG_RE.finditer(text):
    name = _norm(match.group(1))
    if name and name not in actions:
      actions.append(name)
  for match in _LINE_RE.finditer(text):
    name = _norm(match.group(1))
    if name and name not in actions:
      actions.append(name)
  spoken = _TAG_RE.sub("", text)
  spoken = _LINE_RE.sub("", spoken)
  spoken = re.sub(r"<<[^>]*>>?", "", spoken)
  spoken = strip_private_robot_state(spoken)
  spoken = re.sub(r"\n{3,}", "\n\n", spoken).strip()
  if "forward" not in actions and "back" not in actions:
    fwd = list(_SPOKEN_FORWARD_RE.finditer(spoken))
    back = list(_SPOKEN_BACK_RE.finditer(spoken))
    if fwd or back:
      last_fwd = fwd[-1].start() if fwd else -1
      last_back = back[-1].start() if back else -1
      actions.append("forward" if last_fwd >= last_back else "back")
  return spoken, actions


def strip_robot_actions(text: str) -> str:
  spoken, _ = parse_robot_actions(text)
  return spoken


# Voice stop while already rolling — not used to *start* play.
_STOP_USER_RE = re.compile(
  r"(?iu)(?<!\w)("
  r"stop+|halt+|freeze|"
  r"stopp+|bleib(?:\s+bitte)?\s+stehen|stehen\s+bleiben|"
  r"h[oö]r\s+auf|"
  r"arr[eê]te(?:z|s)?|reste\s+(?:ici|l[aà])"
  r")(?!\w)"
)


def user_asked_to_stop(text: str) -> bool:
  """Safety brake while already rolling — not used to start play or classify chat."""
  return bool(text and _STOP_USER_RE.search(text.strip()))
