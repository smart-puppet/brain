"""Parse hidden robot commands from LLM replies (not from user regex)."""

from __future__ import annotations

import re

_TAG_RE = re.compile(
  r"<<\s*(follow|seek|stop|idle|look|see|capture|back|reverse|backward|recule)\s*>>",
  re.IGNORECASE,
)
_LINE_RE = re.compile(
  r"(?im)^\s*(?:ACTION|ACT)\s*:\s*"
  r"(follow|seek|stop|idle|look|see|capture|back|reverse|backward|recule)\s*$"
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
}


def _norm(raw: str) -> str:
  return _NORM.get(raw.strip().lower(), "")


def parse_robot_actions(text: str) -> tuple[str, list[str]]:
  """Return (spoken_text, actions) where actions are follow|seek|idle|look|back."""
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
  spoken = re.sub(r"\n{3,}", "\n\n", spoken).strip()
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
  """True when the child clearly asked to stay still (any of en/fr/de)."""
  return bool(text and _STOP_USER_RE.search(text.strip()))


_FOLLOW_USER_RE = re.compile(
  r"(?iu)("
  r"follow\s+me|come\s+(?:here|with(?:\s+me)?|closer)|"
  r"suis[- ]moi|viens[- ](?:ici|avec(?:\s+moi)?)|(?<!\w)viens\s*[!.?]|"
  r"folge\s+mir|komm\s+(?:mit|her)"
  r")"
)
_SEEK_USER_RE = re.compile(
  r"(?iu)("
  r"hide[- ]?and[- ]?seek|"
  r"cache[- ]?cache|"
  r"verstecken"
  r")"
)


def user_asked_to_follow(text: str) -> bool:
  """True when the child asked the robot to come/follow (not merely mentioned play)."""
  return bool(text and _FOLLOW_USER_RE.search(text.strip()))


def user_asked_to_seek(text: str) -> bool:
  """True when the child asked to play hide-and-seek."""
  return bool(text and _SEEK_USER_RE.search(text.strip()))
