"""Live play speeds (follow turn / seek turn / forward) from Eye or MQTT."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

SPEED_MIN = 20
SPEED_MAX = 200
SPEEDS_FILE = "play.speeds"
SPEED_KEYS = ("follow_turn", "seek_turn", "forward")

DEFAULT_SPEEDS = {
  "follow_turn": 125,
  "seek_turn": 125,
  "forward": 105,
}


def clamp_speed(value: Any, default: int) -> int:
  try:
    speed = int(value)
  except (TypeError, ValueError):
    return int(default)
  return max(SPEED_MIN, min(SPEED_MAX, speed))


def normalize_speeds(data: Any, defaults: dict[str, int] | None = None) -> dict[str, int]:
  base = dict(DEFAULT_SPEEDS)
  if defaults:
    for key in SPEED_KEYS:
      if key in defaults:
        base[key] = clamp_speed(defaults[key], base[key])
  src = data if isinstance(data, dict) else {}
  return {
    "follow_turn": clamp_speed(src.get("follow_turn"), base["follow_turn"]),
    "seek_turn": clamp_speed(src.get("seek_turn"), base["seek_turn"]),
    "forward": clamp_speed(src.get("forward"), base["forward"]),
  }


def play_speeds_path(config_dir: str | Path) -> Path:
  return Path(config_dir) / SPEEDS_FILE


def read_play_speeds(
  config_dir: str | Path,
  defaults: dict[str, int] | None = None,
) -> dict[str, int]:
  path = play_speeds_path(config_dir)
  if not path.is_file():
    return normalize_speeds({}, defaults)
  try:
    raw = json.loads(path.read_text(encoding="utf-8"))
  except (OSError, json.JSONDecodeError):
    return normalize_speeds({}, defaults)
  return normalize_speeds(raw, defaults)


def write_play_speeds(config_dir: str | Path, speeds: dict[str, Any]) -> tuple[Path, dict[str, int]]:
  config_path = Path(config_dir)
  config_path.mkdir(parents=True, exist_ok=True)
  normalized = normalize_speeds(speeds)
  path = play_speeds_path(config_path)
  tmp = path.with_name(path.name + ".tmp")
  tmp.write_text(json.dumps(normalized) + "\n", encoding="utf-8")
  tmp.replace(path)
  return path, normalized


def resolve_play_speeds(follow: dict[str, Any], config_dir: str | Path) -> dict[str, int]:
  turn = int(follow.get("turn_speed", DEFAULT_SPEEDS["follow_turn"]))
  defaults = {
    "follow_turn": turn,
    "seek_turn": int(follow.get("seek_turn_speed", turn)),
    "forward": int(follow.get("forward_speed", DEFAULT_SPEEDS["forward"])),
  }
  return read_play_speeds(config_dir, defaults)
