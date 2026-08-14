"""Pure scene → nudge policy for follow / seek (no MQTT, easy to test).

Person follow uses YOLO ``person`` in ``robot/nav/scene``. Obstacle checks
use ``closest_m`` and ``sectors``, but treat the tracked person as *not* an
obstacle (eyes marks people as no-go in the costmap).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class PlayConfig:
  follow_stop_m: float = 0.9
  obstacle_m: float = 0.5
  sector_block_m: float = 0.7
  person_margin_m: float = 0.2
  forward_speed: int = 90
  forward_dur_ms: int = 500
  backward_speed: int = 90
  backward_dur_ms: int = 500
  turn_speed: int = 110
  turn_dur_ms: int = 280
  search_turn_dur_ms: int = 700
  lost_ticks_max: int = 2
  found_m: float = 1.15
  seek_giveup_ticks: int = 24
  doa_deadband_deg: float = 25.0


@dataclass
class PlayMemory:
  lost_ticks: int = 0
  search_dir: str = "turn_left"


@dataclass(frozen=True)
class DriveNudge:
  cmd: str
  speed: int = 0
  dur_ms: int = 0
  reason: str = ""
  person: Optional[dict[str, Any]] = field(default=None, compare=False)


def _finite_m(value: Any) -> Optional[float]:
  if value is None:
    return None
  try:
    dist = float(value)
  except (TypeError, ValueError):
    return None
  if dist != dist or dist < 0:  # NaN
    return None
  return dist


def nearest_person(scene: dict[str, Any]) -> Optional[dict[str, Any]]:
  objects = scene.get("objects") or []
  if not isinstance(objects, list):
    return None
  people: list[tuple[float, dict[str, Any]]] = []
  for obj in objects:
    if not isinstance(obj, dict):
      continue
    if str(obj.get("label") or "").replace("_", " ").strip().lower() != "person":
      continue
    dist = _finite_m(obj.get("dist_m"))
    people.append((dist if dist is not None else 99.0, obj))
  if not people:
    return None
  people.sort(key=lambda item: item[0])
  return people[0][1]


def _freer_turn(sectors: dict[str, Any]) -> str:
  left = _finite_m(sectors.get("left")) or 0.0
  right = _finite_m(sectors.get("right")) or 0.0
  return "turn_left" if left >= right else "turn_right"


def _blocked_ahead_of_person(
  *,
  person_m: Optional[float],
  closest_m: Optional[float],
  center_m: Optional[float],
  cfg: PlayConfig,
) -> bool:
  """True when something other than the person sits in the path."""
  if closest_m is not None and closest_m < cfg.obstacle_m:
    if person_m is None or closest_m + cfg.person_margin_m < person_m:
      return True
  if center_m is not None and center_m < cfg.sector_block_m:
    if person_m is None or center_m + cfg.person_margin_m < person_m:
      return True
  return False


def plan_follow(
  scene: dict[str, Any],
  mem: PlayMemory,
  cfg: PlayConfig,
  *,
  heading_error_deg: Optional[float] = None,
) -> DriveNudge:
  person = nearest_person(scene)
  sectors = scene.get("sectors") if isinstance(scene.get("sectors"), dict) else {}
  closest_m = _finite_m(scene.get("closest_m"))
  center_m = _finite_m(sectors.get("center"))

  if person is None:
    return _plan_lost(scene, mem, cfg, heading_error_deg=heading_error_deg)

  mem.lost_ticks = 0
  person_m = _finite_m(person.get("dist_m"))
  bearing = str(person.get("bearing") or "center").lower()

  if _blocked_ahead_of_person(
    person_m=person_m, closest_m=closest_m, center_m=center_m, cfg=cfg
  ):
    turn = _freer_turn(sectors)
    return DriveNudge(
      turn,
      speed=cfg.turn_speed,
      dur_ms=cfg.turn_dur_ms,
      reason="avoid",
      person=person,
    )

  if person_m is not None and person_m <= cfg.follow_stop_m:
    return DriveNudge("idle", reason="close", person=person)

  if bearing == "left":
    return DriveNudge(
      "turn_left",
      speed=cfg.turn_speed,
      dur_ms=cfg.turn_dur_ms,
      reason="turn_to_person",
      person=person,
    )
  if bearing == "right":
    return DriveNudge(
      "turn_right",
      speed=cfg.turn_speed,
      dur_ms=cfg.turn_dur_ms,
      reason="turn_to_person",
      person=person,
    )

  return DriveNudge(
    "forward",
    speed=cfg.forward_speed,
    dur_ms=cfg.forward_dur_ms,
    reason="approach",
    person=person,
  )


def _plan_lost(
  scene: dict[str, Any],
  mem: PlayMemory,
  cfg: PlayConfig,
  *,
  heading_error_deg: Optional[float],
) -> DriveNudge:
  """No YOLO person: turn toward last voice DoA, or creep forward if facing them."""
  mem.lost_ticks += 1
  sectors = scene.get("sectors") if isinstance(scene.get("sectors"), dict) else {}
  closest_m = _finite_m(scene.get("closest_m"))
  center_m = _finite_m(sectors.get("center"))
  if heading_error_deg is not None and abs(heading_error_deg) >= cfg.doa_deadband_deg:
    cmd = "turn_right" if heading_error_deg > 0 else "turn_left"
    span = min(abs(heading_error_deg), 90.0) / 90.0
    dur = max(250, int(cfg.search_turn_dur_ms * span))
    return DriveNudge(cmd, speed=cfg.turn_speed, dur_ms=dur, reason="turn_to_voice")
  blocked = _blocked_ahead_of_person(
    person_m=None, closest_m=closest_m, center_m=center_m, cfg=cfg
  )
  facing = heading_error_deg is not None and abs(heading_error_deg) < cfg.doa_deadband_deg
  if facing and not blocked and (closest_m is None or closest_m > cfg.follow_stop_m):
    return DriveNudge(
      "forward",
      speed=cfg.forward_speed,
      dur_ms=cfg.forward_dur_ms,
      reason="approach_voice",
    )
  if mem.lost_ticks >= cfg.lost_ticks_max and heading_error_deg is None:
    mem.search_dir = "turn_right" if mem.search_dir == "turn_left" else "turn_left"
    return DriveNudge(
      mem.search_dir,
      speed=cfg.turn_speed,
      dur_ms=cfg.search_turn_dur_ms,
      reason="search",
    )
  return DriveNudge("idle", reason="lost")


def plan_seek(
  scene: dict[str, Any],
  mem: PlayMemory,
  cfg: PlayConfig,
  *,
  heading_error_deg: Optional[float] = None,
) -> DriveNudge:
  """Look around until a person is seen, then follow until close (found).

  While lost, only turn in place — never roll toward the last voice. Give up
  after ``seek_giveup_ticks`` so the game cannot run forever.
  """
  person = nearest_person(scene)
  if person is not None:
    person_m = _finite_m(person.get("dist_m"))
    if person_m is not None and person_m <= cfg.found_m:
      mem.lost_ticks = 0
      return DriveNudge("idle", reason="found", person=person)
    return plan_follow(scene, mem, cfg, heading_error_deg=heading_error_deg)
  return _plan_seek_lost(mem, cfg, heading_error_deg=heading_error_deg)


def _plan_seek_lost(
  mem: PlayMemory,
  cfg: PlayConfig,
  *,
  heading_error_deg: Optional[float],
) -> DriveNudge:
  mem.lost_ticks += 1
  if mem.lost_ticks >= max(1, cfg.seek_giveup_ticks):
    return DriveNudge("idle", reason="giveup")
  if heading_error_deg is not None and abs(heading_error_deg) >= cfg.doa_deadband_deg:
    cmd = "turn_right" if heading_error_deg > 0 else "turn_left"
    span = min(abs(heading_error_deg), 90.0) / 90.0
    dur = max(250, int(cfg.search_turn_dur_ms * span))
    return DriveNudge(cmd, speed=cfg.turn_speed, dur_ms=dur, reason="turn_to_voice")
  if mem.lost_ticks % 2 == 1:
    mem.search_dir = "turn_right" if mem.search_dir == "turn_left" else "turn_left"
    return DriveNudge(
      mem.search_dir,
      speed=cfg.turn_speed,
      dur_ms=cfg.search_turn_dur_ms,
      reason="search",
    )
  return DriveNudge("idle", reason="lost")


def plan(
  mode: str,
  scene: dict[str, Any],
  mem: PlayMemory,
  cfg: PlayConfig,
  *,
  heading_error_deg: Optional[float] = None,
) -> DriveNudge:
  if mode == "follow":
    return plan_follow(scene, mem, cfg, heading_error_deg=heading_error_deg)
  if mode == "seek":
    return plan_seek(scene, mem, cfg, heading_error_deg=heading_error_deg)
  return DriveNudge("idle", reason="idle")
