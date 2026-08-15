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
  forward_speed: int = 105
  forward_dur_ms: int = 500
  backward_speed: int = 105
  backward_dur_ms: int = 500
  turn_speed: int = 125
  turn_dur_ms: int = 280
  search_turn_dur_ms: int = 1800
  search_turn_ticks: int = 4
  search_forward_ticks: int = 4
  search_forward_dur_ms: int = 1200
  lost_ticks_max: int = 2
  found_m: float = 1.15
  seek_giveup_ticks: int = 40
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


def _flip_search_dir(mem: PlayMemory) -> None:
  mem.search_dir = "turn_right" if mem.search_dir == "turn_left" else "turn_left"


def _search_turn(mem: PlayMemory, cfg: PlayConfig, *, reason: str) -> DriveNudge:
  return DriveNudge(
    mem.search_dir,
    speed=cfg.turn_speed,
    dur_ms=cfg.search_turn_dur_ms,
    reason=reason,
  )


def _plan_lost(
  scene: dict[str, Any],
  mem: PlayMemory,
  cfg: PlayConfig,
  *,
  heading_error_deg: Optional[float],
) -> DriveNudge:
  """No YOLO person: wait a tick or two for flicker, then sweep one way.

  Do not chase last-voice DoA — it freezes after speech and spins forever.
  """
  del scene, heading_error_deg
  mem.lost_ticks += 1
  if mem.lost_ticks <= max(0, cfg.lost_ticks_max):
    return DriveNudge("idle", reason="lost")
  turn_n = max(1, cfg.search_turn_ticks)
  scan_index = mem.lost_ticks - max(0, cfg.lost_ticks_max) - 1
  if scan_index > 0 and scan_index % turn_n == 0:
    _flip_search_dir(mem)
  return _search_turn(mem, cfg, reason="scan")


def plan_seek(
  scene: dict[str, Any],
  mem: PlayMemory,
  cfg: PlayConfig,
  *,
  heading_error_deg: Optional[float] = None,
) -> DriveNudge:
  """Wander the room until a person is seen, then follow until close (found).

  While lost, roll forward when the path is clear and turn to look around —
  do not spin in place or chase last-voice DoA. Give up after
  ``seek_giveup_ticks`` so the game cannot run forever.
  """
  person = nearest_person(scene)
  if person is not None:
    person_m = _finite_m(person.get("dist_m"))
    if person_m is not None and person_m <= cfg.found_m:
      mem.lost_ticks = 0
      return DriveNudge("idle", reason="found", person=person)
    return plan_follow(scene, mem, cfg, heading_error_deg=heading_error_deg)
  return _plan_seek_lost(scene, mem, cfg)


def _plan_seek_lost(
  scene: dict[str, Any],
  mem: PlayMemory,
  cfg: PlayConfig,
) -> DriveNudge:
  mem.lost_ticks += 1
  if mem.lost_ticks >= max(1, cfg.seek_giveup_ticks):
    return DriveNudge("idle", reason="giveup")
  sectors = scene.get("sectors") if isinstance(scene.get("sectors"), dict) else {}
  closest_m = _finite_m(scene.get("closest_m"))
  center_m = _finite_m(sectors.get("center"))
  blocked = _blocked_ahead_of_person(
    person_m=None, closest_m=closest_m, center_m=center_m, cfg=cfg
  )
  if blocked:
    mem.search_dir = _freer_turn(sectors)
    return _search_turn(mem, cfg, reason="search")
  # Sweep one heading, then roll several times into the room, then the other way.
  turn_n = max(1, cfg.search_turn_ticks)
  fwd_n = max(1, cfg.search_forward_ticks)
  cycle = turn_n + fwd_n
  phase = (mem.lost_ticks - 1) % cycle
  if phase == 0 and mem.lost_ticks > 1:
    _flip_search_dir(mem)
  if phase < turn_n:
    return _search_turn(mem, cfg, reason="search")
  return DriveNudge(
    "forward",
    speed=cfg.forward_speed,
    dur_ms=cfg.search_forward_dur_ms,
    reason="search",
  )


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
