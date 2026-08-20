"""Pure scene → nudge policy for follow / seek (no MQTT, easy to test).

Person follow uses YOLO ``person`` in ``robot/nav/scene``. Obstacle checks
use ``closest_m`` and ``sectors``, but treat the tracked person as *not* an
obstacle (eyes marks people as no-go in the costmap).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Optional

from puppet.play.explore import OccupancyMap


@dataclass
class PlayConfig:
  follow_stop_m: float = 1.5
  obstacle_m: float = 0.8
  sector_block_m: float = 0.7
  person_margin_m: float = 0.2
  floor_block_pct: float = 0.12
  forward_speed: int = 105
  forward_dur_ms: int = 500
  backward_speed: int = 105
  backward_dur_ms: int = 500
  turn_speed: int = 125
  seek_turn_speed: int = 125
  turn_dur_ms: int = 280
  search_turn_dur_ms: int = 500
  search_turn_ticks: int = 1
  search_forward_ticks: int = 12
  search_forward_dur_ms: int = 1500
  lost_ticks_max: int = 2
  found_m: float = 2.0
  seek_giveup_ticks: int = 240
  seek_giveup_s: float = 180.0
  seek_map: bool = True
  forward_m_per_s: float = 0.30
  map_res_m: float = 0.1
  map_size_m: float = 12.0
  turn_ms_per_deg: int = 8
  follow_spin_deg: int = 360
  follow_recover_deg: int = 180
  seek_face_deg: int = 90
  doa_deadband_deg: float = 25.0
  # 0 = mechanical (tests). YAML default ~0.35 = livelier wander; speeds stay as set on Eye.
  alive_jitter: float = 0.0
  unstick_after: int = 2
  # After this many reverses without escaping, commit to a U-turn toward the freer side.
  uturn_after: int = 3
  uturn_ticks: int = 2
  uturn_dur_ms: int = 900
  # Jittered pulses shorter than this feel like twitching instead of rolling.
  min_pulse_ms: int = 320


@dataclass
class PlayMemory:
  lost_ticks: int = 0
  search_dir: str = "turn_left"
  cycle_turn_n: int = 0
  cycle_fwd_n: int = 0
  stuck_ticks: int = 0
  last_stuck: str = ""
  wander_i: int = 0
  retreats: int = 0
  uturn_left: int = 0
  committed_dir: str = ""
  had_person: bool = False
  last_person_side: str = ""
  search_phase: str = ""
  spun_ms: int = 0
  last_roll_dir: str = ""
  face_target_ms: int = 0
  had_scan: bool = False
  explore: Any = field(default=None, repr=False)
  rng: Any = field(default=None, repr=False, compare=False)


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


def _rng(mem: PlayMemory) -> random.Random:
  if mem.rng is None:
    mem.rng = random.Random()
  return mem.rng


def _dur(cfg: PlayConfig, base_ms: int, mem: PlayMemory) -> int:
  """Jitter pulse length, not wheel speed (Eye sliders stay in charge)."""
  base = max(80, int(base_ms))
  if cfg.alive_jitter <= 0:
    return base
  lo = max(0.55, 1.0 - cfg.alive_jitter)
  hi = 1.0 + min(0.55, cfg.alive_jitter)
  return max(max(80, cfg.min_pulse_ms), int(base * _rng(mem).uniform(lo, hi)))


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


def _freer_turn(
  sectors: dict[str, Any],
  mem: Optional[PlayMemory] = None,
  *,
  jitter: float = 0.0,
) -> str:
  left = _finite_m(sectors.get("left")) or 0.0
  right = _finite_m(sectors.get("right")) or 0.0
  if abs(left - right) < 0.25 and mem is not None and jitter > 0:
    return "turn_left" if _rng(mem).random() < 0.5 else "turn_right"
  return "turn_left" if left >= right else "turn_right"


def _blocked_ahead_of_person(
  *,
  person_m: Optional[float],
  closest_m: Optional[float],
  center_m: Optional[float],
  cfg: PlayConfig,
  floor_ahead_pct: Optional[float] = None,
) -> bool:
  """True when something other than the person sits in the path."""
  if closest_m is not None and closest_m < cfg.obstacle_m:
    if person_m is None or closest_m + cfg.person_margin_m < person_m:
      return True
  if center_m is not None and center_m < cfg.sector_block_m:
    if person_m is None or center_m + cfg.person_margin_m < person_m:
      return True
  if (
    floor_ahead_pct is not None
    and floor_ahead_pct < cfg.floor_block_pct
    and (person_m is None or person_m > cfg.follow_stop_m + 0.4)
  ):
    return True
  return False


def _clear_stuck(mem: PlayMemory) -> None:
  mem.stuck_ticks = 0
  mem.last_stuck = ""
  mem.retreats = 0
  mem.uturn_left = 0
  mem.committed_dir = ""


def _escaped(
  *,
  closest_m: Optional[float],
  floor_ahead_pct: Optional[float],
  cfg: PlayConfig,
) -> bool:
  """True when the view is open enough to leave a corner, not a 1 m twitch-gap."""
  far = closest_m is None or closest_m >= cfg.obstacle_m + 0.45
  open_floor = floor_ahead_pct is None or floor_ahead_pct >= 0.28
  return far and open_floor


def _still_trapped(
  mem: PlayMemory,
  cfg: PlayConfig,
  *,
  blocked: bool,
  closest_m: Optional[float],
  floor_ahead_pct: Optional[float],
) -> bool:
  if _escaped(
    closest_m=closest_m, floor_ahead_pct=floor_ahead_pct, cfg=cfg
  ):
    return False
  if mem.uturn_left > 0:
    return True
  if blocked:
    return True
  return False


def _commit_turn(
  mem: PlayMemory,
  sectors: dict[str, Any],
  *,
  jitter: float = 0.0,
) -> str:
  """Pick a side once and keep it until the corner is actually left."""
  if mem.committed_dir in ("turn_left", "turn_right"):
    mem.search_dir = mem.committed_dir
    return mem.search_dir
  mem.search_dir = _freer_turn(sectors, mem, jitter=jitter)
  mem.committed_dir = mem.search_dir
  return mem.search_dir


def _both_sides_tight(sectors: dict[str, Any], cfg: PlayConfig) -> bool:
  lim = max(cfg.sector_block_m, cfg.obstacle_m)
  left = _finite_m(sectors.get("left"))
  right = _finite_m(sectors.get("right"))
  # Missing range is unknown, not a wall — wood-floor NaNs used to reverse forever.
  return (left is not None and left < lim) and (right is not None and right < lim)


def _unstick(
  mem: PlayMemory,
  cfg: PlayConfig,
  sectors: dict[str, Any],
  *,
  person: Optional[dict[str, Any]] = None,
  closest_m: Optional[float] = None,
  turn_reason: str = "avoid",
) -> DriveNudge:
  """Simple recovery: reverse, then re-capture and re-plan on next tick."""
  del sectors, closest_m, turn_reason
  mem.stuck_ticks += 1
  mem.last_stuck = "back"
  mem.retreats += 1
  mem.uturn_left = 0
  mem.committed_dir = ""
  return DriveNudge(
    "backward",
    speed=cfg.backward_speed,
    dur_ms=_dur(cfg, cfg.backward_dur_ms, mem),
    reason="unstick",
    person=person,
  )


def _floor_pct(scene: dict[str, Any]) -> Optional[float]:
  value = scene.get("floor_ahead_pct")
  if isinstance(value, (int, float)):
    return float(value)
  return None


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
    # Close legs often drop the box while the child is still in front.
    # If they walked off left/right, turn that way — do not stand still.
    exited = mem.last_person_side in ("left", "right")
    if (
      mem.had_person
      and not exited
      and closest_m is not None
      and closest_m <= cfg.follow_stop_m
    ):
      return DriveNudge("idle", reason="close")
    return _plan_lost(scene, mem, cfg, heading_error_deg=heading_error_deg)

  mem.lost_ticks = 0
  mem.wander_i = 0
  mem.had_person = True
  mem.search_phase = ""
  mem.spun_ms = 0
  person_m = _finite_m(person.get("dist_m"))
  bearing = str(person.get("bearing") or "center").lower()
  if bearing in ("left", "right"):
    mem.last_person_side = bearing
  elif not mem.last_person_side:
    mem.last_person_side = "center"
  _clear_stuck(mem)

  if person_m is not None and person_m <= cfg.follow_stop_m:
    return DriveNudge("idle", reason="close", person=person)

  if bearing == "left":
    return DriveNudge(
      "turn_left",
      speed=cfg.turn_speed,
      dur_ms=_dur(cfg, cfg.turn_dur_ms, mem),
      reason="turn_to_person",
      person=person,
    )
  if bearing == "right":
    return DriveNudge(
      "turn_right",
      speed=cfg.turn_speed,
      dur_ms=_dur(cfg, cfg.turn_dur_ms, mem),
      reason="turn_to_person",
      person=person,
    )

  if cfg.alive_jitter > 0 and _rng(mem).random() < min(0.2, cfg.alive_jitter * 0.55):
    peek = _freer_turn(sectors, mem, jitter=cfg.alive_jitter)
    return DriveNudge(
      peek,
      speed=cfg.turn_speed,
      dur_ms=_dur(cfg, max(120, int(cfg.turn_dur_ms * 0.55)), mem),
      reason="wiggle",
      person=person,
    )

  return DriveNudge(
    "forward",
    speed=cfg.forward_speed,
    dur_ms=_dur(cfg, cfg.forward_dur_ms, mem),
    reason="approach",
    person=person,
  )


def _spin_target_ms(cfg: PlayConfig, deg: int) -> int:
  return max(1, int(deg) * max(1, int(cfg.turn_ms_per_deg)))


def _begin_follow_search(mem: PlayMemory, cfg: PlayConfig) -> None:
  if mem.search_phase:
    return
  if mem.had_person and mem.last_person_side in ("left", "right"):
    mem.search_phase = "recover"
    mem.search_dir = "turn_left" if mem.last_person_side == "left" else "turn_right"
  else:
    mem.search_phase = "spin"
    if mem.search_dir not in ("turn_left", "turn_right"):
      mem.search_dir = "turn_left"
  mem.spun_ms = 0
  _clear_stuck(mem)


def _plan_lost(
  scene: dict[str, Any],
  mem: PlayMemory,
  cfg: PlayConfig,
  *,
  heading_error_deg: Optional[float],
) -> DriveNudge:
  """No YOLO person: spin in place until someone appears, then give up.

  Start / no last side: one full turn. Person walked off left/right: keep
  turning that way until they are back or 180°, then the same full-turn look.
  Do not chase last-voice DoA.
  """
  del scene, heading_error_deg
  exited = mem.last_person_side in ("left", "right")
  if mem.had_person and not exited:
    mem.lost_ticks += 1
    if mem.lost_ticks <= max(0, cfg.lost_ticks_max):
      return DriveNudge("idle", reason="lost")
  _begin_follow_search(mem, cfg)
  recover_ms = _spin_target_ms(cfg, cfg.follow_recover_deg)
  spin_ms = _spin_target_ms(cfg, cfg.follow_spin_deg)
  if mem.search_phase == "recover" and mem.spun_ms >= recover_ms:
    mem.search_phase = "spin"
    mem.spun_ms = 0
  if mem.search_phase == "spin" and mem.spun_ms >= spin_ms:
    return DriveNudge("idle", reason="nofollow")
  target_ms = recover_ms if mem.search_phase == "recover" else spin_ms
  remaining = max(80, target_ms - mem.spun_ms)
  dur = min(max(80, int(cfg.search_turn_dur_ms)), remaining)
  mem.spun_ms += dur
  return DriveNudge(
    mem.search_dir,
    speed=cfg.turn_speed,
    dur_ms=dur,
    reason="recover" if mem.search_phase == "recover" else "scan",
  )


def _seek_next_dir(mem: PlayMemory, sectors: dict[str, Any], cfg: PlayConfig) -> str:
  """After a station look, roll toward the freer side — prefer not retracing."""
  left_m = _finite_m(sectors.get("left")) or 0.0
  right_m = _finite_m(sectors.get("right")) or 0.0
  freer = "turn_left" if left_m >= right_m else "turn_right"
  other = "turn_right" if freer == "turn_left" else "turn_left"
  other_m = right_m if other == "turn_right" else left_m
  if mem.last_roll_dir == freer and other_m >= cfg.sector_block_m:
    return other
  if abs(left_m - right_m) < 0.25 and mem.last_roll_dir in ("turn_left", "turn_right"):
    return other
  return freer


def _begin_seek_search(mem: PlayMemory) -> None:
  if mem.search_phase:
    return
  if mem.had_person and mem.last_person_side in ("left", "right"):
    mem.search_phase = "recover"
    mem.search_dir = "turn_left" if mem.last_person_side == "left" else "turn_right"
  else:
    mem.search_phase = "scan"
    if mem.search_dir not in ("turn_left", "turn_right"):
      mem.search_dir = "turn_left"
  mem.spun_ms = 0
  mem.wander_i = 0


def _seek_map(mem: PlayMemory, cfg: PlayConfig) -> OccupancyMap:
  if mem.explore is None:
    mem.explore = OccupancyMap(res_m=cfg.map_res_m, size_m=cfg.map_size_m)
  return mem.explore


def _commit_seek(mem: PlayMemory, cfg: PlayConfig, nudge: DriveNudge) -> DriveNudge:
  if cfg.seek_map and nudge.cmd != "idle" and nudge.dur_ms > 0:
    _seek_map(mem, cfg).apply_nudge(nudge, cfg)
  return nudge


def _seek_should_giveup(mem: PlayMemory, cfg: PlayConfig) -> bool:
  if mem.lost_ticks >= max(1, cfg.seek_giveup_ticks):
    return True
  if cfg.seek_giveup_s > 0 and mem.explore is not None:
    return mem.explore.elapsed_s() >= cfg.seek_giveup_s
  return False


def _pick_seek_heading(
  mem: PlayMemory,
  sectors: dict[str, Any],
  cfg: PlayConfig,
  *,
  blocked: bool = False,
) -> tuple[str, int, bool]:
  bearing = None
  if cfg.seek_map and mem.explore is not None:
    bearing = mem.explore.frontier_bearing_deg()
    if bearing is not None and blocked and abs(bearing) <= 40.0:
      bearing = None
  if bearing is not None:
    direction = "turn_left" if bearing > 0 else "turn_right"
    face_ms = max(80, int(min(180.0, abs(bearing)) * max(1, int(cfg.turn_ms_per_deg))))
    return direction, face_ms, abs(bearing) <= 50.0
  direction = _seek_next_dir(mem, sectors, cfg)
  if int(cfg.seek_face_deg) <= 0:
    return direction, 80, True
  return direction, _spin_target_ms(cfg, int(cfg.seek_face_deg)), False


def _seek_boxed_in(
  *,
  closest_m: Optional[float],
  floor_ahead_pct: Optional[float],
  blocked: bool,
) -> bool:
  if blocked:
    return True
  if closest_m is not None and closest_m < 0.9:
    return True
  if floor_ahead_pct is not None and floor_ahead_pct < 0.22:
    return True
  return False


def _seek_spin_nudge(mem: PlayMemory, cfg: PlayConfig, *, target_ms: int, reason: str) -> DriveNudge:
  remaining = max(80, target_ms - mem.spun_ms)
  dur = min(max(80, int(cfg.search_turn_dur_ms)), remaining)
  mem.spun_ms += dur
  return DriveNudge(
    mem.search_dir,
    speed=cfg.seek_turn_speed,
    dur_ms=dur,
    reason=reason,
  )


def plan_seek(
  scene: dict[str, Any],
  mem: PlayMemory,
  cfg: PlayConfig,
  *,
  heading_error_deg: Optional[float] = None,
) -> DriveNudge:
  """Look around the room until a person is seen, then follow until close (found).

  Lost: recover toward the last exit side, then one 360° look, face the next
  heading, and cruise straight until boxed in. Escape with reverse plus a ~90°
  turn, then cruise again. Do not chase last-voice DoA. Give up after
  ``seek_giveup_s`` or ``seek_giveup_ticks``.
  """
  person = nearest_person(scene)
  if person is not None:
    person_m = _finite_m(person.get("dist_m"))
    if person_m is not None and person_m <= cfg.found_m:
      mem.lost_ticks = 0
      mem.wander_i = 0
      mem.search_phase = ""
      mem.spun_ms = 0
      return DriveNudge("idle", reason="found", person=person)
    return _commit_seek(
      mem, cfg, plan_follow(scene, mem, cfg, heading_error_deg=heading_error_deg)
    )
  return _plan_seek_lost(scene, mem, cfg)


def _plan_seek_lost(
  scene: dict[str, Any],
  mem: PlayMemory,
  cfg: PlayConfig,
) -> DriveNudge:
  mem.lost_ticks += 1
  if cfg.seek_map:
    _seek_map(mem, cfg).integrate(scene)
  if _seek_should_giveup(mem, cfg):
    return DriveNudge("idle", reason="giveup")
  exited = mem.last_person_side in ("left", "right")
  if mem.had_person and not exited and not mem.search_phase:
    if mem.lost_ticks <= max(0, cfg.lost_ticks_max):
      return DriveNudge("idle", reason="lost")

  sectors = scene.get("sectors") if isinstance(scene.get("sectors"), dict) else {}
  closest_m = _finite_m(scene.get("closest_m"))
  center_m = _finite_m(sectors.get("center"))
  floor_ahead_pct = _floor_pct(scene)
  blocked = _blocked_ahead_of_person(
    person_m=None,
    closest_m=closest_m,
    center_m=center_m,
    cfg=cfg,
    floor_ahead_pct=floor_ahead_pct,
  )
  boxed = _seek_boxed_in(
    closest_m=closest_m, floor_ahead_pct=floor_ahead_pct, blocked=blocked
  )
  if boxed and mem.search_phase in ("", "scan", "face", "roll"):
    mem.search_phase = "escape"
    mem.wander_i = 0
    mem.spun_ms = 0
  if mem.search_phase == "escape":
    if mem.wander_i == 0:
      mem.wander_i = 1
      return _commit_seek(
        mem,
        cfg,
        DriveNudge(
          "backward",
          speed=cfg.backward_speed,
          dur_ms=_dur(cfg, cfg.backward_dur_ms, mem),
          reason="search",
        ),
      )
    mem.search_dir = _freer_turn(sectors, mem, jitter=cfg.alive_jitter)
    mem.last_roll_dir = mem.search_dir
    mem.search_phase = "roll"
    mem.had_scan = True
    mem.spun_ms = 0
    mem.wander_i = 0
    return _commit_seek(
      mem, cfg, _seek_spin_nudge(mem, cfg, target_ms=_spin_target_ms(cfg, 90), reason="search")
    )

  _begin_seek_search(mem)
  recover_ms = _spin_target_ms(cfg, cfg.follow_recover_deg)
  scan_ms = _spin_target_ms(cfg, cfg.follow_spin_deg)
  face_ms = mem.face_target_ms if mem.face_target_ms > 0 else _spin_target_ms(cfg, max(0, int(cfg.seek_face_deg)))

  if mem.search_phase == "recover" and mem.spun_ms >= recover_ms:
    mem.search_phase = "roll"
    mem.had_scan = True
    mem.wander_i = 0
    mem.spun_ms = 0
  if mem.search_phase == "scan" and mem.spun_ms >= scan_ms:
    mem.had_scan = True
    if cfg.seek_map and mem.explore is not None:
      mem.explore.mark_station()
    direction, face_ms, skip_face = _pick_seek_heading(mem, sectors, cfg, blocked=blocked)
    mem.search_dir = direction
    mem.last_roll_dir = direction
    mem.face_target_ms = face_ms
    mem.spun_ms = 0
    if skip_face:
      mem.search_phase = "roll"
      mem.wander_i = 0
    else:
      mem.search_phase = "face"
  if mem.search_phase == "face" and mem.spun_ms >= (mem.face_target_ms or face_ms):
    mem.search_phase = "roll"
    mem.wander_i = 0
  if mem.search_phase == "roll":
    if boxed:
      mem.search_phase = "escape"
      mem.wander_i = 0
      mem.spun_ms = 0
      return _commit_seek(
        mem,
        cfg,
        DriveNudge(
          "backward",
          speed=cfg.backward_speed,
          dur_ms=_dur(cfg, cfg.backward_dur_ms, mem),
          reason="search",
        ),
      )
    mem.wander_i += 1
    return _commit_seek(
      mem,
      cfg,
      DriveNudge(
        "forward",
        speed=cfg.forward_speed,
        dur_ms=_dur(cfg, cfg.search_forward_dur_ms, mem),
        reason="search",
      ),
    )

  if mem.search_phase == "recover":
    return _commit_seek(mem, cfg, _seek_spin_nudge(mem, cfg, target_ms=recover_ms, reason="recover"))
  if mem.search_phase == "face":
    return _commit_seek(
      mem, cfg, _seek_spin_nudge(mem, cfg, target_ms=mem.face_target_ms or face_ms, reason="search")
    )
  return _commit_seek(mem, cfg, _seek_spin_nudge(mem, cfg, target_ms=scan_ms, reason="scan"))


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
