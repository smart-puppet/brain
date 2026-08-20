"""Occupancy map used by hide-and-seek exploration."""

from __future__ import annotations

import time

from puppet.play.explore import FREE, OccupancyMap, decode_costmap
from puppet.play.policy import DriveNudge, PlayConfig, PlayMemory, plan_seek


def test_snapshot_includes_rle_grid() -> None:
  world = OccupancyMap(res_m=0.1, size_m=6.0)
  world._stamp_cell(0.0, 0.0, FREE)
  snap = world.snapshot()
  assert snap["n"] == world.n
  cells = decode_costmap(
    {"w": snap["n"], "h": snap["n"], "data": snap["grid"]}
  )
  assert cells is not None
  assert int(cells[world.origin, world.origin]) == FREE


def test_decode_costmap_rle() -> None:
  grid = decode_costmap(
    {"w": 3, "h": 2, "encoding": "rle", "data": [[0, 3], [100, 2], [255, 1]]}
  )
  assert grid is not None
  assert grid.shape == (2, 3)
  assert int(grid[0, 0]) == 0
  assert int(grid[1, 0]) == 100
  assert int(grid[1, 2]) == 255


def test_turn_nudge_updates_yaw() -> None:
  world = OccupancyMap(res_m=0.1, size_m=6.0)
  world.apply_nudge(DriveNudge("turn_left", dur_ms=900), PlayConfig(turn_ms_per_deg=10))
  assert abs(world.yaw_deg - 90.0) < 0.01


def test_forward_nudge_moves_along_heading() -> None:
  world = OccupancyMap(res_m=0.1, size_m=6.0)
  cfg = PlayConfig(forward_m_per_s=0.5, forward_speed=100)
  world.apply_nudge(DriveNudge("forward", speed=100, dur_ms=1000), cfg)
  assert world.x == 0.5
  assert abs(world.y) < 1e-9


def test_free_corridor_frontier_is_ahead() -> None:
  world = OccupancyMap(res_m=0.1, size_m=6.0)
  for i in range(1, 14):
    world._stamp_cell(i * 0.1, 0.0, FREE)
  bearing = world.frontier_bearing_deg()
  assert bearing is not None
  assert abs(bearing) < 45.0


def test_dead_ahead_is_not_a_frontier_goal() -> None:
  world = OccupancyMap(res_m=0.1, size_m=8.0)
  for i in range(1, 25):
    world._stamp_cell(i * 0.1, 0.0, FREE)
  world.mark_dead_ahead(range_m=1.5, half_deg=40.0)
  bearing = world.frontier_bearing_deg()
  if bearing is not None:
    assert abs(bearing) < 50.0


def test_near_wall_frontier_loses_to_open_far_cell() -> None:
  world = OccupancyMap(res_m=0.1, size_m=8.0)
  for i in range(1, 25):
    world._stamp_cell(i * 0.1, 0.0, FREE)
  world._stamp_cell(0.4, 0.3, FREE)
  from puppet.play.explore import OCCUPIED
  world._stamp_cell(0.4, 0.4, OCCUPIED)
  bearing = world.frontier_bearing_deg()
  assert bearing is not None
  assert abs(bearing) < 40.0


def test_seek_backs_out_of_a_corner() -> None:
  open_scene = {
    "closest_m": 2.5,
    "sectors": {"left": 2.5, "center": 2.5, "right": 2.5},
    "floor_ahead_pct": 0.4,
    "objects": [],
  }
  corner = {
    "closest_m": 0.65,
    "sectors": {"left": 0.7, "center": 0.65, "right": 0.7},
    "floor_ahead_pct": 0.05,
    "objects": [],
  }
  mem = PlayMemory()
  cfg = PlayConfig(
    seek_map=True,
    seek_giveup_ticks=40,
    seek_giveup_s=0,
    turn_ms_per_deg=10,
    follow_spin_deg=90,
    seek_face_deg=0,
    search_turn_dur_ms=300,
  )
  for _ in range(12):
    nudge = plan_seek(open_scene, mem, cfg)
    if nudge.cmd == "forward":
      break
  else:
    raise AssertionError("never started rolling")
  first = plan_seek(corner, mem, cfg)
  assert first.cmd == "backward"
  second = plan_seek(corner, mem, cfg)
  assert second.cmd in ("turn_left", "turn_right")


def test_seek_gives_up_when_time_elapsed() -> None:
  scene = {
    "closest_m": 2.0,
    "sectors": {"left": 2.0, "center": 2.0, "right": 2.0},
    "objects": [],
  }
  mem = PlayMemory()
  cfg = PlayConfig(seek_map=True, seek_giveup_ticks=1000, seek_giveup_s=30.0)
  first = plan_seek(scene, mem, cfg)
  assert first.reason != "giveup"
  assert mem.explore is not None
  mem.explore.started = time.monotonic() - 31.0
  nudge = plan_seek(scene, mem, cfg)
  assert nudge.reason == "giveup"


def test_seek_faces_mapped_frontier_after_scan() -> None:
  scene = {
    "closest_m": 2.0,
    "sectors": {"left": 0.6, "center": 2.4, "right": 0.6},
    "objects": [],
  }
  mem = PlayMemory()
  cfg = PlayConfig(
    seek_map=True,
    turn_ms_per_deg=10,
    follow_spin_deg=90,
    seek_face_deg=90,
    search_turn_dur_ms=300,
    search_forward_ticks=1,
    seek_giveup_ticks=40,
    seek_giveup_s=0,
  )
  nudges = [plan_seek(scene, mem, cfg) for _ in range(8)]
  assert nudges[0].reason == "scan"
  assert any(n.cmd == "forward" for n in nudges)
  assert mem.explore is not None
  assert mem.explore.frontier_count() >= 0
