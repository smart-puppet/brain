"""Local occupancy map for hide-and-seek (dead-reckoned pose + BEV/sectors).

Not SLAM: pose is the sum of timed nudges. Good enough to prefer unexplored
floor over retracing the same patch of carpet.
"""

from __future__ import annotations

import math
import time
from typing import Any, Optional

import numpy as np

UNKNOWN = 0
FREE = 1
OCCUPIED = 2


def decode_costmap(costmap: Any) -> Optional[np.ndarray]:
  if not isinstance(costmap, dict):
    return None
  try:
    width = int(costmap.get("w") or 0)
    height = int(costmap.get("h") or 0)
  except (TypeError, ValueError):
    return None
  data = costmap.get("data")
  if width <= 0 or height <= 0 or not isinstance(data, list):
    return None
  need = width * height
  flat = np.empty(need, dtype=np.uint8)
  i = 0
  for run in data:
    if not isinstance(run, (list, tuple)) or len(run) < 2:
      return None
    try:
      value = int(run[0])
      count = int(run[1])
    except (TypeError, ValueError):
      return None
    if count <= 0 or i + count > need:
      return None
    flat[i : i + count] = value
    i += count
  if i != need:
    return None
  return flat.reshape(height, width)


def encode_grid_rle(grid: np.ndarray) -> list[list[int]]:
  flat = np.asarray(grid).reshape(-1)
  runs: list[list[int]] = []
  if flat.size == 0:
    return runs
  val = int(flat[0])
  count = 1
  for raw in flat[1:]:
    iv = int(raw)
    if iv == val:
      count += 1
    else:
      runs.append([val, count])
      val, count = iv, 1
  runs.append([val, count])
  return runs


class OccupancyMap:
  """Robot-centric at start: +x forward, +y left, yaw 0 = +x, left turn increases yaw."""

  def __init__(
    self,
    *,
    res_m: float = 0.1,
    size_m: float = 12.0,
  ) -> None:
    self.res_m = max(0.05, float(res_m))
    self.size_m = max(4.0, float(size_m))
    n = max(16, int(round(self.size_m / self.res_m)))
    self.n = n
    self.origin = n // 2
    self.grid = np.zeros((n, n), dtype=np.uint8)
    self.seen = np.zeros((n, n), dtype=np.uint8)
    self.x = 0.0
    self.y = 0.0
    self.yaw_deg = 0.0
    self.started = time.monotonic()

  def elapsed_s(self) -> float:
    return max(0.0, time.monotonic() - self.started)

  def snapshot(self) -> dict[str, Any]:
    known = int(np.count_nonzero(self.grid != UNKNOWN))
    return {
      "frontiers": self.frontier_count(),
      "known": known,
      "x": round(self.x, 2),
      "y": round(self.y, 2),
      "yaw": round(self.yaw_deg, 1),
      "n": int(self.n),
      "res_m": float(self.res_m),
      "size_m": float(self.size_m),
      "grid": encode_grid_rle(self.grid),
    }

  def apply_nudge(self, nudge: Any, cfg: Any) -> None:
    cmd = str(getattr(nudge, "cmd", "") or "")
    dur_ms = max(0, int(getattr(nudge, "dur_ms", 0) or 0))
    if dur_ms <= 0:
      return
    ms_per_deg = max(1, int(getattr(cfg, "turn_ms_per_deg", 8)))
    if cmd == "turn_left":
      self.yaw_deg += dur_ms / ms_per_deg
    elif cmd == "turn_right":
      self.yaw_deg -= dur_ms / ms_per_deg
    elif cmd in ("forward", "backward"):
      speed = int(getattr(nudge, "speed", 0) or getattr(cfg, "forward_speed", 105))
      base_speed = max(1, int(getattr(cfg, "forward_speed", 105)))
      rate = float(getattr(cfg, "forward_m_per_s", 0.3)) * (speed / base_speed)
      dist = (dur_ms / 1000.0) * max(0.05, rate)
      if cmd == "backward":
        dist = -dist
      rad = math.radians(self.yaw_deg)
      self.x += dist * math.cos(rad)
      self.y += dist * math.sin(rad)
    self.yaw_deg = (self.yaw_deg + 180.0) % 360.0 - 180.0
    self._stamp_cell(self.x, self.y, FREE)

  def integrate(self, scene: dict[str, Any]) -> None:
    bev = decode_costmap(scene.get("costmap"))
    if bev is not None:
      self._integrate_bev(bev, float((scene.get("costmap") or {}).get("res_m") or 0.05))
      return
    sectors = scene.get("sectors") if isinstance(scene.get("sectors"), dict) else {}
    self._integrate_sectors(sectors)

  def frontier_bearing_deg(self) -> Optional[float]:
    """Bearing to a far, open frontier in the robot frame (+ = left)."""
    goal = self._best_frontier()
    if goal is None:
      return None
    dx = goal[0] - self.x
    dy = goal[1] - self.y
    rad = math.radians(self.yaw_deg)
    xr = math.cos(rad) * dx + math.sin(rad) * dy
    yr = -math.sin(rad) * dx + math.cos(rad) * dy
    if xr * xr + yr * yr < 0.04:
      return None
    return math.degrees(math.atan2(yr, xr))

  def mark_station(self, radius_m: float = 1.1) -> None:
    """Forget nearby frontiers so the next goal is farther into the room."""
    radius = max(1, int(round(radius_m / self.res_m)))
    idx = self._world_to_idx(self.x, self.y)
    if idx is None:
      return
    cr, cc = idx
    yy, xx = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    disk = xx * xx + yy * yy <= radius * radius
    r0, r1 = cr - radius, cr + radius + 1
    c0, c1 = cc - radius, cc + radius + 1
    sr0, sr1 = max(0, r0), min(self.n, r1)
    sc0, sc1 = max(0, c0), min(self.n, c1)
    self.seen[sr0:sr1, sc0:sc1] = np.maximum(
      self.seen[sr0:sr1, sc0:sc1],
      disk[sr0 - r0 : sr1 - r0, sc0 - c0 : sc1 - c0].astype(np.uint8),
    )

  def mark_dead_ahead(self, range_m: float = 1.8, half_deg: float = 55.0) -> None:
    """Closed-ahead corners are dead ends — do not keep picking them as goals."""
    steps = max(1, int(round(range_m / self.res_m)))
    yaw = math.radians(self.yaw_deg)
    for i in range(1, steps + 1):
      r = i * self.res_m
      for deg in range(-int(half_deg), int(half_deg) + 1, 5):
        bearing = yaw + math.radians(deg)
        wx = self.x + r * math.cos(bearing)
        wy = self.y + r * math.sin(bearing)
        idx = self._world_to_idx(wx, wy)
        if idx is None:
          continue
        row, col = idx
        self.seen[row, col] = 1
        if r >= 0.45 and self.grid[row, col] == UNKNOWN:
          self.grid[row, col] = OCCUPIED if r >= 0.9 else FREE

  def frontier_count(self) -> int:
    return int(np.count_nonzero(self._frontier_mask()))

  def has_frontier(self) -> bool:
    return self.frontier_count() > 0

  def _world_to_idx(self, x: float, y: float) -> Optional[tuple[int, int]]:
    col = self.origin + int(round(x / self.res_m))
    row = self.origin - int(round(y / self.res_m))
    if 0 <= row < self.n and 0 <= col < self.n:
      return row, col
    return None

  def _idx_to_world(self, row: int, col: int) -> tuple[float, float]:
    x = (col - self.origin) * self.res_m
    y = (self.origin - row) * self.res_m
    return x, y

  def _stamp_cell(self, x: float, y: float, value: int) -> None:
    idx = self._world_to_idx(x, y)
    if idx is None:
      return
    row, col = idx
    if value == OCCUPIED:
      self.grid[row, col] = OCCUPIED
    elif self.grid[row, col] != OCCUPIED:
      self.grid[row, col] = value

  def _integrate_bev(self, bev: np.ndarray, bev_res: float) -> None:
    gh, gw = bev.shape[:2]
    yaw = math.radians(self.yaw_deg)
    cy, sy = math.cos(yaw), math.sin(yaw)
    res = max(0.04, float(bev_res))
    for row in range(gh):
      fwd = (gh - 1 - row) * res
      for col in range(gw):
        cell = int(bev[row, col])
        if cell == 255:
          continue
        left = ((gw - 1) * 0.5 - col) * res
        wx = self.x + fwd * cy - left * sy
        wy = self.y + fwd * sy + left * cy
        self._stamp_cell(wx, wy, FREE if cell == 0 else OCCUPIED)

  def _integrate_sectors(self, sectors: dict[str, Any]) -> None:
    wedges = (
      ("left", 50.0),
      ("center", 0.0),
      ("right", -50.0),
    )
    for name, offset in wedges:
      try:
        dist = float(sectors.get(name))
      except (TypeError, ValueError):
        continue
      if dist != dist or dist <= 0.15:
        continue
      bearing = math.radians(self.yaw_deg + offset)
      steps = max(1, int(dist / self.res_m))
      for i in range(1, steps):
        r = i * self.res_m
        self._stamp_cell(self.x + r * math.cos(bearing), self.y + r * math.sin(bearing), FREE)
      if dist < 4.5:
        self._stamp_cell(
          self.x + dist * math.cos(bearing),
          self.y + dist * math.sin(bearing),
          OCCUPIED,
        )

  def _frontier_mask(self) -> np.ndarray:
    free = self.grid == FREE
    unknown = self.grid == UNKNOWN
    neigh = (
      np.roll(free, 1, 0)
      | np.roll(free, -1, 0)
      | np.roll(free, 1, 1)
      | np.roll(free, -1, 1)
    )
    mask = unknown & neigh
    mask[0, :] = False
    mask[-1, :] = False
    mask[:, 0] = False
    mask[:, -1] = False
    return mask

  def _best_frontier(self) -> Optional[tuple[float, float]]:
    mask = self._frontier_mask()
    occ = self.grid == OCCUPIED
    occ_n = (
      np.roll(occ, 1, 0).astype(np.int16)
      + np.roll(occ, -1, 0)
      + np.roll(occ, 1, 1)
      + np.roll(occ, -1, 1)
    )
    mask &= occ_n < 2
    open_mask = mask & (occ_n == 0)
    if open_mask.any():
      mask = open_mask
    if self.seen.any():
      far = mask & (self.seen == 0)
      if far.any():
        mask = far
    if not mask.any():
      return None
    rows, cols = np.nonzero(mask)
    best: Optional[tuple[float, float]] = None
    best_score = -1.0
    min_d = 1.2
    max_d = min(5.5, self.size_m * 0.45)
    for row, col in zip(rows, cols):
      x, y = self._idx_to_world(int(row), int(col))
      d = math.hypot(x - self.x, y - self.y)
      if d < min_d or d > max_d:
        continue
      score = d * (4.0 - float(occ_n[row, col]))
      if score > best_score:
        best_score = score
        best = (x, y)
    return best
