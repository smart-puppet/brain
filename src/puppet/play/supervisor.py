"""Background play loop: capture scene → policy → timed drive nudges."""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from typing import Any, Callable, Optional

from puppet.mqtt.drive import DriveClient
from puppet.mqtt.scene import SceneIngest
from puppet.play.policy import DriveNudge, PlayConfig, PlayMemory, plan

logger = logging.getLogger(__name__)

BusyFn = Callable[[], bool]


class PlaySupervisor:
  """Owns follow / seek while the voice pipeline stays free to talk.

  Safety:
  - only ``dur>0`` nudges (MCU self-limits)
  - idle when the voice pipeline is busy (thinking / speaking)
  - idle immediately on ``stop()`` / ``idle`` mode
  - ``allow_motion`` must be true to publish anything but idle
  """

  def __init__(
    self,
    *,
    scene: SceneIngest,
    drive: DriveClient,
    config: Optional[PlayConfig] = None,
    allow_motion: bool = False,
    tick_s: float = 0.15,
    cmd_topic: str = "robot/play/cmd",
    status_topic: str = "robot/play/status",
    speeds_topic: str = "robot/play/speeds",
    busy_fn: Optional[BusyFn] = None,
    heading_fn: Optional[Callable[[], Optional[float]]] = None,
    announce_fn: Optional[Callable[[str], None]] = None,
  ) -> None:
    self._scene = scene
    self._drive = drive
    self.cfg = config or PlayConfig()
    self.allow_motion = bool(allow_motion)
    self._tick_s = max(0.05, float(tick_s))
    self.cmd_topic = cmd_topic
    self.status_topic = status_topic
    self.speeds_topic = speeds_topic
    self._busy_fn = busy_fn
    self._heading_fn = heading_fn
    self._announce_fn = announce_fn
    self._lock = threading.Lock()
    self._mode = "idle"
    self._mem = PlayMemory()
    self._thread: Optional[threading.Thread] = None
    self._stop = threading.Event()
    self._last_cmd: Optional[str] = None
    self._status: dict[str, Any] = {"mode": "idle", "reason": "idle"}
    self._cmd_client = None
    self._pending_announce: Optional[str] = None

  @property
  def mode(self) -> str:
    with self._lock:
      return self._mode

  def status(self) -> dict[str, Any]:
    with self._lock:
      return dict(self._status)

  def start(self) -> None:
    if self._thread is not None:
      return
    self._stop.clear()
    self._thread = threading.Thread(target=self._run, name="puppet-play", daemon=True)
    self._thread.start()
    self._start_cmd_listener()
    logger.info(
      "Play supervisor started (motion=%s tick=%.2fs)",
      "on" if self.allow_motion else "off",
      self._tick_s,
    )

  def close(self) -> None:
    self.set_mode("idle")
    self._idle(force=True)
    self._stop.set()
    if self._thread is not None:
      self._thread.join(timeout=2.0)
      self._thread = None
    if self._cmd_client is not None:
      try:
        self._cmd_client.loop_stop()
        self._cmd_client.disconnect()
      except Exception:
        pass
      self._cmd_client = None

  def set_mode(self, mode: str, *, announce: bool = True) -> str:
    mode = (mode or "idle").strip().lower()
    if mode not in ("idle", "follow", "seek"):
      mode = "idle"
    busy = self._is_busy()
    with self._lock:
      previous = self._mode
      same = mode == previous
      self._mode = mode
      if not same:
        self._mem = PlayMemory()
      self._status = {
        "mode": mode,
        "reason": "start" if mode != "idle" else "idle",
        "busy": busy,
      }
      if announce and not same and mode == "seek":
        self._pending_announce = "seek"
    if mode == "idle":
      self._idle(force=True)
    if not same:
      logger.info("Play mode → %s", mode)
      self._publish_status()
    return mode

  def take_pending_announce(self) -> Optional[str]:
    with self._lock:
      kind = self._pending_announce
      self._pending_announce = None
      return kind

  def apply_speeds(self, payload: dict[str, Any]) -> dict[str, int]:
    """Live follow-turn / seek-turn / forward from Eye sliders."""
    from puppet.play.speeds import normalize_speeds

    current = {
      "follow_turn": self.cfg.turn_speed,
      "seek_turn": self.cfg.seek_turn_speed,
      "forward": self.cfg.forward_speed,
    }
    speeds = normalize_speeds(payload, current)
    with self._lock:
      self.cfg.turn_speed = speeds["follow_turn"]
      self.cfg.seek_turn_speed = speeds["seek_turn"]
      self.cfg.forward_speed = speeds["forward"]
    self._drive.turn_speed = speeds["follow_turn"]
    logger.info(
      "Play speeds follow_turn=%s seek_turn=%s forward=%s",
      speeds["follow_turn"],
      speeds["seek_turn"],
      speeds["forward"],
    )
    return speeds

  def backup_once(self) -> None:
    """Stop follow/seek, then one timed reverse nudge."""
    self.set_mode("idle", announce=False)
    self._apply(
      DriveNudge(
        "backward",
        speed=self.cfg.backward_speed,
        dur_ms=self.cfg.backward_dur_ms,
        reason="voice_back",
      )
    )

  def forward_once(self) -> None:
    """Stop follow/seek, then one timed forward nudge."""
    self.set_mode("idle", announce=False)
    self._apply(
      DriveNudge(
        "forward",
        speed=self.cfg.forward_speed,
        dur_ms=self.cfg.forward_dur_ms,
        reason="voice_forward",
      )
    )

  def _start_cmd_listener(self) -> None:
    try:
      import paho.mqtt.client as mqtt
    except ImportError:
      return
    client = mqtt.Client(
      mqtt.CallbackAPIVersion.VERSION2,
      client_id=f"puppet_play_{os.getpid()}",
    )

    def _on_connect(c, userdata, flags, reason_code, properties=None):
      c.subscribe(self.cmd_topic, qos=1)
      c.subscribe(self.speeds_topic, qos=1)

    def _on_message(c, userdata, msg):
      try:
        payload = json.loads(msg.payload.decode("utf-8"))
      except Exception:
        return
      if not isinstance(payload, dict):
        return
      topic = msg.topic or ""
      if topic == self.speeds_topic:
        self.apply_speeds(payload)
        return
      mode = str(payload.get("mode") or payload.get("cmd") or "").lower()
      if mode in ("stop", "idle", "follow", "seek", "back", "forward"):
        if mode == "back":
          self.backup_once()
        elif mode == "forward":
          self.forward_once()
        else:
          self.set_mode("idle" if mode == "stop" else mode)

    client.on_connect = _on_connect
    client.on_message = _on_message
    try:
      client.connect(self._scene.broker, self._scene.port, keepalive=30)
      client.loop_start()
      self._cmd_client = client
    except Exception as exc:  # noqa: BLE001
      logger.warning("Play cmd MQTT failed: %s", exc)

  def _publish_status(self) -> None:
    if self._cmd_client is None:
      return
    try:
      self._cmd_client.publish(self.status_topic, json.dumps(self.status()), qos=0)
    except Exception:
      pass

  def _is_busy(self) -> bool:
    return self._busy_fn is not None and bool(self._busy_fn())

  def _idle(self, *, force: bool = False) -> None:
    if not force and self._last_cmd == "idle":
      return
    self._drive.idle()
    self._last_cmd = "idle"

  def _apply(self, nudge) -> None:
    if nudge.cmd == "idle":
      self._idle()
      return
    if not self.allow_motion:
      logger.warning("Play would %s dur=%sms but allow_motion=false", nudge.cmd, nudge.dur_ms)
      self._idle()
      return
    result = self._drive.nudge(nudge.cmd, dur_ms=nudge.dur_ms, speed=nudge.speed)
    if result.get("ok"):
      self._last_cmd = nudge.cmd
      logger.info(
        "Play drive %s speed=%s dur=%sms reason=%s",
        nudge.cmd,
        nudge.speed,
        nudge.dur_ms,
        nudge.reason,
      )
    else:
      logger.warning("Play nudge failed: %s", result.get("error"))

  def _run(self) -> None:
    last_busy: Optional[bool] = None
    while not self._stop.wait(self._tick_s):
      with self._lock:
        mode = self._mode
        pending = self._pending_announce
      busy = self._is_busy()
      if mode in ("follow", "seek") and busy != last_busy:
        last_busy = busy
        with self._lock:
          self._status = {**self._status, "mode": mode, "busy": busy}
        self._publish_status()
      if mode == "idle":
        last_busy = None
        if not pending:
          continue
      if busy:
        # Skip motion ticks; status.busy pauses the DeepStream camera pipeline.
        continue
      kind = self.take_pending_announce()
      if kind:
        if kind == "seek":
          self._idle(force=True)
        if self._announce_fn is not None:
          try:
            self._announce_fn(kind)
          except Exception:
            logger.exception("Play announce failed (%s)", kind)
        continue
      if mode == "idle":
        continue
      result = self._scene.request_capture()
      if not result.get("ok"):
        age = self._scene.scene_age_s()
        cached = self._scene.latest_scene()
        if cached and age < 2.5:
          logger.warning(
            "Play capture failed (%s); using cached scene age=%.1fs",
            result.get("error"),
            age,
          )
          result = {"ok": True, **cached}
        else:
          logger.warning("Play capture failed: %s", result.get("error"))
          # Do not publish idle — that stops the chassis while DoA face-turns still work.
          continue
      with self._lock:
        mem = self._mem
        cfg = self.cfg
        mode_now = self._mode
      heading_error_deg = None
      if self._heading_fn is not None:
        try:
          heading_error_deg = self._heading_fn()
        except Exception:
          heading_error_deg = None
      nudge = plan(mode_now, result, mem, cfg, heading_error_deg=heading_error_deg)
      person = nudge.person or {}
      status = {
        "mode": mode_now,
        "reason": nudge.reason,
        "cmd": nudge.cmd,
        "dur": nudge.dur_ms,
        "person_m": person.get("dist_m"),
        "person_side": person.get("bearing"),
        "closest_m": result.get("closest_m"),
        "busy": False,
        "ts": time.time(),
      }
      if mode_now == "seek" and getattr(mem, "explore", None) is not None:
        status["explore"] = mem.explore.snapshot()
      with self._lock:
        self._status = status
      logger.info(
        "Play %s → %s (%s) person=%s/%s closest=%s floor=%s",
        mode_now,
        nudge.cmd,
        nudge.reason,
        person.get("bearing"),
        person.get("dist_m"),
        result.get("closest_m"),
        result.get("floor_ahead_pct"),
      )
      if mode_now == "seek" and status.get("explore"):
        exp = status["explore"]
        logger.info(
          "Play explore frontiers=%s known=%s pose=%.2f,%.2f yaw=%.0f",
          exp.get("frontiers"),
          exp.get("known"),
          exp.get("x") or 0.0,
          exp.get("y") or 0.0,
          exp.get("yaw") or 0.0,
        )
      self._apply(nudge)
      self._publish_status()
      if nudge.reason in ("found", "giveup", "nofollow"):
        self.set_mode("idle", announce=False)
        if self._announce_fn is not None:
          try:
            self._announce_fn(nudge.reason)
          except Exception:
            logger.exception("Play announce failed (%s)", nudge.reason)
        continue
      if nudge.dur_ms > 0 and nudge.cmd != "idle":
        # A new dur>0 command preempts the chassis. Finish this sweep first.
        self._stop.wait(min(nudge.dur_ms, 8000) / 1000.0)
