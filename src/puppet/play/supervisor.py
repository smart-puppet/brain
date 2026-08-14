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
    busy_fn: Optional[BusyFn] = None,
    heading_fn: Optional[Callable[[], Optional[float]]] = None,
  ) -> None:
    self._scene = scene
    self._drive = drive
    self.cfg = config or PlayConfig()
    self.allow_motion = bool(allow_motion)
    self._tick_s = max(0.05, float(tick_s))
    self.cmd_topic = cmd_topic
    self.status_topic = status_topic
    self._busy_fn = busy_fn
    self._heading_fn = heading_fn
    self._lock = threading.Lock()
    self._mode = "idle"
    self._mem = PlayMemory()
    self._thread: Optional[threading.Thread] = None
    self._stop = threading.Event()
    self._last_cmd: Optional[str] = None
    self._status: dict[str, Any] = {"mode": "idle", "reason": "idle"}
    self._cmd_client = None

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

  def set_mode(self, mode: str) -> str:
    mode = (mode or "idle").strip().lower()
    if mode not in ("idle", "follow", "seek"):
      mode = "idle"
    with self._lock:
      same = mode == self._mode
      self._mode = mode
      if not same:
        self._mem = PlayMemory()
      self._status = {"mode": mode, "reason": "start" if mode != "idle" else "idle"}
    if mode == "idle":
      self._idle(force=True)
    if not same:
      logger.info("Play mode → %s", mode)
      self._publish_status()
    return mode

  def backup_once(self) -> None:
    """Stop follow/seek, then one timed reverse nudge."""
    self.set_mode("idle")
    self._apply(
      DriveNudge(
        "backward",
        speed=self.cfg.backward_speed,
        dur_ms=self.cfg.backward_dur_ms,
        reason="voice_back",
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

    def _on_message(c, userdata, msg):
      try:
        payload = json.loads(msg.payload.decode("utf-8"))
      except Exception:
        return
      if not isinstance(payload, dict):
        return
      mode = str(payload.get("mode") or payload.get("cmd") or "").lower()
      if mode in ("stop", "idle", "follow", "seek", "back"):
        if mode == "back":
          self.backup_once()
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
    while not self._stop.wait(self._tick_s):
      mode = self.mode
      if mode == "idle":
        continue
      if self._busy_fn is not None and self._busy_fn():
        # Skip ticks but do not publish idle — that cancelled DoA face-speaker turns.
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
        "ts": time.time(),
      }
      with self._lock:
        self._status = status
      logger.info(
        "Play %s → %s (%s) person=%s/%s closest=%s",
        mode_now,
        nudge.cmd,
        nudge.reason,
        person.get("bearing"),
        person.get("dist_m"),
        result.get("closest_m"),
      )
      self._apply(nudge)
      self._publish_status()
      if nudge.reason == "found":
        self.set_mode("idle")
