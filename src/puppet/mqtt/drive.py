"""Publish one-shot drive turns so the chassis can face the speaker."""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Optional

from puppet.core.audio.respeaker import signed_heading_error_deg

logger = logging.getLogger(__name__)


class DriveClient:
  """Thin MQTT publisher for robot/drive/cmd (turn toward DoA)."""

  def __init__(
    self,
    *,
    broker: str = "127.0.0.1",
    port: int = 1883,
    cmd_topic: str = "robot/drive/cmd",
    front_deg: float = 60.0,
    deadband_deg: float = 25.0,
    max_turn_deg: float = 120.0,
    ms_per_deg: float = 8.0,
    turn_speed: int = 120,
    ttl_ms: int = 300,
    invert: bool = False,
  ) -> None:
    self.broker = broker
    self.port = port
    self.cmd_topic = cmd_topic
    self.front_deg = float(front_deg)
    self.deadband_deg = float(deadband_deg)
    self.max_turn_deg = float(max_turn_deg)
    self.ms_per_deg = float(ms_per_deg)
    self.turn_speed = int(turn_speed)
    self.ttl_ms = int(ttl_ms)
    self.invert = bool(invert)
    self._client = None
    self._error: Optional[str] = None

  def start(self) -> None:
    try:
      import paho.mqtt.client as mqtt
    except ImportError:
      self._error = "paho-mqtt not installed"
      logger.warning(self._error)
      return
    client = mqtt.Client(
      mqtt.CallbackAPIVersion.VERSION2,
      client_id=f"puppet_drive_{os.getpid()}",
    )
    try:
      client.connect(self.broker, self.port, keepalive=30)
      client.loop_start()
      self._client = client
      logger.info("Drive MQTT publisher → %s @ %s:%s", self.cmd_topic, self.broker, self.port)
    except Exception as exc:  # noqa: BLE001
      self._error = str(exc)
      logger.warning("Drive MQTT connect failed: %s", exc)

  def stop(self) -> None:
    if self._client is not None:
      self._client.loop_stop()
      self._client.disconnect()
      self._client = None

  def publish_cmd(self, body: dict[str, Any]) -> dict[str, Any]:
    if self._client is None:
      return {"ok": False, "error": self._error or "mqtt not connected"}
    try:
      self._client.publish(self.cmd_topic, json.dumps(body), qos=1)
    except Exception as exc:  # noqa: BLE001
      return {"ok": False, "error": str(exc)}
    return {"ok": True, "published": body}

  def face_azimuth(self, azimuth_deg: float) -> dict[str, Any]:
    """Turn left/right so chassis front aligns with the given DoA azimuth."""
    error = signed_heading_error_deg(azimuth_deg, front_deg=self.front_deg)
    if self.invert:
      error = -error
    abs_err = abs(error)
    if abs_err < self.deadband_deg:
      logger.info(
        "DoA face skip (within deadband): az=%s° front=%s° err=%+.0f°",
        int(azimuth_deg) % 360,
        int(self.front_deg),
        error,
      )
      return {"ok": True, "skipped": True, "error_deg": error, "azimuth_deg": int(azimuth_deg) % 360}

    turn_deg = min(abs_err, self.max_turn_deg)
    dur_ms = max(80, int(round(turn_deg * self.ms_per_deg)))
    # Positive error → speaker on the right → turn right
    cmd = "turn_right" if error > 0 else "turn_left"
    body = {
      "cmd": cmd,
      "speed": self.turn_speed,
      "ttl": self.ttl_ms,
      "dur": dur_ms,
    }
    result = self.publish_cmd(body)
    if result.get("ok"):
      logger.info(
        "DoA face az=%s° front=%s° err=%+.0f° → %s dur=%sms",
        int(azimuth_deg) % 360,
        int(self.front_deg),
        error,
        cmd,
        dur_ms,
      )
    else:
      logger.warning("DoA face turn failed: %s", result.get("error"))
    result.update(
      {
        "azimuth_deg": int(azimuth_deg) % 360,
        "error_deg": error,
        "turn_deg": turn_deg,
        "cmd": cmd,
        "dur": dur_ms,
      }
    )
    return result
