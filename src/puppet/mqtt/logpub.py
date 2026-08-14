"""Publish Python logs to MQTT so eyes debug_web can tail brain/drive."""

from __future__ import annotations

import json
import logging
import os
from typing import Callable, Optional

logger = logging.getLogger(__name__)

_SKIP_LOGGERS = ("paho", "uvicorn", "urllib3", "asyncio")


class MqttLogHandler(logging.Handler):
  """QoS-0 JSON lines on robot/log/<source>. Never logs its own publish errors."""

  def __init__(self, publish: Callable[[str], None], *, source: str) -> None:
    super().__init__()
    self._publish = publish
    self._source = source

  def emit(self, record: logging.LogRecord) -> None:
    name = record.name or ""
    if any(name == p or name.startswith(p + ".") for p in _SKIP_LOGGERS):
      return
    try:
      msg = record.getMessage()
    except Exception:
      return
    if not msg:
      return
    lowered = msg.lower()
    if record.levelno <= logging.DEBUG and (
      "heartbeat" in lowered or "uart send st" in lowered or "uart send hb" in lowered
    ):
      return
    payload = {
      "ts": record.created,
      "level": record.levelname,
      "logger": name,
      "msg": msg[:800],
      "source": self._source,
    }
    try:
      self._publish(json.dumps(payload, ensure_ascii=False))
    except Exception:
      pass


class MqttLogPublisher:
  """Own a small MQTT client and attach a handler to the process root logger."""

  def __init__(
    self,
    *,
    broker: str = "127.0.0.1",
    port: int = 1883,
    topic: str = "robot/log/brain",
    source: str = "brain",
    username: Optional[str] = None,
    password: Optional[str] = None,
  ) -> None:
    self.broker = broker
    self.port = port
    self.topic = topic
    self.source = source
    self.username = username
    self.password = password
    self._client = None
    self._handler: Optional[MqttLogHandler] = None

  def start(self) -> None:
    try:
      import paho.mqtt.client as mqtt
    except ImportError:
      logger.warning("MQTT log publisher skipped (paho-mqtt missing)")
      return
    client = mqtt.Client(
      mqtt.CallbackAPIVersion.VERSION2,
      client_id=f"puppet_log_{self.source}_{os.getpid()}",
    )
    if self.username:
      client.username_pw_set(self.username, self.password)
    try:
      client.connect(self.broker, self.port, keepalive=30)
      client.loop_start()
    except Exception as exc:  # noqa: BLE001
      logger.warning("MQTT log publisher connect failed: %s", exc)
      return
    self._client = client

    def _publish(payload: str) -> None:
      if self._client is None:
        return
      self._client.publish(self.topic, payload, qos=0)

    handler = MqttLogHandler(_publish, source=self.source)
    handler.setLevel(logging.DEBUG)
    logging.getLogger().addHandler(handler)
    self._handler = handler
    logger.info("MQTT logs → %s", self.topic)

  def stop(self) -> None:
    if self._handler is not None:
      logging.getLogger().removeHandler(self._handler)
      self._handler = None
    if self._client is not None:
      try:
        self._client.loop_stop()
        self._client.disconnect()
      except Exception:
        pass
      self._client = None
