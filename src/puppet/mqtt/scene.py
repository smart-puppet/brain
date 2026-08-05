"""Subscribe to robot/nav/scene for Gemma vision context."""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from typing import Any, Optional

logger = logging.getLogger(__name__)


class SceneIngest:
    """Caches the latest eyes traversability hint from MQTT."""

    def __init__(
        self,
        *,
        broker: str = "127.0.0.1",
        port: int = 1883,
        topic: str = "robot/nav/scene",
        min_interval_s: float = 1.0,
    ) -> None:
        self.broker = broker
        self.port = port
        self.topic = topic
        self.min_interval_s = min_interval_s
        self._lock = threading.Lock()
        self._hint = ""
        self._objects: list[dict[str, Any]] = []
        self._ts = 0.0
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
            client_id=f"puppet_scene_{os.getpid()}",
        )
        client.on_connect = self._on_connect
        client.on_message = self._on_message
        try:
            client.connect(self.broker, self.port, keepalive=30)
            client.loop_start()
            self._client = client
            logger.info("Vision MQTT subscribed to %s @ %s:%s", self.topic, self.broker, self.port)
        except Exception as exc:  # noqa: BLE001
            self._error = str(exc)
            logger.warning("Vision MQTT connect failed: %s", exc)

    def stop(self) -> None:
        if self._client is not None:
            self._client.loop_stop()
            self._client.disconnect()
            self._client = None

    def _on_connect(self, client, userdata, flags, reason_code, properties=None):
        client.subscribe(self.topic)

    def _on_message(self, client, userdata, msg):
        try:
            payload = json.loads(msg.payload.decode("utf-8"))
        except Exception:
            return
        now = time.time()
        with self._lock:
            if now - self._ts < self.min_interval_s and self._hint:
                # Still refresh silently but throttle isn't critical for cache.
                pass
            self._hint = str(payload.get("hint") or "")
            objs = payload.get("objects") or []
            if isinstance(objs, list):
                self._objects = objs[:8]
            self._ts = now

    def context_line(self) -> str:
        with self._lock:
            if not self._hint:
                return ""
            bits = [f"Vision: {self._hint}"]
            if self._objects:
                brief = []
                for o in self._objects[:5]:
                    lab = o.get("label", "?")
                    dist = o.get("dist_m")
                    br = o.get("bearing", "")
                    if dist is not None:
                        brief.append(f"{lab}@{dist}m/{br}")
                    else:
                        brief.append(f"{lab}/{br}")
                bits.append("Objects: " + ", ".join(brief))
            age = time.time() - self._ts
            if age > 5.0:
                bits.append(f"(stale {age:.0f}s)")
            return " | ".join(bits)
