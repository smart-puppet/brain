from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Any, Literal, Protocol

from puppet.hardware.mouth_debug import MouthDebug, configure_mouth_logging
from puppet.tts.alignments import resolve_mouth_mode
from puppet.tts.types import MouthEvent

logger = logging.getLogger(__name__)

MouthMode = Literal["word", "fallback"]


@dataclass(frozen=True)
class _TimelineItem:
  target_samples: int
  open_mouth: bool
  generation: int
  source: str


class PlaybackSync(Protocol):
  def samples_written(self) -> int: ...

  def playback_position_samples(self) -> int: ...


class MouthController(Protocol):
  def append_timeline(
    self,
    events: list[MouthEvent],
    playback: PlaybackSync,
    *,
    playback_delay_samples: int = 0,
    source: str = "phoneme",
  ) -> None: ...

  def pump_timeline(self, playback: PlaybackSync) -> None: ...

  def on_phrase_end(self) -> None: ...

  def reset(self) -> None: ...

  def clear_sync(self) -> None: ...

  def close_for_listen(self) -> None: ...

  def open_at_start(self) -> None: ...

  def on_reply_sync_start(self) -> None: ...

  def close(self) -> None: ...


class NullMouth:
  def append_timeline(
    self,
    events: list[MouthEvent],
    playback: PlaybackSync,
    *,
    playback_delay_samples: int = 0,
    source: str = "phoneme",
  ) -> None:
    del events, playback, playback_delay_samples, source

  def pump_timeline(self, playback: PlaybackSync) -> None:
    del playback

  def on_phrase_end(self) -> None:
    pass

  def reset(self) -> None:
    pass

  def clear_sync(self) -> None:
    pass

  def close_for_listen(self) -> None:
    pass

  def open_at_start(self) -> None:
    pass

  def on_reply_sync_start(self) -> None:
    pass

  def close(self) -> None:
    pass


class PhonemeMouth:
  """Binary jaw (closed_deg / open_deg only), synced via the audio sample clock."""

  def __init__(
    self,
    pca: object,
    *,
    channel: int,
    closed_deg: float,
    open_deg: float,
    mode: MouthMode = "word",
    start_delay_ms: int = 0,
    sample_rate: int = 22050,
    bus: object | None = None,
    debug: MouthDebug | None = None,
  ) -> None:
    self._debug = debug or MouthDebug(enabled=False, sample_rate=sample_rate)
    self._sample_rate = max(1, int(sample_rate))
    self._pca = pca
    self._channel = channel
    self._closed_deg = float(closed_deg)
    self._open_deg = float(open_deg)
    self._mode = mode
    self._start_delay_s = max(0.0, start_delay_ms / 1000.0)
    self._start_delay_samples = max(0, int(round(self._sample_rate * start_delay_ms / 1000)))
    self._reply_sample_offset = 0
    self._start_delay_applied = False
    self._is_open = False
    self._lock = threading.Lock()
    self._bus = bus
    self._generation = 0
    self._timeline_lock = threading.Lock()
    self._timeline_cond = threading.Condition(self._timeline_lock)
    self._timeline_queue: list[_TimelineItem] = []
    self._timeline_pumper: threading.Thread | None = None
    # Drop intermediate toggles only when the playhead is this far behind.
    self._timeline_late_catchup_samples = max(1, int(self._sample_rate * 0.12))
    self._timeline_pump_s = 0.010

  def _playback_samples(self, playback: PlaybackSync | None) -> int:
    if playback is None:
      return 0
    if hasattr(playback, "playback_position_samples"):
      return int(playback.playback_position_samples())
    return int(playback.samples_written())

  def _set_angle(
    self,
    angle: float,
    *,
    generation: int = 0,
    source: str = "phoneme",
    playback: PlaybackSync | None = None,
    target_samples: int = 0,
  ) -> None:
    self._is_open = angle == self._open_deg
    with self._lock:
      self._pca.set_servo_angle(self._channel, angle)
    self._debug.servo(
      open_mouth=self._is_open,
      angle=angle,
      target_samples=target_samples,
      playback_samples=self._playback_samples(playback),
      generation=generation,
      source=source,
    )

  def _apply_open(
    self,
    open_mouth: bool,
    *,
    target_samples: int = 0,
    playback: PlaybackSync | None = None,
    generation: int = 0,
    source: str = "phoneme",
    force: bool = False,
  ) -> None:
    if not force and open_mouth == self._is_open:
      return
    angle = self._open_deg if open_mouth else self._closed_deg
    self._set_angle(
      angle,
      generation=generation,
      source=source,
      playback=playback,
      target_samples=target_samples,
    )

  def on_phrase_end(self) -> None:
    pass

  def append_timeline(
    self,
    events: list[MouthEvent],
    playback: PlaybackSync,
    *,
    playback_delay_samples: int = 0,
    source: str = "phoneme",
  ) -> None:
    if not events:
      return
    generation = self._generation
    reply_sample = int(events[0].sample_offset) if events else 0
    self._debug.timeline_scheduled(
      events,
      reply_sample=reply_sample,
      playback_delay_samples=playback_delay_samples,
      generation=generation,
      source=source,
    )
    items = [
      _TimelineItem(
        target_samples=max(
          0,
          int(event.sample_offset) + playback_delay_samples + self._reply_sample_offset,
        ),
        open_mouth=bool(event.open),
        generation=generation,
        source=source,
      )
      for event in events
    ]
    with self._timeline_cond:
      self._timeline_queue.extend(items)
      self._timeline_queue.sort(key=lambda item: item.target_samples)
      self._timeline_cond.notify_all()
    self.pump_timeline(playback)
    self._start_timeline_pumper(playback)

  def _start_timeline_pumper(self, playback: PlaybackSync) -> None:
    """Poll the audio playhead so jaw events fire between ALSA writes."""
    with self._timeline_cond:
      if self._timeline_pumper is not None and self._timeline_pumper.is_alive():
        return
      generation = self._generation
      self._timeline_pumper = threading.Thread(
        target=self._timeline_pump_loop,
        args=(playback, generation),
        daemon=True,
        name="puppet-mouth-timeline",
      )
      self._timeline_pumper.start()

  def _timeline_pump_loop(self, playback: PlaybackSync, generation: int) -> None:
    idle_rounds = 0
    while generation == self._generation:
      self.pump_timeline(playback)
      with self._timeline_cond:
        if self._timeline_queue:
          idle_rounds = 0
          self._timeline_cond.wait(timeout=self._timeline_pump_s)
        else:
          idle_rounds += 1
          # Keep pumping briefly after the queue drains while ALSA still plays.
          if idle_rounds >= 20:
            break
          self._timeline_cond.wait(timeout=self._timeline_pump_s)

  def pump_timeline(self, playback: PlaybackSync) -> None:
    position = self._playback_samples(playback)
    due: list[_TimelineItem] = []
    with self._timeline_cond:
      while self._timeline_queue:
        item = self._timeline_queue[0]
        if item.generation != self._generation:
          self._timeline_queue.pop(0)
          continue
        if item.target_samples > position:
          break
        due.append(self._timeline_queue.pop(0))
    if not due:
      return
    # Only collapse a backlog when we are late; on-time bursts keep every toggle.
    late = position > due[0].target_samples + self._timeline_late_catchup_samples
    if late and len(due) >= 2:
      self._debug.timeline_coalesced(len(due), applied=due[-1])
      due = [due[-1]]
    for item in due:
      if item.generation != self._generation:
        self._debug.generation_cancelled(item.generation)
        continue
      if position > item.target_samples + self._timeline_late_catchup_samples:
        self._debug.timeline_late_catchup(
          target_samples=item.target_samples,
          playback_samples=position,
          generation=item.generation,
          open_mouth=item.open_mouth,
        )
      self._apply_open(
        item.open_mouth,
        target_samples=item.target_samples,
        playback=playback,
        generation=item.generation,
        source=item.source,
        force=True,
      )

  def _clear_sync_state(self) -> None:
    self._generation += 1
    self._reply_sample_offset = 0
    self._start_delay_applied = False
    with self._timeline_cond:
      self._timeline_queue.clear()
      self._timeline_cond.notify_all()
    self._debug.reset(self._generation)

  def clear_sync(self) -> None:
    """Drop queued mouth events without moving the servo."""
    self._clear_sync_state()
    self._is_open = False

  def close_for_listen(self) -> None:
    """Closed-jaw idle pose while Puppet is listening for speech."""
    self._is_open = False
    with self._lock:
      self._pca.set_servo_angle(self._channel, self._closed_deg)

  def open_at_start(self) -> None:
    """Open-jaw pose when the system boots."""
    self._is_open = True
    with self._lock:
      self._pca.set_servo_angle(self._channel, self._open_deg)

  def reset(self) -> None:
    self.clear_sync()
    self.close_for_listen()

  def on_reply_sync_start(self) -> None:
    self._reply_sample_offset = self._start_delay_samples
    self._debug.reply_sync_start()

  def close(self) -> None:
    if getattr(self, "_shutdown", False):
      return
    self._shutdown = True
    self._generation += 1
    self._reply_sample_offset = 0
    self._start_delay_applied = False
    with self._timeline_cond:
      self._timeline_queue.clear()
      self._timeline_cond.notify_all()
    self._is_open = False
    with self._lock:
      self._pca.set_servo_angle(self._channel, self._closed_deg)
    bus = self._bus
    if bus is not None and hasattr(bus, "close"):
      bus.close()
      self._bus = None


def create_mouth(config: dict[str, Any], *, sample_rate: int = 22050) -> MouthController:
  mouth_cfg = config.get("puppet", {}).get("mouth", {})
  if not mouth_cfg.get("enabled", False):
    return NullMouth()

  configure_mouth_logging(config)

  try:
    from smbus2 import SMBus
  except ImportError as exc:
    raise RuntimeError(
      "puppet.mouth.enabled requires smbus2. Install with: pip install smbus2"
    ) from exc

  from puppet.hardware.pca9685 import PCA9685

  bus_no = int(mouth_cfg.get("i2c_bus", 7))
  address = int(mouth_cfg.get("i2c_address", 0x40))
  channel = int(mouth_cfg.get("channel", 15))
  pwm_hz = float(mouth_cfg.get("pwm_hz", 50.0))
  mode = resolve_mouth_mode(config)
  debug = MouthDebug(
    enabled=bool(mouth_cfg.get("debug", False)),
    sample_rate=sample_rate,
  )

  bus = SMBus(bus_no)
  pca = PCA9685(bus, address=address)
  pca.set_pwm_frequency(pwm_hz)

  mouth = PhonemeMouth(
    pca,
    channel=channel,
    closed_deg=float(mouth_cfg.get("closed_deg", 0.0)),
    open_deg=float(mouth_cfg.get("open_deg", 25.0)),
    mode=mode,  # type: ignore[arg-type]
    start_delay_ms=int(mouth_cfg.get("start_delay_ms", 0)),
    sample_rate=sample_rate,
    bus=bus,
    debug=debug,
  )
  logger.info(
    "Mouth servo on I2C bus %d ch %d (binary %.0f° / %.0f°, mode=%s%s)",
    bus_no,
    channel,
    mouth._closed_deg,
    mouth._open_deg,
    mode,
    ", debug on" if debug._enabled else "",
  )
  mouth.open_at_start()
  return mouth
