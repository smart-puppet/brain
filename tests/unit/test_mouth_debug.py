import logging

from puppet.hardware.mouth_debug import MouthDebug, configure_mouth_logging
from puppet.tts.types import MouthEvent


def test_mouth_debug_logs_when_enabled(caplog) -> None:
  caplog.set_level(logging.DEBUG, logger="puppet.mouth")
  dbg = MouthDebug(enabled=True, sample_rate=22050)
  dbg.timeline_scheduled(
    [MouthEvent(100, True), MouthEvent(500, False)],
    reply_sample=0,
    playback_delay_samples=4096,
    generation=1,
    source="phoneme",
  )
  assert any("schedule phoneme" in r.message for r in caplog.records)
  assert any("OPEN" in r.message for r in caplog.records)


def test_mouth_debug_silent_when_disabled(caplog) -> None:
  caplog.set_level(logging.DEBUG, logger="puppet.mouth")
  dbg = MouthDebug(enabled=False, sample_rate=22050)
  dbg.servo(
    open_mouth=True,
    angle=25.0,
    target_samples=100,
    playback_samples=100,
    generation=1,
  )
  assert not [r for r in caplog.records if r.name == "puppet.mouth"]


def test_configure_mouth_logging_enables_logger() -> None:
  configure_mouth_logging({"puppet": {"mouth": {"debug": True}}})
  assert logging.getLogger("puppet.mouth").level == logging.DEBUG
