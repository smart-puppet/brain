import threading
from unittest.mock import MagicMock

from puppet.hardware.mouth import PhonemeMouth
from puppet.tts.alignments import holds_ms_to_timeline
from puppet.tts.types import MouthEvent


class FakePlayback:
  def __init__(self) -> None:
    self._lock = threading.Lock()
    self._samples = 0
    self._cond = threading.Condition(self._lock)

  def samples_written(self) -> int:
    with self._lock:
      return self._samples

  def playback_position_samples(self) -> int:
    return self.samples_written()

  def advance(self, n: int) -> None:
    with self._cond:
      self._samples += n
      self._cond.notify_all()


def test_fallback_holds_to_timeline_alternates() -> None:
  events = holds_ms_to_timeline([50, 50, 50], sample_rate=22050)
  assert len(events) == 3
  assert events[0] == MouthEvent(0, True)
  assert events[1].open is False
  assert events[2].open is True
  assert events[1].sample_offset > 0
  assert events[2].sample_offset > events[1].sample_offset


def test_fallback_mode_timeline_follows_playback_clock() -> None:
  pca = MagicMock()
  playback = FakePlayback()
  mouth = PhonemeMouth(pca, channel=15, closed_deg=0.0, open_deg=25.0, mode="fallback")
  timeline = holds_ms_to_timeline([50, 50, 50], sample_rate=22050)
  mouth.append_timeline(timeline, playback, source="fixed")
  playback.advance(1200)
  mouth.pump_timeline(playback)
  pca.set_servo_angle.assert_any_call(15, 25.0)
  mouth.reset()


def test_fallback_mode_timeline_closes_after_hold() -> None:
  pca = MagicMock()
  playback = FakePlayback()
  mouth = PhonemeMouth(
    pca,
    channel=15,
    closed_deg=0.0,
    open_deg=25.0,
    mode="fallback",
    sample_rate=22050,
  )
  timeline = holds_ms_to_timeline([80, 300], sample_rate=22050)
  mouth.append_timeline(timeline, playback, source="phoneme")
  playback.advance(5000)
  mouth.pump_timeline(playback)
  calls = [c.args for c in pca.set_servo_angle.call_args_list]
  assert (15, 25.0) in calls
  assert (15, 0.0) in calls
  mouth.reset()


def test_pump_timeline_keeps_on_time_toggles() -> None:
  pca = MagicMock()
  playback = FakePlayback()
  mouth = PhonemeMouth(pca, channel=15, closed_deg=0.0, open_deg=25.0, mode="word", sample_rate=22050)
  mouth.append_timeline(
    [
      MouthEvent(100, True),
      MouthEvent(200, False),
      MouthEvent(300, True),
      MouthEvent(400, False),
    ],
    playback,
    source="phoneme",
  )
  pca.reset_mock()
  playback.advance(500)
  mouth.pump_timeline(playback)
  calls = [c.args for c in pca.set_servo_angle.call_args_list]
  assert calls == [(15, 25.0), (15, 0.0), (15, 25.0), (15, 0.0)]
  mouth.reset()


def test_pump_timeline_coalesces_only_when_late() -> None:
  pca = MagicMock()
  playback = FakePlayback()
  mouth = PhonemeMouth(pca, channel=15, closed_deg=0.0, open_deg=25.0, mode="word", sample_rate=22050)
  mouth.append_timeline(
    [
      MouthEvent(0, True),
      MouthEvent(100, False),
      MouthEvent(200, True),
      MouthEvent(300, False),
    ],
    playback,
    source="phoneme",
  )
  pca.reset_mock()
  # Jump far past the late-catchup window (~120ms @ 22.05kHz).
  playback.advance(10_000)
  mouth.pump_timeline(playback)
  calls = [c.args for c in pca.set_servo_angle.call_args_list]
  assert calls == [(15, 0.0)]
  mouth.reset()


def test_word_mode_timeline_follows_playback_clock() -> None:
  pca = MagicMock()
  playback = FakePlayback()
  mouth = PhonemeMouth(pca, channel=15, closed_deg=0.0, open_deg=25.0, mode="word")
  mouth.append_timeline(
    [
      MouthEvent(100, True),
      MouthEvent(300, False),
    ],
    playback,
  )
  playback.advance(150)
  mouth.pump_timeline(playback)
  pca.set_servo_angle.assert_any_call(15, 25.0)
  mouth.reset()


def test_word_mode_timeline_applies_late_events() -> None:
  pca = MagicMock()
  playback = FakePlayback()
  mouth = PhonemeMouth(
    pca,
    channel=15,
    closed_deg=0.0,
    open_deg=25.0,
    mode="word",
    sample_rate=22050,
  )
  mouth.append_timeline(
    [MouthEvent(100, True), MouthEvent(300, False)],
    playback,
    playback_delay_samples=0,
  )
  pca.reset_mock()
  playback.advance(5000)
  mouth.pump_timeline(playback)
  # Far behind: coalesce to the final closed state.
  pca.set_servo_angle.assert_called_with(15, 0.0)
  mouth.reset()


def test_word_mode_timeline_uses_playback_delay() -> None:
  pca = MagicMock()
  playback = FakePlayback()
  mouth = PhonemeMouth(
    pca,
    channel=15,
    closed_deg=0.0,
    open_deg=25.0,
    mode="word",
    sample_rate=22050,
  )
  mouth.on_reply_sync_start()
  mouth.append_timeline(
    [MouthEvent(0, True), MouthEvent(500, False)],
    playback,
    playback_delay_samples=2205,
  )
  playback.advance(2200)
  mouth.pump_timeline(playback)
  pca.set_servo_angle.assert_not_called()
  playback.advance(100)
  mouth.pump_timeline(playback)
  pca.set_servo_angle.assert_any_call(15, 25.0)
  mouth.reset()


def test_word_mode_start_delay_synced_to_playback() -> None:
  pca = MagicMock()
  playback = FakePlayback()
  mouth = PhonemeMouth(
    pca,
    channel=15,
    closed_deg=0.0,
    open_deg=25.0,
    mode="word",
    start_delay_ms=100,
    sample_rate=22050,
  )
  mouth.on_reply_sync_start()
  mouth.append_timeline([MouthEvent(0, True), MouthEvent(500, False)], playback)
  playback.advance(1000)
  mouth.pump_timeline(playback)
  pca.set_servo_angle.assert_not_called()
  # Open target is ~2205 samples; stay inside the late-catchup window.
  playback.advance(1300)
  mouth.pump_timeline(playback)
  pca.set_servo_angle.assert_any_call(15, 25.0)
  mouth.reset()


def test_word_mode_start_delay_survives_chunk_handoff() -> None:
  pca = MagicMock()
  playback = FakePlayback()
  mouth = PhonemeMouth(
    pca,
    channel=15,
    closed_deg=0.0,
    open_deg=25.0,
    mode="word",
    start_delay_ms=100,
    sample_rate=22050,
  )
  mouth.on_reply_sync_start()
  mouth.append_timeline([MouthEvent(0, True), MouthEvent(500, False)], playback)
  mouth.append_timeline(
    [MouthEvent(1000, True), MouthEvent(1500, False)],
    playback,
  )
  playback.advance(2500)
  mouth.pump_timeline(playback)
  pca.set_servo_angle.assert_any_call(15, 25.0)
  mouth.reset()


def test_close_closes_mouth_on_shutdown() -> None:
  pca = MagicMock()
  mouth = PhonemeMouth(pca, channel=15, closed_deg=20.0, open_deg=0.0, mode="word")
  mouth.close_for_listen()
  pca.reset_mock()
  mouth.close()
  pca.set_servo_angle.assert_called_once_with(15, 20.0)
  mouth.close()
  pca.set_servo_angle.assert_called_once()


def test_open_at_start_opens_mouth() -> None:
  pca = MagicMock()
  mouth = PhonemeMouth(pca, channel=15, closed_deg=20.0, open_deg=5.0, mode="word")
  pca.reset_mock()
  mouth.open_at_start()
  pca.set_servo_angle.assert_called_once_with(15, 5.0)


def test_create_mouth_disabled_by_default() -> None:
  from puppet.hardware.mouth import NullMouth, create_mouth

  mouth = create_mouth({"puppet": {"mouth": {"enabled": False}}})
  assert isinstance(mouth, NullMouth)
