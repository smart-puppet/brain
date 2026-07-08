import threading
import time
from unittest.mock import MagicMock

from puppet.hardware.mouth import PhonemeMouth
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

  def wait_until_samples(self, sample_index: int, *, timeout: float | None = None) -> bool:
    deadline = None if timeout is None else time.monotonic() + timeout
    with self._cond:
      while self._samples < sample_index:
        if deadline is not None:
          remaining = deadline - time.monotonic()
          if remaining <= 0:
            return False
          self._cond.wait(timeout=remaining)
        else:
          self._cond.wait()
      return True

  def wait_until_samples(self, sample_index: int, *, timeout: float | None = None) -> bool:
    deadline = None if timeout is None else time.monotonic() + timeout
    with self._cond:
      while self._samples < sample_index:
        if deadline is not None:
          remaining = deadline - time.monotonic()
          if remaining <= 0:
            return False
          self._cond.wait(timeout=remaining)
        else:
          self._cond.wait()
      return True


def test_fallback_mode_flips_while_speaking() -> None:
  pca = MagicMock()
  mouth = PhonemeMouth(
    pca,
    channel=15,
    closed_deg=0.0,
    open_deg=25.0,
    mode="fallback",
    fallback_flip_ms=50,
  )
  mouth.append_fallback_durations([50, 50, 50])
  mouth.on_chunk_play(1000, 22050)
  time.sleep(0.16)
  mouth.reset()
  calls = [c.args for c in pca.set_servo_angle.call_args_list]
  assert (15, 25.0) in calls
  assert (15, 0.0) in calls
  assert len(calls) >= 2


def test_fallback_mode_uses_phoneme_hold_durations() -> None:
  pca = MagicMock()
  mouth = PhonemeMouth(
    pca,
    channel=15,
    closed_deg=0.0,
    open_deg=25.0,
    mode="fallback",
    fallback_flip_ms=200,
  )
  mouth.append_fallback_durations([80, 300, 150])
  mouth.on_chunk_play(1000, 22050)
  time.sleep(0.09)
  calls = [c.args for c in pca.set_servo_angle.call_args_list]
  assert calls[0] == (15, 25.0)
  time.sleep(0.22)
  calls = [c.args for c in pca.set_servo_angle.call_args_list]
  assert (15, 0.0) in calls
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
  playback.advance(5000)
  mouth.pump_timeline(playback)
  pca.set_servo_angle.assert_any_call(15, 25.0)
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


def test_word_mode_follows_chunk_clock() -> None:
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
  from puppet.tts.types import WordCue

  mouth.play_word_chunk([WordCue(0, 50), WordCue(100, 200)], playback, chunk_start_samples=0)
  playback.advance(1)
  time.sleep(0.05)
  pca.set_servo_angle.assert_any_call(15, 25.0)
  playback.advance(5000)
  time.sleep(0.05)
  calls = [c.args for c in pca.set_servo_angle.call_args_list]
  assert (15, 0.0) in calls
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
  from puppet.tts.types import WordCue

  mouth.play_word_chunk([WordCue(0, 50)], playback, chunk_start_samples=0)
  playback.advance(1000)
  time.sleep(0.05)
  pca.set_servo_angle.assert_not_called()
  playback.advance(5000)
  time.sleep(0.05)
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
  from puppet.tts.types import WordCue

  mouth.play_word_chunk([WordCue(0, 40)], playback, chunk_start_samples=0)
  mouth.play_word_chunk([WordCue(0, 40)], playback, chunk_start_samples=1000)
  playback.advance(5000)
  time.sleep(0.05)
  pca.set_servo_angle.assert_any_call(15, 25.0)
  mouth.reset()


def test_close_opens_mouth_on_shutdown() -> None:
  pca = MagicMock()
  mouth = PhonemeMouth(pca, channel=15, closed_deg=20.0, open_deg=0.0, mode="word")
  mouth.close_for_listen()
  pca.reset_mock()
  mouth.close()
  pca.set_servo_angle.assert_called_once_with(15, 0.0)
  mouth.close()
  pca.set_servo_angle.assert_called_once()


def test_create_mouth_disabled_by_default() -> None:
  from puppet.hardware.mouth import NullMouth, create_mouth

  mouth = create_mouth({"puppet": {"mouth": {"enabled": False}}})
  assert isinstance(mouth, NullMouth)
