import threading
import time

from puppet.core.audio.capture import AudioPlayback


def test_playback_position_advances_with_real_time(monkeypatch) -> None:
  playback = AudioPlayback.__new__(AudioPlayback)
  playback.sample_rate = 22050
  playback.channels = 1
  playback._frame_bytes = 2
  playback._write_chunk_bytes = 4096
  playback._write_buffer = bytearray()
  playback._lock = threading.Lock()
  playback._sample_clock = threading.Condition(playback._lock)
  playback._samples_written = 0
  playback._stream_anchor_time = None
  playback._stream_anchor_samples = 0
  playback._aborted = False
  playback._stream = monkeypatch  # unused

  with playback._lock:
    playback._stream_anchor_time = time.monotonic()
    playback._stream_anchor_samples = 0
    playback._samples_written = 100000

  time.sleep(0.1)
  pos = playback.playback_position_samples()
  assert 1500 <= pos <= 3500


def test_wait_until_samples_uses_playback_position(monkeypatch) -> None:
  playback = AudioPlayback.__new__(AudioPlayback)
  playback.sample_rate = 22050
  playback.channels = 1
  playback._frame_bytes = 2
  playback._aborted = False
  playback._stream = monkeypatch
  playback._lock = threading.Lock()
  playback._sample_clock = threading.Condition(playback._lock)
  playback._samples_written = 100000
  playback._stream_anchor_time = time.monotonic()
  playback._stream_anchor_samples = 0

  def advance_time() -> None:
    time.sleep(0.08)
    with playback._lock:
      playback._stream_anchor_time = time.monotonic() - 0.08

  threading.Thread(target=advance_time, daemon=True).start()
  assert playback.wait_until_samples(1500, timeout=1.0)
