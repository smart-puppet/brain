import numpy as np

from puppet.core.audio.buffer import AudioReference, RingBuffer


def test_ring_buffer_read_latest() -> None:
  buf = RingBuffer(capacity_samples=10)
  buf.write(np.array([1.0, 2.0, 3.0], dtype=np.float32))
  latest = buf.read_latest(2)
  assert latest.tolist() == [2.0, 3.0]


def test_audio_reference_clear() -> None:
  ref = AudioReference(sample_rate=16000, max_seconds=1.0)
  ref.write(np.ones(100, dtype=np.float32))
  ref.clear()
  assert ref.recent_rms() == 0.0
  assert np.all(ref.read_aligned(10) == 0.0)
