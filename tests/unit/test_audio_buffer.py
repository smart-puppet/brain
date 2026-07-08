import numpy as np

from puppet.core.audio.buffer import RingBuffer


def test_ring_buffer_read_latest() -> None:
  buf = RingBuffer(capacity_samples=10)
  buf.write(np.array([1.0, 2.0, 3.0], dtype=np.float32))
  latest = buf.read_latest(2)
  assert latest.tolist() == [2.0, 3.0]
