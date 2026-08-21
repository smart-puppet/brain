import numpy as np

from puppet.core.audio.pcm import int16_mono_to_device, prepend_lead_in_silence, resample_linear
from puppet.core.audio.buffer import RingBuffer


def test_ring_buffer_read_delayed() -> None:
  buf = RingBuffer(capacity_samples=20)
  buf.write(np.arange(10, dtype=np.float32))
  delayed = buf.read_delayed(4, delay_samples=2)
  assert delayed.tolist() == [4.0, 5.0, 6.0, 7.0]


def test_resample_linear() -> None:
  src = np.array([0.0, 1.0], dtype=np.float32)
  out = resample_linear(src, 100, 200)
  assert out.size == 4
  assert out[0] == 0.0
  assert out[-1] == 1.0


def test_prepend_lead_in_silence() -> None:
  chunk = np.ones(100, dtype=np.float32)
  out = prepend_lead_in_silence(chunk, 22050, 100)
  assert out.size == 100 + 2205
  assert np.all(out[:2205] == 0.0)
  assert np.all(out[2205:] == 1.0)


def test_int16_mono_to_respeaker_stereo_16k() -> None:
  src = np.array([1000, -1000, 2000, -2000], dtype=np.int16).tobytes()
  same_rate = np.frombuffer(
    int16_mono_to_device(src, src_rate=16000, dst_rate=16000, dst_channels=2),
    dtype=np.int16,
  )
  assert same_rate.tolist() == [1000, 1000, -1000, -1000, 2000, 2000, -2000, -2000]
  down = np.frombuffer(
    int16_mono_to_device(src, src_rate=22050, dst_rate=16000, dst_channels=2),
    dtype=np.int16,
  )
  assert down.size % 2 == 0
  assert down.size < same_rate.size
  assert down[0] == down[1]
