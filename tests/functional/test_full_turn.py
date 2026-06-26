"""Full-turn functional tests (mocked backends; no hardware required)."""

from pathlib import Path

import numpy as np
import pytest

from puppet.core.audio.aec import resample_linear

FIXTURES = Path(__file__).parent / "fixtures"


def test_resample_linear_downsample() -> None:
  src = np.linspace(0.0, 1.0, num=22050, dtype=np.float32)
  dst = resample_linear(src, 22050, 16000)
  assert dst.size == 16000


@pytest.mark.skipif(
  not (FIXTURES / "sample_utterance.wav").is_file(),
  reason="Place sample_utterance.wav in tests/functional/fixtures/",
)
def test_wav_fixture_loadable() -> None:
  import wave

  with wave.open(str(FIXTURES / "sample_utterance.wav"), "rb") as wf:
    assert wf.getframerate() == 16000
    assert wf.getnchannels() == 1
