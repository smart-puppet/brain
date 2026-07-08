from __future__ import annotations

from unittest.mock import MagicMock

from puppet.tts.piper import _load_piper_voice


def test_load_piper_voice_sets_onnx_threads(tmp_path, monkeypatch) -> None:
  model_path = tmp_path / "voice.onnx"
  config_path = tmp_path / "voice.onnx.json"
  model_path.write_bytes(b"onnx")
  config_path.write_text("{}", encoding="utf-8")

  captured: dict[str, int] = {}

  class FakeSessionOptions:
    def __init__(self) -> None:
      self.intra_op_num_threads = 0
      self.inter_op_num_threads = 0

  class FakeSession:
    def __init__(self, model: str, sess_options=None, providers=None) -> None:
      assert providers == ["CPUExecutionProvider"]
      captured["intra"] = sess_options.intra_op_num_threads
      captured["inter"] = sess_options.inter_op_num_threads

  monkeypatch.setattr("puppet.tts.piper.PiperConfig.from_dict", lambda _cfg: MagicMock())
  monkeypatch.setattr("puppet.tts.piper.onnxruntime.SessionOptions", FakeSessionOptions)
  monkeypatch.setattr("puppet.tts.piper.onnxruntime.InferenceSession", FakeSession)

  voice = _load_piper_voice(
    str(model_path),
    str(config_path),
    use_cuda=False,
    n_threads=3,
  )
  assert voice is not None
  assert captured == {"intra": 3, "inter": 3}


def test_load_piper_voice_leaves_default_threads_when_zero(tmp_path, monkeypatch) -> None:
  model_path = tmp_path / "voice.onnx"
  config_path = tmp_path / "voice.onnx.json"
  model_path.write_bytes(b"onnx")
  config_path.write_text("{}", encoding="utf-8")

  captured: dict[str, int] = {}

  class FakeSessionOptions:
    def __init__(self) -> None:
      self.intra_op_num_threads = 7
      self.inter_op_num_threads = 7

  class FakeSession:
    def __init__(self, model: str, sess_options=None, providers=None) -> None:
      captured["intra"] = sess_options.intra_op_num_threads
      captured["inter"] = sess_options.inter_op_num_threads

  monkeypatch.setattr("puppet.tts.piper.PiperConfig.from_dict", lambda _cfg: MagicMock())
  monkeypatch.setattr("puppet.tts.piper.onnxruntime.SessionOptions", FakeSessionOptions)
  monkeypatch.setattr("puppet.tts.piper.onnxruntime.InferenceSession", FakeSession)

  _load_piper_voice(
    str(model_path),
    str(config_path),
    use_cuda=False,
    n_threads=0,
  )
  assert captured == {"intra": 7, "inter": 7}
