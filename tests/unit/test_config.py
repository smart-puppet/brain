from __future__ import annotations

from pathlib import Path

import pytest

from puppet.core.config import apply_language_profile, load_config


def test_load_config_merges_files(tmp_path: Path) -> None:
  (tmp_path / "default.yaml").write_text("audio:\n  sample_rate: 16000\n")
  (tmp_path / "stt.yaml").write_text("stt:\n  backend: parakeet\n")
  (tmp_path / "language.yaml").write_text(
    "language:\n  active: en\n  profiles:\n    en:\n      stt_language: en-US\n"
    "      tts_model_path: models/tts/en.onnx\n"
    "      tts_config_path: models/tts/en.onnx.json\n"
    "      system_prompt: Hello\n"
  )
  cfg = load_config(tmp_path)
  assert cfg["audio"]["sample_rate"] == 16000
  assert cfg["stt"]["backend"] == "parakeet"
  assert cfg["stt"]["language"] == "en-US"


def test_env_override(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
  (tmp_path / "default.yaml").write_text("stt:\n  model_path: a.gguf\n")
  (tmp_path / "language.yaml").write_text(
    "language:\n  active: en\n  profiles:\n    en:\n      stt_language: en-US\n"
  )
  monkeypatch.setenv("PUPPET_STT__MODEL_PATH", "b.gguf")
  cfg = load_config(tmp_path)
  assert cfg["stt"]["model_path"] == "b.gguf"


def test_language_profile_cli_override(tmp_path: Path) -> None:
  (tmp_path / "language.yaml").write_text(
    "language:\n  active: en\n  profiles:\n"
    "    en:\n      stt_language: en-US\n      system_prompt: English\n"
    "    fr:\n      stt_language: fr\n      system_prompt: Français\n"
  )
  cfg = load_config(tmp_path, language="fr")
  assert cfg["language"]["active"] == "fr"
  assert cfg["stt"]["language"] == "fr"
  assert cfg["llm"]["system_prompt"] == "Français"


def test_language_profile_env_override(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
  (tmp_path / "language.yaml").write_text(
    "language:\n  active: en\n  profiles:\n"
    "    en:\n      stt_language: en-US\n"
    "    de:\n      stt_language: de\n      tts_model_path: models/tts/de.onnx\n"
  )
  monkeypatch.setenv("PUPPET_LANGUAGE__ACTIVE", "de")
  cfg = load_config(tmp_path)
  assert cfg["stt"]["language"] == "de"
  assert cfg["tts"]["model_path"] == "models/tts/de.onnx"


def test_unknown_language_profile(tmp_path: Path) -> None:
  (tmp_path / "language.yaml").write_text("language:\n  active: xx\n  profiles:\n    en: {}\n")
  with pytest.raises(ValueError, match="Unknown language profile"):
    load_config(tmp_path)


def test_apply_language_profile_direct() -> None:
  cfg = {
    "language": {
      "active": "de",
      "profiles": {
        "de": {
          "stt_language": "de",
          "tts_model_path": "models/tts/de.onnx",
          "system_prompt": "  Deutsch  \n",
        },
      },
    },
  }
  apply_language_profile(cfg)
  assert cfg["stt"]["language"] == "de"
  assert cfg["llm"]["system_prompt"] == "Deutsch"
