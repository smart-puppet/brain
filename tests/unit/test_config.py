from __future__ import annotations

from pathlib import Path

import pytest

from puppet.core.config import apply_language_profile, get_ready_listen_prompt, load_config


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


def test_local_brain_overlay_is_extra_profile(tmp_path: Path) -> None:
  config_dir = tmp_path / "config"
  lang_dir = config_dir / "language"
  overlay = tmp_path / "brain"
  lang_dir.mkdir(parents=True)
  overlay.mkdir()
  (lang_dir / "en.yaml").write_text(
    "label: English\nstt_language: en-US\nsystem_prompt: Hello\n"
  )
  (lang_dir / "de.yaml").write_text(
    "label: Stock\nstt_language: de-DE\nsystem_prompt: Kace\n"
  )
  (overlay / "de.yaml").write_text(
    "de:\n  label: Deutsch\n  stt_language: de-DE\n  system_prompt: Overlay\n"
  )
  cfg = load_config(config_dir)
  assert cfg["language"]["active"] == "de_1"
  assert cfg["llm"]["system_prompt"] == "Kace"
  assert cfg["language"]["profiles"]["de_1"]["system_prompt"] == "Kace"
  assert cfg["language"]["profiles"]["de_2"]["system_prompt"] == "Overlay"
  assert cfg["language"]["profiles"]["en_1"]["system_prompt"] == "Hello"


def test_numbered_overlay_de_2_is_selectable(tmp_path: Path) -> None:
  from puppet.core.config import list_language_profiles

  config_dir = tmp_path / "config"
  lang_dir = config_dir / "language"
  overlay = tmp_path / "brain"
  lang_dir.mkdir(parents=True)
  overlay.mkdir()
  (lang_dir / "de_1.yaml").write_text(
    "label: Deutsch\nstt_language: de-DE\nsystem_prompt: Kace\n"
  )
  (overlay / "de_2.yaml").write_text(
    "label: Deutsch Maelie\nstt_language: de-DE\nsystem_prompt: Maelie\n"
  )
  (config_dir / "language.active").write_text("de_2\n")
  cfg = load_config(config_dir)
  assert cfg["language"]["active"] == "de_2"
  assert cfg["llm"]["system_prompt"] == "Maelie"
  ids = [p["id"] for p in list_language_profiles(config_dir)]
  assert ids == ["de_1", "de_2"]
  assert list_language_profiles(config_dir)[1]["overlay"] is True
  assert list_language_profiles(config_dir)[1]["button"] == "DE 2"


def test_missing_local_brain_de_falls_back_to_config(tmp_path: Path) -> None:
  config_dir = tmp_path / "config"
  lang_dir = config_dir / "language"
  lang_dir.mkdir(parents=True)
  (lang_dir / "de.yaml").write_text(
    "label: Deutsch\nstt_language: de-DE\nsystem_prompt: Kace\n"
  )
  cfg = load_config(config_dir)
  assert cfg["llm"]["system_prompt"] == "Kace"


def test_language_dir_profiles_load(tmp_path: Path) -> None:
  lang_dir = tmp_path / "language"
  lang_dir.mkdir()
  (lang_dir / "en.yaml").write_text(
    "label: English\nstt_language: en-US\nsystem_prompt: Hello\n"
  )
  (lang_dir / "de.yaml").write_text(
    "label: Deutsch\nstt_language: de\nsystem_prompt: Hallo\n"
  )
  cfg = load_config(tmp_path)
  assert cfg["language"]["active"] == "de_1"
  assert cfg["stt"]["language"] == "de"
  assert cfg["llm"]["system_prompt"] == "Hallo"


def test_language_active_file_overrides_yaml(tmp_path: Path) -> None:
  (tmp_path / "language.yaml").write_text(
    "language:\n  active: en\n  profiles:\n"
    "    en:\n      stt_language: en-US\n"
    "    de:\n      stt_language: de\n"
    "    fr:\n      stt_language: fr\n"
  )
  (tmp_path / "language.active").write_text("fr\n")
  cfg = load_config(tmp_path)
  assert cfg["language"]["active"] == "fr_1"
  assert cfg["stt"]["language"] == "fr"


def test_missing_language_active_defaults_to_german(tmp_path: Path) -> None:
  (tmp_path / "language.yaml").write_text(
    "language:\n  active: en\n  profiles:\n"
    "    en:\n      stt_language: en-US\n"
    "    de:\n      stt_language: de\n"
  )
  cfg = load_config(tmp_path)
  assert cfg["language"]["active"] == "de_1"
  assert cfg["stt"]["language"] == "de"


def test_cli_language_beats_active_file(tmp_path: Path) -> None:
  (tmp_path / "language.yaml").write_text(
    "language:\n  active: en\n  profiles:\n"
    "    en:\n      stt_language: en-US\n      system_prompt: English\n"
    "    fr:\n      stt_language: fr\n      system_prompt: Français\n"
    "    de:\n      stt_language: de\n"
  )
  (tmp_path / "language.active").write_text("de\n")
  cfg = load_config(tmp_path, language="fr")
  assert cfg["language"]["active"] == "fr_1"
  assert cfg["stt"]["language"] == "fr"


def test_language_profile_cli_override(tmp_path: Path) -> None:
  (tmp_path / "language.yaml").write_text(
    "language:\n  active: en\n  profiles:\n"
    "    en:\n      stt_language: en-US\n      system_prompt: English\n"
    "    fr:\n      stt_language: fr\n      system_prompt: Français\n"
  )
  cfg = load_config(tmp_path, language="fr")
  assert cfg["language"]["active"] == "fr_1"
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


def test_mic_profile_respeaker(tmp_path: Path) -> None:
  (tmp_path / "default.yaml").write_text("profile: respeaker\n")
  profiles = tmp_path / "profiles"
  profiles.mkdir()
  (profiles / "respeaker.yaml").write_text(
    "vad:\n  gate_stt: false\npuppet:\n  barge_in_enabled: false\n"
  )
  (tmp_path / "language.yaml").write_text(
    "language:\n  active: en\n  profiles:\n    en:\n      stt_language: en-US\n"
  )
  cfg = load_config(tmp_path)
  assert "profile" not in cfg
  assert cfg["vad"]["gate_stt"] is False
  assert cfg["puppet"]["barge_in_enabled"] is False


def test_unknown_mic_profile(tmp_path: Path) -> None:
  (tmp_path / "default.yaml").write_text("profile: unknown\n")
  (tmp_path / "language.yaml").write_text(
    "language:\n  active: en\n  profiles:\n    en:\n      stt_language: en-US\n"
  )
  with pytest.raises(ValueError, match="Unknown config profile"):
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
  assert cfg["language"]["active"] == "de_1"


def test_ready_listen_prompt_from_profile() -> None:
  cfg = {
    "language": {
      "active": "fr",
      "profiles": {
        "fr": {"ready_listen_prompt": "  Prêt !  "},
      },
    },
  }
  assert get_ready_listen_prompt(cfg) == "Prêt !"


def test_ready_listen_prompt_fallback_by_language() -> None:
  cfg = {"language": {"active": "de", "profiles": {"de": {}}}}
  assert "Kace" in get_ready_listen_prompt(cfg)
  assert get_ready_listen_prompt({"language": {"active": "xx", "profiles": {}}}).startswith("Hello")
