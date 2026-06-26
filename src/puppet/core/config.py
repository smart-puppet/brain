from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

_CONFIG_FILES = (
  "default.yaml",
  "language.yaml",
  "stt.yaml",
  "llm.yaml",
  "tts.yaml",
  "vad.yaml",
  "aec.yaml",
  "puppet.yaml",
)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
  result = dict(base)
  for key, value in override.items():
    if key in result and isinstance(result[key], dict) and isinstance(value, dict):
      result[key] = _deep_merge(result[key], value)
    else:
      result[key] = value
  return result


def _env_overrides(prefix: str = "PUPPET_") -> dict[str, Any]:
  """Map PUPPET_STT__MODEL_PATH to {"stt": {"model_path": "..."}}."""
  nested: dict[str, Any] = {}
  for key, value in os.environ.items():
    if not key.startswith(prefix):
      continue
    parts = key[len(prefix) :].lower().split("__")
    cursor = nested
    for part in parts[:-1]:
      cursor = cursor.setdefault(part, {})
    cursor[parts[-1]] = value
  return nested


def apply_language_profile(config: dict[str, Any]) -> dict[str, Any]:
  """Apply the active language profile to stt, tts, and llm sections."""
  lang_cfg = config.get("language", {})
  active = lang_cfg.get("active", "en")
  profiles = lang_cfg.get("profiles", {})
  if active not in profiles:
    known = ", ".join(sorted(profiles)) or "(none)"
    raise ValueError(f"Unknown language profile '{active}'. Known profiles: {known}")

  profile = profiles[active]
  config.setdefault("stt", {})
  config.setdefault("tts", {})
  config.setdefault("llm", {})

  if "stt_language" in profile:
    config["stt"]["language"] = profile["stt_language"]
  if "tts_model_path" in profile:
    config["tts"]["model_path"] = profile["tts_model_path"]
  if "tts_config_path" in profile:
    config["tts"]["config_path"] = profile["tts_config_path"]
  if "system_prompt" in profile:
    config["llm"]["system_prompt"] = profile["system_prompt"].strip()

  config["language"]["active"] = active
  return config


def load_config(config_dir: str | Path, *, language: str | None = None) -> dict[str, Any]:
  config_path = Path(config_dir)
  if not config_path.is_dir():
    raise FileNotFoundError(f"Config directory not found: {config_path}")

  merged: dict[str, Any] = {}
  for name in _CONFIG_FILES:
    path = config_path / name
    if not path.is_file():
      continue
    with path.open(encoding="utf-8") as fh:
      merged = _deep_merge(merged, yaml.safe_load(fh) or {})

  merged = _deep_merge(merged, _env_overrides())
  if language:
    merged.setdefault("language", {})["active"] = language
  merged = apply_language_profile(merged)
  return merged
