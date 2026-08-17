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
  "puppet.yaml",
)

_KNOWN_PROFILES = ("respeaker", "regular-mic")


def _load_yaml_file(path: Path) -> dict[str, Any]:
  with path.open(encoding="utf-8") as fh:
    data = yaml.safe_load(fh) or {}
  return data if isinstance(data, dict) else {}


def _profile_from_file(path: Path) -> dict[str, Any] | None:
  """Return one language profile from a YAML file, or None if empty."""
  data = _load_yaml_file(path)
  if not data:
    return None
  nested = data.get("language") if isinstance(data.get("language"), dict) else None
  if isinstance(data.get("profiles"), dict):
    profiles = data["profiles"]
  elif nested and isinstance(nested.get("profiles"), dict):
    profiles = nested["profiles"]
  else:
    code = path.stem.lower()
    wrapped = data.get(code)
    if isinstance(wrapped, dict) and set(data.keys()) == {code}:
      return wrapped
    return data
  if not isinstance(profiles, dict):
    return None
  code = path.stem.lower()
  if code in profiles and isinstance(profiles[code], dict):
    return profiles[code]
  if len(profiles) == 1:
    only = next(iter(profiles.values()))
    return only if isinstance(only, dict) else None
  return None


def _profiles_from_dir(lang_dir: Path) -> dict[str, Any]:
  """Load ``<code>.yaml`` files in a directory into a profiles map."""
  if not lang_dir.is_dir():
    return {}
  profiles: dict[str, Any] = {}
  for path in sorted(lang_dir.glob("*.yaml")):
    profile = _profile_from_file(path)
    if profile:
      profiles[path.stem.lower()] = profile
  return profiles


def local_language_dir(config_path: Path) -> Path:
  """``../brain`` next to the config directory (local overlays, not git)."""
  return (Path(config_path) / ".." / "brain").resolve()


def local_german_overlay_path(config_path: Path) -> Path:
  """Optional ``../brain/de.yaml``. Missing → stock German in config/language."""
  return local_language_dir(config_path) / "de.yaml"


def _load_language_dir(config_path: Path) -> dict[str, Any]:
  """Stock ``config/language/``; German uses ``../brain/de.yaml`` only if that file exists."""
  profiles = _profiles_from_dir(config_path / "language")
  overlay = local_german_overlay_path(config_path)
  if overlay.is_file():
    local_de = _profile_from_file(overlay)
    if local_de:
      profiles["de"] = local_de
  if not profiles:
    return {}
  return {"language": {"profiles": profiles}}


def _load_profile(config_path: Path, profile: str) -> dict[str, Any]:
  if profile not in _KNOWN_PROFILES:
    known = ", ".join(_KNOWN_PROFILES)
    raise ValueError(f"Unknown config profile {profile!r}. Known profiles: {known}")
  path = config_path / "profiles" / f"{profile}.yaml"
  if not path.is_file():
    raise FileNotFoundError(f"Profile file not found: {path}")
  with path.open(encoding="utf-8") as fh:
    return yaml.safe_load(fh) or {}


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


LANGUAGE_CODES = ("en", "fr", "de")
DEFAULT_LANGUAGE = "de"
LANGUAGE_ACTIVE_FILE = "language.active"

_READY_LISTEN_PROMPTS: dict[str, str] = {
  "en": "Hello! I'm Kace, and I'm ready to listen!",
  "fr": "Coucou ! Je suis Kace, je suis prêt à t'écouter !",
  "de": "Hallo! Ich bin Kace und bereit zuzuhören!",
}


def parse_language_code(text: str) -> str | None:
  """Return en|fr|de from a one-line file or tiny YAML blob."""
  raw = (text or "").strip()
  if not raw:
    return None
  first = raw.splitlines()[0].strip().lower().strip("'\"")
  if first in LANGUAGE_CODES:
    return first
  try:
    data = yaml.safe_load(raw)
  except yaml.YAMLError:
    return None
  if isinstance(data, str) and data.strip().lower() in LANGUAGE_CODES:
    return data.strip().lower()
  if isinstance(data, dict):
    nested = data.get("language")
    active = None
    if isinstance(nested, dict):
      active = nested.get("active")
    if active is None:
      active = data.get("active")
    code = str(active or "").strip().lower()
    if code in LANGUAGE_CODES:
      return code
  return None


def language_active_path(config_dir: str | Path) -> Path:
  return Path(config_dir) / LANGUAGE_ACTIVE_FILE


def read_language_active_file(config_dir: str | Path) -> str | None:
  path = language_active_path(config_dir)
  if not path.is_file():
    return None
  try:
    return parse_language_code(path.read_text(encoding="utf-8"))
  except OSError:
    return None


def write_language_active_file(config_dir: str | Path, language: str) -> Path:
  code = str(language or "").strip().lower()
  if code not in LANGUAGE_CODES:
    known = ", ".join(LANGUAGE_CODES)
    raise ValueError(f"language must be one of: {known}")
  path = language_active_path(config_dir)
  path.parent.mkdir(parents=True, exist_ok=True)
  tmp = path.with_name(path.name + ".tmp")
  tmp.write_text(code + "\n", encoding="utf-8")
  tmp.replace(path)
  return path


def get_ready_listen_prompt(config: dict[str, Any]) -> str:
  """Localized phrase spoken when Puppet enters streaming listen mode."""
  lang_cfg = config.get("language", {})
  active = str(lang_cfg.get("active", "en"))
  profiles = lang_cfg.get("profiles", {})
  profile = profiles.get(active, {})
  if not isinstance(profile, dict):
    profile = {}
  prompt = profile.get("ready_listen_prompt")
  if isinstance(prompt, str) and prompt.strip():
    return prompt.strip()
  return _READY_LISTEN_PROMPTS.get(active, _READY_LISTEN_PROMPTS["en"])


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

  merged = _deep_merge(merged, _load_language_dir(config_path))

  profile = merged.pop("profile", None)
  if profile is not None:
    merged = _deep_merge(merged, _load_profile(config_path, str(profile)))

  # language.active (Eye) wins over language.yaml; missing file → German.
  file_lang = read_language_active_file(config_path)
  profiles = (merged.get("language") or {}).get("profiles") or {}
  if file_lang:
    merged.setdefault("language", {})["active"] = file_lang
  elif DEFAULT_LANGUAGE in profiles:
    merged.setdefault("language", {})["active"] = DEFAULT_LANGUAGE

  merged = _deep_merge(merged, _env_overrides())
  if language:
    merged.setdefault("language", {})["active"] = language
  merged = apply_language_profile(merged)
  return merged
