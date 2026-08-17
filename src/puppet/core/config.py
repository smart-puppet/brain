from __future__ import annotations

import os
import re
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
LANGUAGE_CODES = ("en", "fr", "de")
DEFAULT_LANGUAGE = "de_1"
LANGUAGE_ACTIVE_FILE = "language.active"
_LANG_ID_RE = re.compile(r"^(en|fr|de)(?:_([1-9]\d*))?$")
_LOCALE_ORDER = {"en": 0, "fr": 1, "de": 2}


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
    if not isinstance(wrapped, dict):
      loc = language_locale(code)
      wrapped = data.get(loc) if loc != code else None
    if isinstance(wrapped, dict) and len(data) == 1:
      return wrapped
    return data
  if not isinstance(profiles, dict):
    return None
  code = path.stem.lower()
  if code in profiles and isinstance(profiles[code], dict):
    return profiles[code]
  loc = language_locale(code)
  if loc in profiles and isinstance(profiles[loc], dict):
    return profiles[loc]
  if len(profiles) == 1:
    only = next(iter(profiles.values()))
    return only if isinstance(only, dict) else None
  return None


def local_language_dir(config_path: Path) -> Path:
  """``../brain`` next to the config directory (local overlays, not git)."""
  return (Path(config_path) / ".." / "brain").resolve()


def local_german_overlay_path(config_path: Path) -> Path:
  """Deprecated path: unnumbered ``../brain/de.yaml`` (loaded as the next de_N)."""
  return local_language_dir(config_path) / "de.yaml"


def _split_stem(stem: str) -> tuple[str, int | None] | None:
  m = _LANG_ID_RE.match((stem or "").strip().lower())
  if not m:
    return None
  num = int(m.group(2)) if m.group(2) else None
  return m.group(1), num


def _next_overlay_id(locale: str, used: set[str]) -> str:
  n = 2
  while f"{locale}_{n}" in used:
    n += 1
  return f"{locale}_{n}"


def iter_language_profile_files(config_path: Path) -> list[tuple[str, Path, bool]]:
  """Return (id, path, overlay) for stock ``language/`` then local ``../brain`` overlays."""
  config_path = Path(config_path)
  found: dict[str, tuple[Path, bool]] = {}

  def _take(lang_dir: Path, overlay: bool) -> None:
    if not lang_dir.is_dir():
      return
    numbered: list[tuple[str, Path]] = []
    unnumbered: list[tuple[str, Path]] = []
    for path in sorted(lang_dir.glob("*.yaml")):
      parsed = _split_stem(path.stem)
      if parsed is None:
        continue
      locale, num = parsed
      if num is None:
        unnumbered.append((locale, path))
      else:
        numbered.append((f"{locale}_{num}", path))
    for pid, path in numbered:
      found[pid] = (path, overlay)
    for locale, path in unnumbered:
      pid = _next_overlay_id(locale, set(found)) if overlay else f"{locale}_1"
      if pid not in found:
        found[pid] = (path, overlay)

  _take(config_path / "language", overlay=False)
  _take(local_language_dir(config_path), overlay=True)
  items = [(pid, path, overlay) for pid, (path, overlay) in found.items()]
  items.sort(
    key=lambda item: (
      _LOCALE_ORDER.get(language_locale(item[0]), 9),
      split_language_id(item[0])[1],
    )
  )
  return items


def _load_language_dir(config_path: Path) -> dict[str, Any]:
  """Stock ``config/language/en_1.yaml`` plus local overlays such as ``../brain/de_2.yaml``."""
  profiles: dict[str, Any] = {}
  for pid, path, _overlay in iter_language_profile_files(config_path):
    profile = _profile_from_file(path)
    if profile:
      profiles[pid] = profile
  if not profiles:
    return {}
  return {"language": {"profiles": profiles}}


def list_language_profiles(config_dir: str | Path) -> list[dict[str, Any]]:
  """Profiles Eye can show: stock *_1 plus any numbered overlays that exist."""
  out: list[dict[str, Any]] = []
  for pid, path, overlay in iter_language_profile_files(Path(config_dir)):
    profile = _profile_from_file(path) or {}
    loc, num = split_language_id(pid)
    label = str(profile.get("label") or "").strip() or pid
    out.append(
      {
        "id": pid,
        "label": label,
        "locale": loc,
        "index": num,
        "overlay": overlay,
        "file": str(path),
        "button": f"{loc.upper()} {num}",
      }
    )
  return out


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


_READY_LISTEN_PROMPTS: dict[str, str] = {
  "en": "Hello! I'm Kace, and I'm ready to listen!",
  "fr": "Coucou ! Je suis Kace, je suis prêt à t'écouter !",
  "de": "Hallo! Ich bin Kace und bereit zuzuhören!",
}


def language_locale(code: str) -> str:
  """``de_2`` → ``de``. Unknown values keep the first two letters."""
  raw = (code or "").strip().lower()
  m = _LANG_ID_RE.match(raw)
  if m:
    return m.group(1)
  return raw[:2] if raw else "en"


def split_language_id(code: str) -> tuple[str, int]:
  raw = (code or "").strip().lower()
  m = _LANG_ID_RE.match(raw)
  if not m:
    return language_locale(raw), 1
  return m.group(1), int(m.group(2) or 1)


def parse_language_id(text: str) -> str | None:
  """Canonical profile id: ``en_1``, ``de_2``. Bare ``en``/``fr``/``de`` means ``*_1``."""
  raw = (text or "").strip().lower().strip("'\"")
  m = _LANG_ID_RE.match(raw)
  if not m:
    return None
  return f"{m.group(1)}_{m.group(2) or 1}"


def parse_language_code(text: str) -> str | None:
  """Canonical id from a one-line file or tiny YAML blob."""
  raw = (text or "").strip()
  if not raw:
    return None
  first = raw.splitlines()[0].strip().lower().strip("'\"")
  parsed = parse_language_id(first)
  if parsed:
    return parsed
  try:
    data = yaml.safe_load(raw)
  except yaml.YAMLError:
    return None
  if isinstance(data, str):
    return parse_language_id(data)
  if isinstance(data, dict):
    nested = data.get("language")
    active = None
    if isinstance(nested, dict):
      active = nested.get("active")
    if active is None:
      active = data.get("active")
    return parse_language_id(str(active or ""))
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
  code = parse_language_id(str(language or ""))
  if not code:
    raise ValueError("language must look like en_1, fr_1, de_1, or de_2")
  path = language_active_path(config_dir)
  path.parent.mkdir(parents=True, exist_ok=True)
  tmp = path.with_name(path.name + ".tmp")
  tmp.write_text(code + "\n", encoding="utf-8")
  tmp.replace(path)
  return path


def _canonicalize_profiles(profiles: dict[str, Any] | None) -> dict[str, Any]:
  out: dict[str, Any] = {}
  for key, value in (profiles or {}).items():
    if not isinstance(value, dict):
      continue
    pid = parse_language_id(str(key))
    if pid:
      out[pid] = value
  return out


def resolve_active_language(active: str | None, profiles: dict[str, Any]) -> str:
  pid = parse_language_id(str(active or "")) or str(active or "").strip().lower()
  if pid in profiles:
    return pid
  loc = language_locale(pid)
  fallback = f"{loc}_1"
  if fallback in profiles:
    return fallback
  if DEFAULT_LANGUAGE in profiles:
    return DEFAULT_LANGUAGE
  known = ", ".join(sorted(profiles)) or "(none)"
  raise ValueError(f"Unknown language profile '{active}'. Known profiles: {known}")


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
  return _READY_LISTEN_PROMPTS.get(language_locale(active), _READY_LISTEN_PROMPTS["en"])


def apply_language_profile(config: dict[str, Any]) -> dict[str, Any]:
  """Apply the active language profile to stt, tts, and llm sections."""
  lang_cfg = config.setdefault("language", {})
  profiles = _canonicalize_profiles(lang_cfg.get("profiles"))
  lang_cfg["profiles"] = profiles
  active = resolve_active_language(lang_cfg.get("active"), profiles)
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

  lang_cfg = merged.setdefault("language", {})
  lang_cfg["profiles"] = _canonicalize_profiles(lang_cfg.get("profiles"))
  # language.active (Eye) wins over language.yaml; missing file → German stock.
  file_lang = read_language_active_file(config_path)
  if file_lang:
    lang_cfg["active"] = file_lang
  elif DEFAULT_LANGUAGE in lang_cfg["profiles"]:
    lang_cfg["active"] = DEFAULT_LANGUAGE

  merged = _deep_merge(merged, _env_overrides())
  if language:
    merged.setdefault("language", {})["active"] = language
  merged = apply_language_profile(merged)
  return merged
