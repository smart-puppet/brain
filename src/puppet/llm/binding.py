from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

KNOWN_BINDINGS = frozenset({"upstream", "prism"})
BUILD_SCRIPTS = {
  "upstream": "./scripts/build_llama.sh",
  "prism": "./scripts/build_llama_prism.sh",
}


def find_repo_root() -> Path:
  here = Path(__file__).resolve()
  for parent in here.parents:
    if (parent / "pyproject.toml").is_file():
      return parent
  return Path.cwd()


def binding_lock_path(repo_root: Path | None = None) -> Path:
  root = repo_root or find_repo_root()
  return root / "vendor/native/llama-binding.lock"


def read_installed_binding(repo_root: Path | None = None) -> str | None:
  path = binding_lock_path(repo_root)
  if not path.is_file():
    return None
  for line in path.read_text(encoding="utf-8").splitlines():
    line = line.strip()
    if line.startswith("#") or "=" not in line:
      continue
    key, value = line.split("=", 1)
    if key.strip() == "LLAMA_BINDING":
      binding = value.strip()
      return binding if binding in KNOWN_BINDINGS else None
  return None


def validate_llama_binding(config: dict, *, repo_root: Path | None = None) -> None:
  """Ensure llm.binding matches the last scripts/build_llama*.sh install."""
  llm_cfg = config.get("llm", {})
  wanted = str(llm_cfg.get("binding", "upstream")).lower()
  if wanted not in KNOWN_BINDINGS:
    known = ", ".join(sorted(KNOWN_BINDINGS))
    raise ValueError(f"llm.binding must be one of: {known}, got {wanted!r}")

  installed = read_installed_binding(repo_root)
  if installed is None:
    logger.warning(
      "No llama binding stamp at vendor/native/llama-binding.lock — "
      "run %s before starting Puppet",
      BUILD_SCRIPTS.get(wanted, "./scripts/build_llama.sh"),
    )
    return

  if installed == wanted:
    return

  raise RuntimeError(
    f"llm.binding is {wanted!r} but the active venv was built for {installed!r}. "
    f"Reinstall with {BUILD_SCRIPTS[wanted]} "
    f"(only one llama-cpp-python build can be active in .venv at a time)."
  )
