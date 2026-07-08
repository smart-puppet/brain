from __future__ import annotations

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
  sys.path.insert(0, str(ROOT / "src"))

from puppet.core import load_config  # noqa: E402


def configure_logging(config: dict, *, trace: bool = False) -> None:
  log_cfg = config.get("logging", {})
  level = log_cfg.get("level", "INFO")
  logging.basicConfig(
    level=getattr(logging, str(level).upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    force=True,
  )
  for name, logger_level in log_cfg.get("loggers", {}).items():
    logging.getLogger(name).setLevel(
      getattr(logging, str(logger_level).upper(), logging.DEBUG)
    )
  if trace:
    logging.getLogger("puppet.trace").setLevel(logging.DEBUG)


def load_puppet_config(config_dir: str, *, language: str | None = None) -> dict:
  return load_config(ROOT / config_dir, language=language)
