from __future__ import annotations

import argparse
import logging
import sys

from puppet.core import load_config
from puppet.orchestrator import Orchestrator


def _configure_logging(config: dict) -> None:
  log_cfg = config.get("logging", {})
  level = log_cfg.get("level", "INFO")
  level_no = getattr(logging, str(level).upper(), logging.INFO)
  logging.basicConfig(
    level=level_no,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
  )
  for name, logger_level in log_cfg.get("loggers", {}).items():
    logging.getLogger(name).setLevel(
      getattr(logging, str(logger_level).upper(), logging.DEBUG)
    )


def main(argv: list[str] | None = None) -> int:
  parser = argparse.ArgumentParser(description="Puppet voice chatbot")
  parser.add_argument(
    "--config",
    default="config",
    help="Path to config directory (default: config/)",
  )
  parser.add_argument(
    "--language",
    "-l",
    choices=["en", "fr", "de"],
    help="Language profile (overrides config/language.yaml)",
  )
  parser.add_argument(
    "--once",
    type=float,
    metavar="SECONDS",
    help="Listen for N seconds then exit (for testing)",
  )
  args = parser.parse_args(argv)

  config = load_config(args.config, language=args.language)
  _configure_logging(config)

  orchestrator = Orchestrator(config)
  try:
    if args.once is not None:
      orchestrator.listen_once(duration_s=args.once)
    else:
      orchestrator.run()
  except Exception:
    logging.exception("Puppet failed")
    return 1
  finally:
    orchestrator.close()
  return 0


if __name__ == "__main__":
  sys.exit(main())
