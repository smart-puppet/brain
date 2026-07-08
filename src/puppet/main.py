from __future__ import annotations

import argparse
import logging
import signal
import sys

from puppet.core import load_config
from puppet.hardware.mouth_debug import configure_mouth_logging
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


def _install_shutdown_signals() -> None:
  """Run normal cleanup on SIGTERM (systemd, docker stop, kill) as well as Ctrl+C."""

  def _interrupt(signum: int, frame: object) -> None:
    del signum, frame
    raise KeyboardInterrupt

  signal.signal(signal.SIGTERM, _interrupt)


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
  parser.add_argument(
    "--mouth-debug",
    action="store_true",
    help="Log jaw servo timelines and moves (puppet.mouth logger)",
  )
  args = parser.parse_args(argv)

  config = load_config(args.config, language=args.language)
  if args.mouth_debug:
    config.setdefault("puppet", {}).setdefault("mouth", {})["debug"] = True
  _configure_logging(config)
  configure_mouth_logging(config)
  _install_shutdown_signals()

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
