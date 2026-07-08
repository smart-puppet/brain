#!/usr/bin/env python3
"""Live STT smoke test: speak into the mic, print streaming partials."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
  sys.path.insert(0, str(_SCRIPTS))

from _runtime import configure_logging, load_puppet_config
from puppet.core.audio import AudioCapture, create_vad, list_input_devices
from puppet.orchestrator.latency import TurnLatencyTracker
from puppet.stt import create_stt


def main(argv: list[str] | None = None) -> int:
  parser = argparse.ArgumentParser(description="Test STT only (mic → parakeet)")
  parser.add_argument("--config", default="config", help="Config directory")
  parser.add_argument("--language", "-l", choices=["en", "fr", "de"], default=None)
  parser.add_argument("--seconds", type=float, default=0.0, help="Stop after N seconds (0 = until Ctrl+C)")
  parser.add_argument("--device", type=int, default=None, help="Input device index")
  parser.add_argument("--list-devices", action="store_true")
  args = parser.parse_args(argv)

  if args.list_devices:
    for dev in list_input_devices():
      flag = "yes" if dev.supports_16k_int16 else "NO"
      print(f"[{dev.index}] {dev.name} (16k_int16={flag})")
    return 0

  config = load_puppet_config(args.config, language=args.language)
  configure_logging(config, trace=True)

  audio_cfg = config.get("audio", {})
  device = args.device if args.device is not None else audio_cfg.get("input_device")
  capture = AudioCapture(
    sample_rate=int(audio_cfg.get("sample_rate", 16000)),
    channels=int(audio_cfg.get("channels", 1)),
    chunk_ms=int(audio_cfg.get("chunk_ms", 20)),
    device_index=device,
  )
  stt = create_stt(config)
  vad = create_vad(config)
  latency = TurnLatencyTracker()
  vad_enabled = bool(config.get("vad", {}).get("enabled", True))
  print(f"Mic [{capture.device_index}] {capture.device_name!r} — speak (Ctrl+C to stop)")
  if vad_enabled:
    print(
      "Latency: pause speaking (~300ms silence); first partial AFTER each "
      "'speech end' line shows ms since VAD detected silence"
    )
  else:
    print("Note: VAD disabled in config — speech-end latency timing unavailable")
  parts: list[str] = []
  deadline = time.monotonic() + args.seconds if args.seconds > 0 else None
  try:
    while deadline is None or time.monotonic() < deadline:
      chunk = capture.read()
      for event in vad.feed(chunk.samples):
        if event.kind == "start":
          latency.clear_speech_window()
        elif event.kind == "end":
          latency.mark_vad_end()
          ahead_ms = latency.ms_since_last_stt_partial()
          if ahead_ms is not None and ahead_ms < 500:
            print(
              f"--- speech end --- "
              f"(last partial {round(ahead_ms)}ms ago; STT may already have this phrase)"
            )
          else:
            print("--- speech end ---")

      segment = stt.feed(chunk.samples, chunk.sample_rate)
      if segment and segment.text:
        parts.append(segment.text)
        latency.mark_stt_partial()
        since_vad_end = (
          latency.mark_first_stt_partial() if segment.text.strip() else None
        )
        suffix = (
          f" ({round(since_vad_end)}ms since speech end)"
          if since_vad_end is not None
          else ""
        )
        print(f"partial: {segment.text!r}{suffix}")
      if segment and segment.end_of_utterance:
        print("<EOU>")
  except KeyboardInterrupt:
    print()
  finally:
    final = stt.finalize()
    if final and final.text:
      parts.append(final.text)
    capture.close()
    stt.close()

  transcript = "".join(parts).strip()
  print(f"Transcript: {transcript!r}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
