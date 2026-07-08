#!/usr/bin/env python3
"""Test TTS only: type text, play audio through the speaker."""

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
from puppet.core.audio import AudioPlayback, prepend_lead_in_silence
from puppet.tts import create_tts


def main(argv: list[str] | None = None) -> int:
  parser = argparse.ArgumentParser(description="Test TTS only (text → Piper → speaker)")
  parser.add_argument("--config", default="config", help="Config directory")
  parser.add_argument("--language", "-l", choices=["en", "fr", "de"], default=None)
  parser.add_argument("text", nargs="*", help="Text to speak (omit for interactive mode)")
  args = parser.parse_args(argv)

  config = load_puppet_config(args.config, language=args.language)
  configure_logging(config)

  tts = create_tts(config)
  audio_cfg = config.get("audio", {})
  playback = AudioPlayback(
    sample_rate=tts.sample_rate(),
    channels=int(audio_cfg.get("channels", 1)),
    device_index=audio_cfg.get("output_device"),
    frames_per_buffer=int(audio_cfg.get("output_frames_per_buffer", 2048)),
  )

  one_shot = bool(args.text)
  text = " ".join(args.text).strip()

  playback.warmup(silence_ms=int(audio_cfg.get("playback_warmup_ms", 150)))

  tts_cfg = config.get("tts", {})

  def _lead_in_ms(text: str) -> int:
    lead_in_ms = int(tts_cfg.get("lead_in_ms", 80))
    max_words = int(tts_cfg.get("short_phrase_max_words", 2))
    short_ms = int(tts_cfg.get("short_phrase_lead_in_ms", 160))
    if len(text.split()) <= max_words:
      lead_in_ms = max(lead_in_ms, short_ms)
    return lead_in_ms

  def speak_line(line: str) -> None:
    print(f"Playing ({tts.sample_rate()} Hz): {line!r}")
    started = time.monotonic()
    first_chunk_at: float | None = None
    synth_done_at: float | None = None
    samples = 0
    lead_in_applied = False

    for chunk in tts.synthesize_stream(line):
      if first_chunk_at is None:
        first_chunk_at = time.monotonic()
      samples_arr = chunk.samples
      if not lead_in_applied:
        samples_arr = prepend_lead_in_silence(samples_arr, tts.sample_rate(), _lead_in_ms(line))
        lead_in_applied = True
      samples += int(samples_arr.size)
      pcm = np.clip(samples_arr * 32767.0, -32768, 32767).astype(np.int16).tobytes()
      playback.play_int16(pcm)

    synth_done_at = time.monotonic()
    playback.flush()
    finished = time.monotonic()

    sample_rate = tts.sample_rate()
    audio_ms = (samples / sample_rate) * 1000.0 if sample_rate else 0.0
    first_chunk_ms = (
      (first_chunk_at - started) * 1000.0 if first_chunk_at is not None else None
    )
    synth_ms = (synth_done_at - started) * 1000.0
    total_ms = (finished - started) * 1000.0
    rtf = audio_ms / synth_ms if synth_ms > 0 else 0.0

    parts = [f"synth={round(synth_ms)}ms", f"audio={round(audio_ms)}ms", f"rtf={rtf:.2f}x"]
    if first_chunk_ms is not None:
      parts.insert(0, f"first_chunk={round(first_chunk_ms)}ms")
    parts.append(f"total={round(total_ms)}ms")
    print(f"timing: {', '.join(parts)}")

  try:
    if one_shot:
      if not text:
        print("No input.", file=sys.stderr)
        return 1
      speak_line(text)
      return 0

    print("Multi-turn TTS test. Empty line or 'quit' to exit.")
    print(
      "Timing: first_chunk=synthesis latency, synth=until last chunk, "
      "audio=duration, rtf=audio/synth, total=includes playback flush"
    )
    while True:
      try:
        text = input("Say: ").strip()
      except EOFError:
        print()
        break
      if not text or text.lower() in {"quit", "exit", "q"}:
        break
      speak_line(text)
  finally:
    playback.close()

  print("Done.")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
