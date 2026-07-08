#!/usr/bin/env python3
"""Record from the mic, print levels, save a WAV, and optionally run parakeet STT."""

from __future__ import annotations

import argparse
import sys
import time
import wave
from pathlib import Path

import numpy as np

from puppet.core.audio.capture import (
  AudioCapture,
  AudioPlayback,
  STT_SAMPLE_RATE,
  list_input_devices,
)
from puppet.core.audio.respeaker import maybe_reset_respeaker_on_start


def _float_to_int16(samples: np.ndarray) -> np.ndarray:
  return np.clip(samples * 32768.0, -32768, 32767).astype(np.int16)


def _save_wav(path: Path, samples: np.ndarray, sample_rate: int) -> None:
  pcm = _float_to_int16(samples)
  with wave.open(str(path), "wb") as wf:
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(sample_rate)
    wf.writeframes(pcm.tobytes())


def main(argv: list[str] | None = None) -> int:
  parser = argparse.ArgumentParser(description="Check microphone capture for STT")
  parser.add_argument("--seconds", type=float, default=5.0, help="Record duration")
  parser.add_argument("--device", type=int, default=None, help="Input device index")
  parser.add_argument("--list-devices", action="store_true", help="List input devices and exit")
  parser.add_argument(
    "--wav",
    type=Path,
    default=Path("mic_check.wav"),
    help="Output WAV path (16 kHz mono int16)",
  )
  parser.add_argument(
    "--model",
    type=Path,
    default=Path("models/stt/nemotron-3.5-asr-streaming-0.6b-q8_0.gguf"),
    help="Parakeet GGUF for a live STT smoke test",
  )
  parser.add_argument("--language", default="en-US", help="Parakeet language prompt")
  args = parser.parse_args(argv)

  if args.list_devices:
    for dev in list_input_devices():
      flag = "yes" if dev.supports_16k_int16 else "NO"
      print(
        f"[{dev.index}] {dev.name} "
        f"(default_rate={int(dev.default_sample_rate)} Hz, "
        f"16k_int16={flag}, max_in_ch={dev.max_input_channels})"
      )
    return 0

  capture = AudioCapture(device_index=args.device)
  print(
    f"Using device [{capture.device_index}] {capture.device_name!r} "
    f"({capture.sample_rate} Hz int16 mono)"
  )
  print(f"Recording {args.seconds:.1f}s — speak now...")

  chunks: list[np.ndarray] = []
  deadline = time.monotonic() + args.seconds
  peaks: list[float] = []
  while time.monotonic() < deadline:
    chunk = capture.read()
    chunks.append(chunk.samples)
    peaks.append(float(np.max(np.abs(chunk.samples))) if chunk.samples.size else 0.0)

  capture.close()
  audio = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
  peak = float(np.max(peaks)) if peaks else 0.0
  rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2))) if audio.size else 0.0

  _save_wav(args.wav, audio, STT_SAMPLE_RATE)
  print(f"Saved {args.wav} ({len(audio) / STT_SAMPLE_RATE:.2f}s @ {STT_SAMPLE_RATE} Hz)")
  print(f"Levels: peak={peak:.4f} rms={rms:.4f} (expect peak > 0.05 when speaking)")

  if peak < 0.01:
    print(
      "WARNING: very low level — wrong input device or muted mic. "
      "Run with --list-devices and set audio.input_device in config/default.yaml",
      file=sys.stderr,
    )

  if not args.model.is_file():
    print(f"Skipping STT (model not found: {args.model})")
    return 0

  try:
    import puppet_parakeet as pk
  except ImportError:
    print("Skipping STT (puppet_parakeet not built — run ./scripts/build_parakeet.sh)")
    return 0

  ctx = pk.load(str(args.model))
  stream = ctx.stream_begin_lang(args.language)
  text_parts: list[str] = []
  block = max(STT_SAMPLE_RATE // 10, 1)
  for offset in range(0, len(audio), block):
    piece = audio[offset : offset + block]
    text, eou = stream.feed(piece, STT_SAMPLE_RATE)
    if text:
      text_parts.append(text)
      print(f"  partial: {text!r}")
    if eou:
      print("  <EOU>")
  tail = stream.finalize()
  if tail:
    text_parts.append(tail)
  transcript = "".join(text_parts).strip()
  print(f"Transcript: {transcript!r}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
