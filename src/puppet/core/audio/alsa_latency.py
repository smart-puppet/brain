from __future__ import annotations

import logging
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_SAMPLE_SPEC_RE = re.compile(
  r"(?P<fmt>s16le|s16be|s32le|float32le)\s+(?P<ch>\d+)ch\s+(?P<hz>\d+)Hz",
  re.IGNORECASE,
)
_BYTES_PER_SAMPLE = {
  "s16le": 2,
  "s16be": 2,
  "s32le": 4,
  "float32le": 4,
}


@dataclass(frozen=True)
class AlsaHwParams:
  rate: int
  channels: int
  period_size: int
  buffer_size: int

  @property
  def period_ms(self) -> float:
    if self.rate <= 0:
      return 0.0
    return self.period_size * 1000.0 / self.rate

  @property
  def buffer_ms(self) -> float:
    if self.rate <= 0:
      return 0.0
    return self.buffer_size * 1000.0 / self.rate


@dataclass(frozen=True)
class PulseAlsaSink:
  name: str
  owner_module: int
  card: int
  device: int
  sample_format: str
  channels: int
  rate: int


def parse_hw_params(text: str) -> AlsaHwParams | None:
  if "closed" in text.splitlines()[:1]:
    return None
  fields: dict[str, str] = {}
  for line in text.splitlines():
    if ":" not in line:
      continue
    key, value = line.split(":", 1)
    fields[key.strip()] = value.strip()
  try:
    rate_token = fields["rate"].split()[0]
    return AlsaHwParams(
      rate=int(rate_token),
      channels=int(fields["channels"]),
      period_size=int(fields["period_size"]),
      buffer_size=int(fields["buffer_size"]),
    )
  except (KeyError, ValueError, IndexError):
    return None


def read_playback_hw_params(card: int, device: int = 0) -> AlsaHwParams | None:
  path = Path(f"/proc/asound/card{card}/pcm{device}p/sub0/hw_params")
  try:
    return parse_hw_params(path.read_text())
  except OSError:
    return None


def parse_sample_spec(text: str) -> tuple[str, int, int] | None:
  match = _SAMPLE_SPEC_RE.search(text)
  if match is None:
    return None
  return match.group("fmt").lower(), int(match.group("ch")), int(match.group("hz"))


_RESPEAKER_MARKERS = ("respeaker", "xvf3800", "seeed")


def pulse_name_is_respeaker(name: str) -> bool:
  lowered = (name or "").lower()
  if ".monitor" in lowered:
    return False
  return any(marker in lowered for marker in _RESPEAKER_MARKERS)


def pick_respeaker_pulse_name(short_list: str) -> str | None:
  """Prefer analog ReSpeaker from `pactl list {sinks,sources} short`."""
  analog = None
  other = None
  for line in short_list.splitlines():
    parts = line.split("\t")
    if len(parts) < 2:
      continue
    name = parts[1].strip()
    if not pulse_name_is_respeaker(name):
      continue
    if "analog" in name.lower():
      analog = name
    else:
      other = other or name
  return analog or other


def pin_respeaker_pulse_defaults() -> str | None:
  """Force Pulse default sink/source onto the XVF3800 so hardware AEC sees TTS."""
  try:
    sinks = _pactl("list", "sinks", "short")
    sources = _pactl("list", "sources", "short")
    default_sink = _pactl("get-default-sink").strip()
    default_source = _pactl("get-default-source").strip()
  except (FileNotFoundError, RuntimeError, subprocess.TimeoutExpired) as exc:
    logger.info("Pulse ReSpeaker pin skipped (%s)", exc)
    return None
  sink = pick_respeaker_pulse_name(sinks)
  source = pick_respeaker_pulse_name(sources)
  if sink and sink != default_sink:
    try:
      _pactl("set-default-sink", sink)
      logger.info("Pulse default sink → %s (XVF3800 AEC far-end)", sink)
      default_sink = sink
    except (RuntimeError, subprocess.TimeoutExpired) as exc:
      logger.warning("Could not set Pulse default sink to ReSpeaker: %s", exc)
  if source and source != default_source:
    try:
      _pactl("set-default-source", source)
      logger.info("Pulse default source → %s", source)
    except (RuntimeError, subprocess.TimeoutExpired) as exc:
      logger.warning("Could not set Pulse default source to ReSpeaker: %s", exc)
  return sink or default_sink


def fragment_bytes(*, rate: int, channels: int, sample_format: str, period_ms: int) -> int:
  width = _BYTES_PER_SAMPLE.get(sample_format.lower(), 2)
  frames = max(1, int(rate * period_ms / 1000))
  return frames * max(1, channels) * width


def parse_pulse_sinks(text: str) -> list[PulseAlsaSink]:
  sinks: list[PulseAlsaSink] = []
  current: dict[str, str] = {}

  def _flush() -> None:
    name = current.get("name")
    spec = parse_sample_spec(current.get("spec", ""))
    if not name or spec is None or "card" not in current or "owner" not in current:
      return
    fmt, channels, rate = spec
    sinks.append(
      PulseAlsaSink(
        name=name,
        owner_module=int(current["owner"]),
        card=int(current["card"]),
        device=int(current.get("device", "0")),
        sample_format=fmt,
        channels=channels,
        rate=rate,
      )
    )

  for raw in text.splitlines():
    line = raw.strip()
    if line.startswith("Sink #"):
      _flush()
      current = {}
      continue
    if line.startswith("Name:"):
      current["name"] = line.split(":", 1)[1].strip()
    elif line.startswith("Sample Specification:"):
      current["spec"] = line.split(":", 1)[1].strip()
    elif line.startswith("Owner Module:"):
      current["owner"] = line.split(":", 1)[1].strip()
    elif line.startswith("alsa.card ="):
      current["card"] = line.split("=", 1)[1].strip().strip('"')
    elif line.startswith("alsa.device ="):
      current["device"] = line.split("=", 1)[1].strip().strip('"')
  _flush()
  return sinks


def parse_alsa_card_modules(text: str) -> dict[int, str]:
  modules: dict[int, str] = {}
  for line in text.splitlines():
    parts = line.split("\t")
    if len(parts) < 3 or parts[1] != "module-alsa-card":
      continue
    try:
      modules[int(parts[0])] = parts[2].strip()
    except ValueError:
      continue
  return modules


def alsa_card_args_need_reload(args: str, *, fragment_size: int, fragments: int) -> bool:
  lowered = f" {args} "
  if " tsched=no " not in lowered and " tsched=false " not in lowered:
    return True
  match = re.search(r"fragment_size=(\d+)", args)
  if match is None or int(match.group(1)) > fragment_size * 2:
    return True
  match = re.search(r"fragments=(\d+)", args)
  if match is None:
    return True
  return int(match.group(1)) > fragments * 2


def rewrite_alsa_card_args(args: str, *, fragment_size: int, fragments: int) -> str:
  tokens = [tok for tok in args.split() if tok]
  updated: list[str] = []
  seen = {"tsched": False, "fragment_size": False, "fragments": False}
  for tok in tokens:
    key = tok.split("=", 1)[0]
    if key == "tsched":
      updated.append("tsched=no")
      seen["tsched"] = True
    elif key == "fragment_size":
      updated.append(f"fragment_size={fragment_size}")
      seen["fragment_size"] = True
    elif key == "fragments":
      updated.append(f"fragments={fragments}")
      seen["fragments"] = True
    else:
      updated.append(tok)
  if not seen["tsched"]:
    updated.append("tsched=no")
  if not seen["fragment_size"]:
    updated.append(f"fragment_size={fragment_size}")
  if not seen["fragments"]:
    updated.append(f"fragments={fragments}")
  return " ".join(updated)


def _load_module_tokens(args: str) -> list[str]:
  """Keep quoted values that contain '=' so Pulse modargs stay intact."""
  tokens: list[str] = []
  for tok in args.split():
    if "=" not in tok:
      tokens.append(tok)
      continue
    key, val = tok.split("=", 1)
    if val.startswith('"') and val.endswith('"') and "=" not in val[1:-1]:
      tokens.append(f"{key}={val[1:-1]}")
    else:
      tokens.append(tok)
  return tokens


def _pactl(*args: str) -> str:
  result = subprocess.run(
    ["pactl", *args],
    check=False,
    capture_output=True,
    text=True,
    timeout=8,
  )
  if result.returncode != 0:
    err = (result.stderr or result.stdout or "").strip()
    raise RuntimeError(err or f"pactl {' '.join(args)} failed")
  return result.stdout


def configure_low_latency_alsa_output(audio_cfg: dict[str, Any] | None = None) -> bool:
  """Reload Pulse's ALSA card so the speaker uses a short period/buffer.

  Pulse ``tsched=yes`` currently opens the ReSpeaker playback PCM with a 1 s
  period / 2 s buffer. PortAudio ``frames_per_buffer`` never changes that.
  """
  cfg = audio_cfg or {}
  period_ms = max(5, int(cfg.get("output_alsa_period_ms", 20)))
  periods = max(2, int(cfg.get("output_alsa_periods", 3)))
  pin_respeaker_pulse_defaults()
  try:
    default_sink = _pactl("get-default-sink").strip()
    sinks = parse_pulse_sinks(_pactl("list", "sinks"))
    modules = parse_alsa_card_modules(_pactl("list", "modules", "short"))
  except (FileNotFoundError, RuntimeError, subprocess.TimeoutExpired) as exc:
    logger.info("ALSA output buffer unchanged (Pulse unavailable: %s)", exc)
    return False

  sink = next((item for item in sinks if item.name == default_sink), None)
  if sink is None:
    logger.info("ALSA output buffer unchanged (no Pulse sink %s)", default_sink)
    return False
  args = modules.get(sink.owner_module)
  if not args:
    logger.info("ALSA output buffer unchanged (sink %s is not module-alsa-card)", sink.name)
    return False

  frag = fragment_bytes(
    rate=sink.rate,
    channels=sink.channels,
    sample_format=sink.sample_format,
    period_ms=period_ms,
  )
  before = read_playback_hw_params(sink.card, sink.device)
  if not alsa_card_args_need_reload(args, fragment_size=frag, fragments=periods):
    if before is not None:
      logger.info(
        "ALSA playback card%s already short: period=%d (%.0fms) buffer=%d (%.0fms)",
        sink.card,
        before.period_size,
        before.period_ms,
        before.buffer_size,
        before.buffer_ms,
      )
    return False

  new_args = rewrite_alsa_card_args(args, fragment_size=frag, fragments=periods)
  before_ms = before.buffer_ms if before is not None else None
  load_tokens = _load_module_tokens(new_args)
  try:
    _pactl("unload-module", str(sink.owner_module))
    try:
      _pactl("load-module", "module-alsa-card", *load_tokens)
      _pactl("set-default-sink", sink.name)
    except (RuntimeError, subprocess.TimeoutExpired):
      _pactl("load-module", "module-alsa-card", *_load_module_tokens(args))
      _pactl("set-default-sink", sink.name)
      raise
  except (RuntimeError, subprocess.TimeoutExpired) as exc:
    logger.warning("Could not shrink ALSA output buffer: %s", exc)
    return False

  after = read_playback_hw_params(sink.card, sink.device)
  if after is not None:
    logger.info(
      "ALSA playback card%s period=%d (%.0fms) buffer=%d (%.0fms) tsched=off",
      sink.card,
      after.period_size,
      after.period_ms,
      after.buffer_size,
      after.buffer_ms,
    )
  elif before_ms is not None:
    logger.info(
      "ALSA playback card%s reloaded tsched=off target %dms x %d (was buffer %.0fms)",
      sink.card,
      period_ms,
      periods,
      before_ms,
    )
  else:
    logger.info(
      "ALSA playback card%s reloaded tsched=off target %dms x %d",
      sink.card,
      period_ms,
      periods,
    )
  return True
