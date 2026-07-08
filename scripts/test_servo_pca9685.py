#!/usr/bin/env python3
"""Smoke-test a PCA9685 servo driver on Jetson I2C (MF90 on channel 15)."""

from __future__ import annotations

import argparse
import sys
import time

try:
  from smbus2 import SMBus
except ImportError as exc:
  raise SystemExit(
    "smbus2 is required. Install with: pip install smbus2\n"
    "Or: sudo apt install python3-smbus  (then use smbus instead)"
  ) from exc

from puppet.hardware.pca9685 import PCA9685

# PCA9685 registers (for probe output)
_MODE1 = 0x00
_MODE2 = 0x01
_MODE1_SLEEP = 0x10


def scan_bus(bus: SMBus) -> list[int]:
  found: list[int] = []
  for address in range(0x03, 0x78):
    try:
      bus.read_byte(address)
      found.append(address)
    except OSError:
      pass
  return found


def probe_pca9685(pca: PCA9685) -> None:
  mode1 = pca.read_byte(_MODE1)
  mode2 = pca.read_byte(_MODE2)
  print(f"PCA9685 MODE1=0x{mode1:02x}  MODE2=0x{mode2:02x}")


def _sweep_angles(min_deg: float, max_deg: float, step_deg: float) -> list[float]:
  if min_deg == max_deg:
    return [min_deg]
  if step_deg <= 0:
    return [min_deg, max_deg, min_deg]
  up: list[float] = []
  angle = min_deg
  while angle <= max_deg + 1e-6:
    up.append(angle)
    angle += step_deg
  if len(up) <= 1:
    return up
  down: list[float] = []
  angle = up[-1] - step_deg
  while angle >= min_deg - 1e-6:
    down.append(angle)
    angle -= step_deg
  return up + down


def run_sweep(
  pca: PCA9685,
  channel: int,
  *,
  min_deg: float,
  max_deg: float,
  step_deg: float,
  pause_s: float,
  cycles: int = 1,
) -> None:
  path = _sweep_angles(min_deg, max_deg, step_deg)
  for cycle in range(1, cycles + 1):
    print(f"Cycle {cycle}/{cycles}")
    for angle in path:
      print(f"  channel {channel} → {angle:.0f}°")
      pca.set_servo_angle(channel, angle)
      time.sleep(pause_s)


def main(argv: list[str] | None = None) -> int:
  parser = argparse.ArgumentParser(
    description="Test PCA9685 + MF90 servo on Jetson I2C (default: bus 1, channel 15)",
  )
  parser.add_argument(
    "--bus",
    type=int,
    default=7,
    help="I2C bus number (Jetson Orin Nano pins 3/5 → usually bus 7)",
  )
  parser.add_argument(
    "--address",
    type=lambda s: int(s, 0),
    default=0x40,
    help="PCA9685 I2C address (default 0x40)",
  )
  parser.add_argument(
    "--channel",
    type=int,
    default=15,
    help="PWM channel 0..15 (your MF90 is on 15)",
  )
  parser.add_argument(
    "--frequency",
    type=float,
    default=50.0,
    help="Servo PWM frequency in Hz (standard hobby servos: 50)",
  )
  parser.add_argument("--scan", action="store_true", help="Scan I2C bus and exit")
  parser.add_argument("--probe", action="store_true", help="Read PCA9685 registers and exit")
  parser.add_argument(
    "--angle",
    type=float,
    default=None,
    help="Move servo to angle 0..180 and hold",
  )
  parser.add_argument(
    "--sweep",
    action="store_true",
    help="Sweep servo from --sweep-min to --sweep-max by --sweep-step",
  )
  parser.add_argument(
    "--sweep-min",
    type=float,
    default=0.0,
    help="Sweep start angle in degrees (default 0)",
  )
  parser.add_argument(
    "--sweep-max",
    type=float,
    default=25.0,
    help="Sweep end angle in degrees (default 25)",
  )
  parser.add_argument(
    "--sweep-step",
    type=float,
    default=0.0,
    help="Sweep step in degrees; 0 = jump straight min↔max (default 0)",
  )
  parser.add_argument(
    "--sweep-cycles",
    type=int,
    default=3,
    help="Round trips 0→max→0 to repeat (default 3)",
  )
  parser.add_argument(
    "--hold",
    type=float,
    default=2.0,
    help="Seconds to hold after --angle (default 2)",
  )
  parser.add_argument(
    "--pulse-us",
    type=int,
    default=None,
    help="Set raw pulse width in microseconds (bypasses angle mapping)",
  )
  parser.add_argument(
    "--pause",
    type=float,
    default=1.0,
    help="Pause between positions during --sweep (default 1)",
  )
  args = parser.parse_args(argv)

  if not 0 <= args.channel <= 15:
    print("channel must be 0..15", file=sys.stderr)
    return 1

  dev_path = f"/dev/i2c-{args.bus}"
  print(f"Opening {dev_path} (PCA9685 @ {args.address:#04x})")

  try:
    with SMBus(args.bus) as bus:
      if args.scan:
        found = scan_bus(bus)
        print(f"I2C scan on bus {args.bus}: {[f'{a:#04x}' for a in found] or 'no devices'}")
        if args.address not in found:
          print(
            f"Warning: {args.address:#04x} not seen. Check wiring, power, and "
            f"sudo i2cdetect -y {args.bus}",
            file=sys.stderr,
          )
          return 1
        return 0

      pca = PCA9685(bus, address=args.address)

      if args.probe:
        probe_pca9685(pca)
        return 0

      pca.set_pwm_frequency(args.frequency)
      probe_pca9685(pca)
      mode1 = pca.read_byte(_MODE1)
      if mode1 & _MODE1_SLEEP:
        print(
          f"Warning: PCA9685 still in SLEEP (MODE1=0x{mode1:02x}) — PWM outputs disabled.",
          file=sys.stderr,
        )
      print(f"PWM frequency set to {args.frequency:.0f} Hz")

      if args.sweep:
        if args.sweep_cycles < 1:
          print("--sweep-cycles must be >= 1", file=sys.stderr)
          return 1
        print(
          f"Sweeping servo {args.sweep_min:.0f}°↔{args.sweep_max:.0f}° "
          f"({args.sweep_cycles} cycle(s))..."
        )
        run_sweep(
          pca,
          args.channel,
          min_deg=args.sweep_min,
          max_deg=args.sweep_max,
          step_deg=args.sweep_step,
          pause_s=args.pause,
          cycles=args.sweep_cycles,
        )
        print("Sweep done.")
        return 0

      if args.pulse_us is not None:
        print(f"Setting channel {args.channel} to {args.pulse_us} µs")
        pca.set_pulse_us(args.channel, args.pulse_us, frequency_hz=args.frequency)
      else:
        angle = 90.0 if args.angle is None else args.angle
        print(f"Setting channel {args.channel} to {angle:.1f}°")
        pca.set_servo_angle(args.channel, angle)

      off_ticks = pca.read_pwm_off(args.channel)
      print(f"PWM register: OFF ticks = {off_ticks}")
      if off_ticks == 0:
        print(
          "Warning: PWM register still 0 — check OE→GND, V+ power, and channel wiring.",
          file=sys.stderr,
        )
      if args.pulse_us is not None or args.angle is not None or args.hold > 0:
        print(f"Holding for {args.hold:.1f}s (Ctrl+C to stop early)")
        time.sleep(args.hold)
      return 0

  except FileNotFoundError:
    print(
      f"{dev_path} not found. Enable I2C and check bus number.\n"
      f"  sudo /opt/nvidia/jetson-io/jetson-io.py   # or jetson-config\n"
      f"  ls /dev/i2c-*",
      file=sys.stderr,
    )
    return 1
  except PermissionError:
    print(
      f"No permission for {dev_path}. Try:\n"
      f"  sudo usermod -aG i2c $USER   # then log out/in\n"
      f"  or run with sudo",
      file=sys.stderr,
    )
    return 1
  except OSError as exc:
    print(
      f"I2C error: {exc}\n"
      f"Check power (V+ for servo, VCC for logic), GND common, SDA/SCL on bus {args.bus}.",
      file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
  raise SystemExit(main())
