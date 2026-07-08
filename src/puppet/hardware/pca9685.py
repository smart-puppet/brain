from __future__ import annotations

import time

# PCA9685 registers
_MODE1 = 0x00
_PRESCALE = 0xFE
_LED0_ON_L = 0x06

_MODE1_SLEEP = 0x10
_MODE1_AI = 0x08
_MODE1_RESTART = 0x80

_SERVO_MIN_US = 500
_SERVO_MAX_US = 2500


class PCA9685:
  """Minimal PCA9685 driver over I2C (smbus2 SMBus instance)."""

  def __init__(self, bus: object, address: int = 0x40) -> None:
    self._bus = bus
    self._address = address

  def read_byte(self, register: int) -> int:
    return self._bus.read_byte_data(self._address, register)

  def set_pwm_frequency(self, hz: float) -> None:
    if not 24 <= hz <= 1526:
      raise ValueError(f"PWM frequency out of range: {hz} Hz")
    prescale = int(round(25_000_000 / (4096 * hz))) - 1
    prescale = max(3, min(prescale, 255))

    old_mode = self.read_byte(_MODE1)
    self._bus.write_byte_data(self._address, _MODE1, (old_mode & 0x7F) | _MODE1_SLEEP)
    self._bus.write_byte_data(self._address, _PRESCALE, prescale)
    awake = old_mode & ~_MODE1_SLEEP
    self._bus.write_byte_data(self._address, _MODE1, awake)
    time.sleep(0.005)
    self._bus.write_byte_data(
      self._address,
      _MODE1,
      awake | _MODE1_RESTART | _MODE1_AI,
    )
    time.sleep(0.005)

  def read_pwm_off(self, channel: int) -> int:
    base = _LED0_ON_L + 4 * channel
    lo = self._bus.read_byte_data(self._address, base + 2)
    hi = self._bus.read_byte_data(self._address, base + 3)
    return ((hi & 0x0F) << 8) | lo

  def set_pwm(self, channel: int, on: int, off: int) -> None:
    if not 0 <= channel <= 15:
      raise ValueError(f"channel must be 0..15, got {channel}")
    on = max(0, min(on, 4095))
    off = max(0, min(off, 4095))
    base = _LED0_ON_L + 4 * channel
    self._bus.write_byte_data(self._address, base, on & 0xFF)
    self._bus.write_byte_data(self._address, base + 1, (on >> 8) & 0x0F)
    self._bus.write_byte_data(self._address, base + 2, off & 0xFF)
    self._bus.write_byte_data(self._address, base + 3, (off >> 8) & 0x0F)

  def set_pulse_us(self, channel: int, pulse_us: int, *, frequency_hz: float = 50.0) -> None:
    period_us = 1_000_000 / frequency_hz
    pulse_us = max(0, min(int(pulse_us), int(period_us)))
    ticks = int(round(pulse_us * 4096 / period_us))
    self.set_pwm(channel, 0, ticks)

  def set_servo_angle(self, channel: int, angle_deg: float) -> None:
    angle_deg = max(0.0, min(float(angle_deg), 180.0))
    span = _SERVO_MAX_US - _SERVO_MIN_US
    pulse_us = _SERVO_MIN_US + (angle_deg / 180.0) * span
    self.set_pulse_us(channel, pulse_us)
