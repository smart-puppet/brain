from unittest.mock import patch

from puppet.core.audio.respeaker import (
  device_name_matches_respeaker,
  maybe_reset_respeaker_on_start,
)


def test_device_name_matches_respeaker() -> None:
  assert device_name_matches_respeaker("reSpeaker XVF3800 4-Mic Array: USB Audio")
  assert device_name_matches_respeaker("Seeed Studio device")
  assert not device_name_matches_respeaker("USB PnP Sound Device")


def test_maybe_reset_skips_when_never() -> None:
  assert not maybe_reset_respeaker_on_start({"audio": {"respeaker": {"usb_reset_on_start": "never"}}})


def test_maybe_reset_auto_requires_match() -> None:
  config = {
    "audio": {
      "respeaker": {"usb_reset_on_start": "auto"},
      "input_device": None,
    }
  }
  with patch(
    "puppet.core.audio.respeaker._resolve_input_device_name",
    return_value="Built-in Audio",
  ):
    with patch("puppet.core.audio.respeaker.find_respeaker_usb_devices", return_value=[]):
      assert not maybe_reset_respeaker_on_start(config)


def test_maybe_reset_auto_by_device_name() -> None:
  config = {"audio": {"respeaker": {"usb_reset_on_start": "auto", "settle_ms": 0}}}
  with patch(
    "puppet.core.audio.respeaker._resolve_input_device_name",
    return_value="reSpeaker XVF3800",
  ):
    with patch("puppet.core.audio.respeaker.find_respeaker_usb_devices") as find:
      with patch("puppet.core.audio.respeaker.reset_respeaker_usb", return_value=True) as reset:
        from puppet.core.audio.respeaker import RespeakerUsbDevice

        find.return_value = [RespeakerUsbDevice(0x2886, 0x001A, 1, 5)]
        assert maybe_reset_respeaker_on_start(config)
        reset.assert_called_once()


def test_doa_reading_compass() -> None:
  from puppet.core.audio.respeaker import DoaReading

  assert DoaReading(0, True).compass == "N"
  assert DoaReading(90, True).compass == "E"
  assert DoaReading(180, False).compass == "S"


def test_doa_monitor_logs_when_speaking() -> None:
  from puppet.core.audio.respeaker import DoaReading, RespeakerDoaMonitor

  monitor = RespeakerDoaMonitor(
    {"audio": {"respeaker": {"doa_debug": True, "doa_poll_ms": 0, "settle_ms": 0}}}
  )
  reading = DoaReading(45, True)
  with patch.object(monitor, "_ensure_usb", return_value=True):
    with patch("puppet.core.audio.respeaker.read_doa", return_value=reading):
      monitor.maybe_log(speech_active=True)
  assert monitor._last_log == (45, True)
  monitor.close()


def test_doa_monitor_silent_when_not_speaking() -> None:
  from puppet.core.audio.respeaker import RespeakerDoaMonitor

  monitor = RespeakerDoaMonitor({"audio": {"respeaker": {"doa_debug": True}}})
  with patch("puppet.core.audio.respeaker.read_doa") as read_doa:
    monitor.maybe_log(speech_active=False)
    read_doa.assert_not_called()
  monitor.close()
