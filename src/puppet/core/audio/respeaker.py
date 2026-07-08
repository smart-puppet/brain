from __future__ import annotations

import glob
import logging
import os
import struct
import time
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

# Seeed reSpeaker XVF3800 USB IDs (DFU + runtime).
DEFAULT_RESPEAKER_VENDOR_ID = 0x2886
DEFAULT_RESPEAKER_PRODUCT_IDS = (0x001A, 0x001a)

DEFAULT_NAME_PATTERNS = (
  "respeaker",
  "xvf3800",
  "xvf38",
  "see studio",
  "seeed",
)

USBDEVFS_RESET = 21780  # _IO('U', 20) on Linux
CONTROL_SUCCESS = 0
SERVICER_COMMAND_RETRY = 64

# GPO_SERVICER_RESID DOA_VALUE: azimuth 0–359°, speech flag.
_DOA_RESID = 20
_DOA_CMDID = 18
_DOA_VALUE_COUNT = 2


@dataclass(frozen=True)
class DoaReading:
  azimuth_deg: int
  speech_detected: bool

  @property
  def compass(self) -> str:
    dirs = ("N", "NE", "E", "SE", "S", "SW", "W", "NW")
    return dirs[int((self.azimuth_deg + 22.5) // 45) % 8]


@dataclass(frozen=True)
class RespeakerUsbDevice:
  vendor_id: int
  product_id: int
  bus: int
  device: int
  sysfs_path: str | None = None
  sysfs_name: str | None = None

  @property
  def dev_path(self) -> str:
    return f"/dev/bus/usb/{self.bus:03d}/{self.device:03d}"


def _normalize_patterns(patterns: Any) -> tuple[str, ...]:
  if patterns is None:
    return DEFAULT_NAME_PATTERNS
  if isinstance(patterns, str):
    return (patterns.lower(),)
  return tuple(str(p).lower() for p in patterns if str(p).strip())


def device_name_matches_respeaker(
  name: str,
  patterns: tuple[str, ...] | None = None,
) -> bool:
  lowered = name.lower()
  for pattern in patterns or DEFAULT_NAME_PATTERNS:
    if pattern in lowered:
      return True
  return False


def find_respeaker_usb_devices(
  *,
  vendor_id: int = DEFAULT_RESPEAKER_VENDOR_ID,
  product_ids: tuple[int, ...] | list[int] | None = None,
) -> list[RespeakerUsbDevice]:
  """Return attached ReSpeaker USB devices (Linux sysfs)."""
  if os.name != "posix":
    return []

  wanted = {int(pid) for pid in (product_ids or DEFAULT_RESPEAKER_PRODUCT_IDS)}
  vendor_hex = f"{int(vendor_id):04x}"
  found: list[RespeakerUsbDevice] = []

  for vendor_path in glob.glob("/sys/bus/usb/devices/*/idVendor"):
    try:
      with open(vendor_path, encoding="utf-8") as handle:
        if handle.read().strip().lower() != vendor_hex:
          continue
      base = os.path.dirname(vendor_path)
      with open(os.path.join(base, "idProduct"), encoding="utf-8") as handle:
        product_id = int(handle.read().strip(), 16)
      if product_id not in wanted:
        continue
      bus = int(open(os.path.join(base, "busnum"), encoding="utf-8").read().strip())
      dev = int(open(os.path.join(base, "devnum"), encoding="utf-8").read().strip())
      product_name = None
      product_path = os.path.join(base, "product")
      if os.path.isfile(product_path):
        with open(product_path, encoding="utf-8") as handle:
          product_name = handle.read().strip() or None
      found.append(
        RespeakerUsbDevice(
          vendor_id=int(vendor_id),
          product_id=product_id,
          bus=bus,
          device=dev,
          sysfs_path=base,
          sysfs_name=product_name,
        )
      )
    except (OSError, ValueError):
      continue
  return found


def _open_xvf_usb(vendor_id: int, product_id: int) -> object | None:
  try:
    import usb.core
  except ImportError:
    return None
  dev = usb.core.find(idVendor=vendor_id, idProduct=product_id)
  if dev is None:
    return None
  return dev


def _wait_for_usb_state(
  *,
  vendor_id: int,
  product_ids: tuple[int, ...] | list[int],
  present: bool,
  timeout_s: float,
) -> bool:
  deadline = time.monotonic() + max(0.05, timeout_s)
  while time.monotonic() < deadline:
    seen = bool(find_respeaker_usb_devices(vendor_id=vendor_id, product_ids=product_ids))
    if seen == present:
      return True
    time.sleep(0.05)
  return bool(find_respeaker_usb_devices(vendor_id=vendor_id, product_ids=product_ids)) == present


def _close_xvf_usb(dev: object | None) -> None:
  if dev is None:
    return
  try:
    import usb.util

    usb.util.dispose_resources(dev)
  except Exception:
    pass


def read_doa(
  *,
  vendor_id: int = DEFAULT_RESPEAKER_VENDOR_ID,
  product_ids: tuple[int, ...] | list[int] | None = None,
  dev: object | None = None,
) -> DoaReading | None:
  """Read direction-of-arrival from XVF3800 (0–359°). Requires pyusb."""
  try:
    import usb.core
    import usb.util
  except ImportError:
    return None

  own_dev = dev is None
  if own_dev:
    for product_id in product_ids or DEFAULT_RESPEAKER_PRODUCT_IDS:
      dev = _open_xvf_usb(vendor_id, int(product_id))
      if dev is not None:
        break
  if dev is None:
    return None

  try:
    windex = _DOA_RESID
    wvalue = 0x80 | _DOA_CMDID
    length = _DOA_VALUE_COUNT * 2 + 1
    for _ in range(100):
      response = dev.ctrl_transfer(
        usb.util.CTRL_IN | usb.util.CTRL_TYPE_VENDOR | usb.util.CTRL_RECIPIENT_DEVICE,
        0,
        wvalue,
        windex,
        length,
        10_000,
      )
      status = int(response[0])
      if status == CONTROL_SUCCESS:
        azimuth, speech = struct.unpack_from("<HH", response, 1)
        return DoaReading(
          azimuth_deg=int(azimuth) % 360,
          speech_detected=bool(speech),
        )
      if status != SERVICER_COMMAND_RETRY:
        return None
      time.sleep(0.01)
    return None
  except usb.core.USBError:
    return None
  finally:
    if own_dev:
      _close_xvf_usb(dev)


def _xvf3800_firmware_reboot(vendor_id: int, product_id: int) -> bool:
  """Send XVF3800 REBOOT command (Seeed workaround for warm-boot USB hang)."""
  try:
    import usb.core
    import usb.util
  except ImportError:
    logger.debug("pyusb not installed; skipping XVF3800 firmware reboot")
    return False

  dev = _open_xvf_usb(vendor_id, product_id)
  if dev is None:
    return False

  try:
    # APPLICATION_SERVICER_RESID REBOOT: resid=48, cmdid=7, uint8 value 1
    dev.ctrl_transfer(
      usb.util.CTRL_OUT | usb.util.CTRL_TYPE_VENDOR | usb.util.CTRL_RECIPIENT_DEVICE,
      0,
      7,
      48,
      struct.pack("B", 1),
      100_000,
    )
    # Match xvf_host.py behavior, then verify device actually re-enumerates.
    pids = (int(product_id),)
    vanished = _wait_for_usb_state(vendor_id=vendor_id, product_ids=pids, present=False, timeout_s=2.0)
    reappeared = _wait_for_usb_state(vendor_id=vendor_id, product_ids=pids, present=True, timeout_s=5.0)
    if not vanished or not reappeared:
      logger.warning(
        "XVF3800 firmware reboot command sent but USB re-enumeration not observed "
        "(vanished=%s reappeared=%s)",
        vanished,
        reappeared,
      )
    return True
  except usb.core.USBError as exc:
    logger.warning("XVF3800 firmware reboot failed (pid=0x%04x): %s", product_id, exc)
    return False
  finally:
    _close_xvf_usb(dev)


def _usb_port_reset(device: RespeakerUsbDevice) -> bool:
  """Linux USB port reset via USBDEVFS_RESET."""
  if os.name != "posix" or not os.path.exists(device.dev_path):
    return False
  try:
    import fcntl

    with open(device.dev_path, "wb", buffering=0) as handle:
      fcntl.ioctl(handle, USBDEVFS_RESET, 0)
    return True
  except OSError as exc:
    logger.warning("USB port reset failed for %s: %s", device.dev_path, exc)
    return False


def _usb_authorized_cycle(device: RespeakerUsbDevice) -> bool:
  """Host-side deauthorize/reauthorize (closest software equivalent to replug)."""
  if os.name != "posix" or not device.sysfs_path:
    return False
  auth_path = os.path.join(device.sysfs_path, "authorized")
  if not os.path.exists(auth_path):
    return False
  try:
    with open(auth_path, "w", encoding="utf-8") as handle:
      handle.write("0")
    time.sleep(0.35)
    with open(auth_path, "w", encoding="utf-8") as handle:
      handle.write("1")
    return True
  except OSError as exc:
    logger.warning("USB authorize cycle failed for %s: %s", auth_path, exc)
    return False


def reset_respeaker_usb(
  device: RespeakerUsbDevice,
  *,
  method: str = "firmware",
) -> bool:
  method = method.lower()
  if method == "firmware":
    return _xvf3800_firmware_reboot(device.vendor_id, device.product_id)
  if method == "usb_port":
    return _usb_port_reset(device)
  if method == "usb_cycle":
    return _usb_authorized_cycle(device)
  if method == "both":
    fw = _xvf3800_firmware_reboot(device.vendor_id, device.product_id)
    if fw:
      return True
    if _usb_port_reset(device):
      return True
    return _usb_authorized_cycle(device)
  if method == "all":
    if _xvf3800_firmware_reboot(device.vendor_id, device.product_id):
      return True
    if _usb_port_reset(device):
      return True
    return _usb_authorized_cycle(device)
  raise ValueError(f"Unsupported respeaker reset_method: {method!r}")


def _resolve_input_device_name(config: dict[str, Any], device_index: int | None) -> str | None:
  from puppet.core.audio.capture import list_input_devices

  devices = list_input_devices()
  if device_index is not None:
    for dev in devices:
      if dev.index == int(device_index):
        return dev.name
    return None
  try:
    import pyaudio

    pa = pyaudio.PyAudio()
    try:
      info = pa.get_default_input_device_info()
      return str(info.get("name", ""))
    finally:
      pa.terminate()
  except Exception:
    return None


class RespeakerDoaMonitor:
  """Poll XVF3800 DoA and emit debug logs while the user is speaking."""

  def __init__(self, config: dict[str, Any]) -> None:
    rs_cfg = config.get("audio", {}).get("respeaker", {})
    self._enabled = bool(rs_cfg.get("doa_debug", False))
    self._poll_s = max(0.05, int(rs_cfg.get("doa_poll_ms", 250)) / 1000.0)
    self._vendor_id = int(rs_cfg.get("vendor_id", DEFAULT_RESPEAKER_VENDOR_ID))
    self._product_ids = tuple(int(pid) for pid in rs_cfg.get("product_ids", DEFAULT_RESPEAKER_PRODUCT_IDS))
    self._last_poll = 0.0
    self._last_log: tuple[int, bool] | None = None
    self._usb_dev: object | None = None
    self._active_product_id: int | None = None

  @property
  def enabled(self) -> bool:
    return self._enabled

  def close(self) -> None:
    _close_xvf_usb(self._usb_dev)
    self._usb_dev = None
    self._active_product_id = None

  def _ensure_usb(self) -> bool:
    if self._usb_dev is not None:
      return True
    for product_id in self._product_ids:
      dev = _open_xvf_usb(self._vendor_id, product_id)
      if dev is not None:
        self._usb_dev = dev
        self._active_product_id = product_id
        return True
    return False

  def maybe_log(self, *, speech_active: bool) -> None:
    if not self._enabled or not speech_active:
      return
    now = time.monotonic()
    if now - self._last_poll < self._poll_s:
      return
    self._last_poll = now
    if not self._ensure_usb():
      return
    reading = read_doa(
      vendor_id=self._vendor_id,
      product_ids=(self._active_product_id,) if self._active_product_id is not None else self._product_ids,
      dev=self._usb_dev,
    )
    if reading is None:
      return
    key = (reading.azimuth_deg, reading.speech_detected)
    if key == self._last_log:
      return
    self._last_log = key
    logger.debug(
      "DoA voice direction %d° (%s)%s",
      reading.azimuth_deg,
      reading.compass,
      " speech" if reading.speech_detected else "",
    )


def maybe_reset_respeaker_on_start(
  config: dict[str, Any],
  *,
  device_index: int | None = None,
) -> bool:
  """Reset ReSpeaker USB firmware/port before opening the mic, if configured."""
  audio_cfg = config.get("audio", {})
  rs_cfg = audio_cfg.get("respeaker", {})
  mode = str(rs_cfg.get("usb_reset_on_start", "never")).lower()
  if mode == "never":
    return False

  vendor_id = int(rs_cfg.get("vendor_id", DEFAULT_RESPEAKER_VENDOR_ID))
  product_ids = rs_cfg.get("product_ids", list(DEFAULT_RESPEAKER_PRODUCT_IDS))
  name_patterns = _normalize_patterns(rs_cfg.get("name_patterns"))
  method = str(rs_cfg.get("reset_method", "firmware"))
  settle_ms = max(0, int(rs_cfg.get("settle_ms", 1500)))

  usb_devices = find_respeaker_usb_devices(vendor_id=vendor_id, product_ids=product_ids)
  input_name = _resolve_input_device_name(config, device_index)
  name_match = bool(input_name and device_name_matches_respeaker(input_name, name_patterns))
  usb_match = bool(usb_devices)

  if mode == "auto" and not (name_match or usb_match):
    return False
  if mode not in ("auto", "always"):
    logger.warning("Unknown audio.respeaker.usb_reset_on_start=%r (use auto|always|never)", mode)
    return False
  if not usb_devices:
    if mode == "always":
      logger.warning(
        "audio.respeaker.usb_reset_on_start=always but no ReSpeaker USB device "
        "(vid=0x%04x) found",
        vendor_id,
      )
    return False

  target = usb_devices[0]
  logger.info(
    "Resetting ReSpeaker USB (bus=%03d dev=%03d pid=0x%04x method=%s%s)",
    target.bus,
    target.device,
    target.product_id,
    method,
    f", mic={input_name!r}" if input_name else "",
  )
  ok = reset_respeaker_usb(target, method=method)
  if not ok:
    logger.warning(
      "ReSpeaker reset did not complete; install pyusb for firmware reboot "
      "(pip install pyusb) or run as root for usb_port reset"
    )
    logger.warning(
      "If reset failed with Access denied/Permission denied, configure USB permissions "
      "(udev rule for 2886:001a + user in audio group). See docs/audio-pipeline.md "
      "section 'Linux permissions for software reset'."
    )
    return False
  if settle_ms > 0:
    logger.info("Waiting %dms for ReSpeaker to re-enumerate", settle_ms)
    time.sleep(settle_ms / 1000.0)
  return True
