from puppet.core.audio.alsa_latency import (
  _load_module_tokens,
  alsa_card_args_need_reload,
  fragment_bytes,
  parse_alsa_card_modules,
  parse_hw_params,
  parse_pulse_sinks,
  rewrite_alsa_card_args,
)

_HW_HUGE = """\
access: MMAP_INTERLEAVED
format: S16_LE
subformat: STD
channels: 2
rate: 16000 (16000/1)
period_size: 16000
buffer_size: 32000
"""

_SINKS = """\
Sink #1
	Name: alsa_output.platform-sound.analog-stereo
	Sample Specification: s16le 2ch 44100Hz
	Owner Module: 9
		alsa.device = "0"
		alsa.card = "2"

Sink #2
	Name: alsa_output.usb-Seeed_Studio_reSpeaker_XVF3800_4-Mic_Array_114993701261700006-00.analog-stereo
	Sample Specification: s16le 2ch 16000Hz
	Owner Module: 22
		alsa.device = "0"
		alsa.card = "0"
"""

_MODULES = """\
9	module-alsa-card	device_id="2" name="platform-sound" card_name="alsa_card.platform-sound" namereg_fail=false tsched=yes use_ucm=yes
22	module-alsa-card	device_id="0" name="usb-Seeed_Studio_reSpeaker_XVF3800_4-Mic_Array_114993701261700006-00" card_name="alsa_card.usb-Seeed" namereg_fail=false tsched=yes use_ucm=yes
"""


def test_parse_hw_params_one_second_period() -> None:
  params = parse_hw_params(_HW_HUGE)
  assert params is not None
  assert params.period_size == 16000
  assert params.buffer_size == 32000
  assert params.period_ms == 1000.0
  assert params.buffer_ms == 2000.0


def test_parse_hw_params_closed() -> None:
  assert parse_hw_params("closed\n") is None


def test_parse_pulse_respeaker_sink() -> None:
  sinks = parse_pulse_sinks(_SINKS)
  assert sinks[1].card == 0
  assert sinks[1].owner_module == 22
  assert sinks[1].rate == 16000
  assert sinks[1].channels == 2


def test_fragment_bytes_20ms_16k_stereo() -> None:
  assert fragment_bytes(rate=16000, channels=2, sample_format="s16le", period_ms=20) == 1280


def test_rewrite_disables_tsched_and_sets_short_period() -> None:
  modules = parse_alsa_card_modules(_MODULES)
  args = rewrite_alsa_card_args(modules[22], fragment_size=1280, fragments=3)
  assert "tsched=no" in args
  assert "tsched=yes" not in args
  assert "fragment_size=1280" in args
  assert "fragments=3" in args
  assert "device_id=\"0\"" in args


def test_load_module_tokens_keep_embedded_equals() -> None:
  tokens = _load_module_tokens(
    'device_id="0" card_properties="module-udev-detect.discovered=1" tsched=no'
  )
  assert "device_id=0" in tokens
  assert 'card_properties="module-udev-detect.discovered=1"' in tokens
  assert "tsched=no" in tokens


def test_need_reload_when_tsched_yes() -> None:
  assert alsa_card_args_need_reload(
    'device_id="0" tsched=yes',
    fragment_size=1280,
    fragments=3,
  )
  assert not alsa_card_args_need_reload(
    'device_id="0" tsched=no fragment_size=1280 fragments=3',
    fragment_size=1280,
    fragments=3,
  )
