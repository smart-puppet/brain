# Audio pipeline

## Sample rates

| Stage | Typical rate |
|-------|----------------|
| Mic / STT (parakeet) | 16 kHz mono float32 |
| TTS (Piper) | 22.05 kHz mono int16 |

Default mic profile is `respeaker` (`config/default.yaml`). See [config/README.md](../config/README.md) for the full file map. ReSpeaker uses continuous STT decode with barge-in disabled.

## Voice activity detection (Silero VAD)

Silero VAD runs on 16 kHz mic audio before STT:

- **STT gating** (`vad.gate_stt`): when `false` (default), STT keeps decoding continuously.
- **Barge-in** (`puppet.barge_in_enabled`): when `true`, user speech during playback can interrupt the assistant.
- Chunks are buffered internally to Silero's required **512-sample windows** (32 ms at 16 kHz).

Mic-specific VAD/barge-in defaults live in `config/profiles/respeaker.yaml` or `config/profiles/regular-mic.yaml`. Silence timing in `config/vad.yaml`:

| Key | Default | Description |
|-----|---------|-------------|
| `threshold` | 0.3 | Speech probability threshold |
| `min_silence_duration_ms` | 350 | Silence before VAD declares speech ended |
| `force_cpu` | true | Keep VAD on CPU (leave GPU for LLM) |

For a regular USB mic, set `profile: regular-mic` in `config/default.yaml` and use PulseAudio/PipeWire AEC.

Download models once: `./scripts/download_models.sh`

### ReSpeaker XVF3800 USB reset

After a warm host reboot, the ReSpeaker can enumerate but deliver broken USB audio until its firmware is rebooted (Seeed workaround). Puppet can do this automatically before opening the mic:

See `config/profiles/respeaker.yaml` for USB reset and interrupt settings.

`usb_cycle` toggles Linux sysfs `authorized` (0→1), which is often closer to a physical unplug/replug than a plain USBDEVFS reset.

During a reply on ReSpeaker, Puppet now uses a two-step interruption policy:

1. Pause TTS immediately when speech is detected.
2. If STT decodes text (`puppet.interrupt_min_chars`), cancel the current generation and continue with appended user text.
   If no text is decoded before `audio.respeaker.interrupt_timeout_ms`, treat it as noise and resume playback.

#### Linux permissions for software reset (important)

If logs show errors like `Access denied (insufficient permissions)` or `Permission denied` for:

- firmware reboot (pyusb),
- `/dev/bus/usb/...` (usb_port reset),
- `/sys/bus/usb/devices/.../authorized` (usb_cycle),

then reset is blocked by OS permissions (not by Puppet logic).

Create a udev rule:

```bash
sudo tee /etc/udev/rules.d/99-respeaker-xvf3800.rules >/dev/null <<'EOF'
SUBSYSTEM=="usb", ATTR{idVendor}=="2886", ATTR{idProduct}=="001a", MODE="0660", GROUP="audio", TAG+="uaccess"
EOF

sudo udevadm control --reload-rules
sudo udevadm trigger
```

Verify your user is in the `audio` group (and re-login if you just added it):

```bash
groups
sudo usermod -aG audio $USER
```

Then unplug/replug the mic once and restart Puppet.

Temporary workaround while permissions are unresolved:

```yaml
audio:
  respeaker:
    usb_reset_on_start: never
```

Set `audio.input_device` to the ReSpeaker ALSA/Pulse device. Use a USB 3.0 (xHCI) port if capture is silent despite a moving VU meter.

Enable direction-of-arrival debug (requires `pyusb`):

```yaml
audio:
  respeaker:
    doa_debug: true
    doa_poll_ms: 250
```

Logs look like `DoA voice direction 127° (SE) speech` on the `puppet.respeaker` logger (visible when `logging.level: DEBUG`).

### VAD disabled

If you disable VAD (`vad.enabled: false`), speech detection falls back to mic RMS (`audio.speech_rms_threshold`). Without this, STT was reset on every chunk after each reply and Nemotron never decoded speech.

## Latency budget

- **STT**: parakeet streaming — partial words appended to `conversation.draft_user` immediately
- **LLM**: starts only after **VAD reports silence** (plus STT gap/tail), so generation is not cancelled mid-utterance
- **TTS**: phrase-level Piper streaming as soon as sentence boundaries appear in LLM output
- **Interrupts**: VAD + STT classify noise (restart same context) vs real speech (append to draft, regenerate)

### Streaming config (`config/puppet.yaml`)

| Key | Default | Role |
|-----|---------|------|
| `stt_gap_ms` | 600 | Quiet STT after VAD end before LLM starts |
| `stt_tail_ms` | 1000 | Keep feeding STT after VAD end (captures trailing words) |
| `min_user_chars` | 3 | Minimum draft length to trigger LLM |
| `restart_on_partial` | true | Restart LLM when new STT words arrive during generation |
| `interrupt_min_chars` | 2 | STT length to treat interrupt as real speech |
| `interrupt_eval_ms` | 700 | Window to capture interrupt speech |

After each completed reply, `run_puppet` logs one INFO line with a latency bar and LLM perf (same style as `test_llm.py`):

```
latency speech→audio 1100ms [████▓▓▓░░░░░░░░░░░░] wait 400ms | ttft 500ms | buffer 200ms | play 400ms | total 1500ms | llm_wall 2353ms  |  ctx ...
```

- **speech→audio** — user stopped speaking (VAD end) → first audio from the speaker (the main perceived latency)
- **wait** — speech end → LLM request (STT tail + gap)
- **ttft** — LLM request → first token (compare to `test_llm`)
- **buffer** — first token → first speaker (phrase batching + Piper synth)
- **play** — first audio → end of reply playback
- **llm_wall** — LLM prompt + generation decode time from llama.cpp perf counters

At DEBUG, the first `tts playing` trace line also shows ms since speech end.

## Functional test fixtures

Place WAV files in `tests/functional/fixtures/`:

- `sample_utterance.wav` — 16 kHz mono speech
