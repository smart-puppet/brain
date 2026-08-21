# Audio pipeline

## Sample rates

| Stage | Typical rate |
|-------|----------------|
| Mic / STT (parakeet) | 16 kHz mono float32 |
| TTS (Piper) | 22.05 kHz mono int16 → stereo on ReSpeaker |

Default mic profile is `respeaker` (`config/default.yaml`). See [config/README.md](../config/README.md) for the full file map.

## ReSpeaker XVF3800 + hardware AEC (`trust_aec`)

Play TTS on the array’s **stereo analog sink** so the XVF3800 far-end reference matches the speaker. Capture is AEC-cleaned — treat residual TTS as gone.

With `audio.respeaker.trust_aec: true` (respeaker profile default):

- **No post-reply echo / fresh-speech gate** — mic opens immediately after a reply (the old 2.5 s `Fresh-speech wait` was dropping real turns).
- **No pause/probe barge-in** — Silero `START` while the robot talks **cancels** the reply (no holdoff / echo-match / resume).
- **RMS barge-in** stays off; that path is for `regular-mic` only.
- Silero still gates STT for turn boundaries; the firmware speech bit backs DoA and STT feed when Silero is slow.

## Voice activity detection (Silero VAD)

ReSpeaker firmware flags speech for **DoA / face-the-speaker**. Silero runs on the same 16 kHz PCM Nemotron sees, with hysteresis, so the brain knows utterance **start and end**. The LLM waits for Silero `END` plus STT gap/tail — or starts after STT has been quiet ~0.8 s if VAD stays open.

- **STT gating** (`vad.gate_stt`): feed Nemotron only while Silero (or firmware speech) says speech, plus a short tail after `END`.
- Chunks are buffered to Silero’s **512-sample windows** (32 ms at 16 kHz).

Mic-specific defaults: `config/profiles/respeaker.yaml` or `regular-mic.yaml`. Timing in `config/vad.yaml`:

| Key | Default | Description |
|-----|---------|-------------|
| `threshold` | 0.25 | Speech probability to open a turn (end is threshold − 0.15) |
| `min_silence_duration_ms` | 450 | Silence before VAD declares speech ended |
| `force_cpu` | true | Keep VAD on CPU (leave GPU for LLM) |

For a generic USB mic, set `profile: regular-mic` (software barge-in + echo quiet gate).

Download models once: `./scripts/download_models.sh`

### ReSpeaker XVF3800 USB reset

After a warm host reboot, the ReSpeaker can enumerate but deliver broken USB audio until its firmware is rebooted (Seeed workaround). Puppet can do this automatically before opening the mic:

See `config/profiles/respeaker.yaml` for USB reset settings.

`usb_cycle` toggles Linux sysfs `authorized` (0→1), which is often closer to a physical unplug/replug than a plain USBDEVFS reset.

Canned play announces (hide-and-seek count / “found you”) still skip AEC cancel so a short residual cannot abort the countdown. Cancelling seek aborts that announce so a button restart does not wait on the leftover count.

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

### Face the speaker (DoA → drive)

With the ReSpeaker profile, Puppet can turn toward the child before answering:

```yaml
audio:
  respeaker:
    face_speaker: true
    doa_front_deg: 60      # ~60° = in front; ~180° = to the robot's right
    doa_deadband_deg: 25
    doa_ms_per_deg: 8      # tune spin duration
    doa_invert: false
```

Brain latches DoA while you speak, then publishes a one-shot `turn_left` / `turn_right` on `robot/drive/cmd` when the reply starts (needs the drive MQTT bridge running). Set `doa_invert: true` if left/right feel mirrored.

### VAD disabled

If you disable VAD (`vad.enabled: false`), speech detection falls back to mic RMS (`audio.speech_rms_threshold`). Without this, STT was reset on every chunk after each reply and Nemotron never decoded speech.

## Latency budget

- **STT**: parakeet streaming — partial words appended to `conversation.draft_user` immediately
- **LLM**: starts only after **VAD reports silence** (plus STT gap/tail), so generation is not cancelled mid-utterance
- **TTS**: phrase-level Piper streaming as soon as sentence boundaries appear in LLM output
- **Interrupts (ReSpeaker AEC)**: Silero speech during TTS cancels the reply
- **Interrupts (regular-mic)**: RMS barge-in and/or pause/probe paths in the profile

### Streaming config (`config/puppet.yaml`)

| Key | Default | Role |
|-----|---------|------|
| `stt_gap_ms` | 200 | Quiet STT after VAD end before LLM starts |
| `stt_tail_ms` | 800 | Keep feeding STT after VAD end (covers 320ms Nemotron chunk) |
| `min_user_chars` | 3 | Minimum draft length to trigger LLM |
| `restart_on_partial` | true | Restart LLM when new STT words arrive during generation |
| `echo_quiet_ms` / `fresh_speech_timeout_ms` | 0 on ReSpeaker | Post-reply bleed gate (regular-mic); disabled with `trust_aec` |
| `barge_in_grace_ms` | 1500 | RMS barge-in only (`regular-mic`) |

After each completed reply, `run_puppet` logs one INFO line. The headline and the three bar segments are the same number (VAD end → first audible speech):

```
latency 1100ms [█████▓▓▓▓▓░░░]  wait 400ms | llm 500ms | tts 200ms | llm_wall 2353ms  |  ctx ...
```

- **wait** (█) — Silero end → LLM request (STT tail + gap)
- **llm** (▓) — LLM request → first token
- **tts** (░) — first token → first audible speech (Piper + lead-in + ALSA buffer)
- **llm_wall** — llama.cpp prompt + generation time (compare to `test_llm`)

Capture uses a 20 ms ALSA period (`audio.chunk_ms`). ReSpeaker playback is **stereo at Piper's 22.05 kHz** on the XVF3800 USB sink (Pulse resamples to the array's native 16 kHz). Hardware AEC only cancels if TTS uses that same analog-stereo sink — not the Jetson onboard/`platform-sound` output. Opening PortAudio at 16 kHz native stalls Pulse writes, so the ready prompt never finishes and the mic loop never starts. On start the brain pins Pulse default sink/source to the ReSpeaker and reloads the ALSA card with `tsched=off` and a 20 ms period × 3 (~60 ms buffer). Pulse clients still use 20 ms fragments (`PULSE_LATENCY_MSEC`). Startup logs report `ALSA playback card…` and `Speaker opened: … 22050 Hz 2ch`.

At DEBUG, the first `tts playing` trace line also shows ms since speech end.

## Functional test fixtures

Place WAV files in `tests/functional/fixtures/`:

- `sample_utterance.wav` — 16 kHz mono speech
