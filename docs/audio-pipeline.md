# Audio pipeline

## Sample rates

| Stage | Typical rate |
|-------|----------------|
| Mic / STT (parakeet) | 16 kHz mono float32 |
| TTS (Piper) | 22.05 kHz mono int16 |

Default mic profile is `respeaker` (`config/default.yaml`). See [config/README.md](../config/README.md) for the full file map. ReSpeaker gates STT on Silero and barges in by pausing TTS, then decoding.

## Voice activity detection (Silero VAD)

ReSpeaker firmware already flags speech for **direction of arrival** (DoA / face-the-speaker). That hardware bit is **not** the STT gate. Silero runs on the same 16 kHz PCM that Nemotron decodes, with hysteresis (start ≥ threshold, end after `min_silence_duration_ms`), so the brain knows when an utterance **starts and ends**. The LLM waits for Silero `END` plus STT gap/tail — that is why Silero stays on even though the array has its own VAD.

- **STT gating** (`vad.gate_stt`): when `true` (ReSpeaker default), Nemotron is fed only while Silero says speech, plus a short tail after `END`. Silence and post-reply echo are **not** decoded. On `START` the streaming decoder is reset and ~400 ms of pre-roll is flushed in, so leftover TTS is not sitting in the 4.5 s left context.
- **Barge-in during TTS** (`audio.respeaker.interrupt_while_speaking`): Silero speech while Piper is playing → holdoff (ignore bleed) → **pause TTS**, reset STT, decode. If the text is the robot, playback resumes and a cooldown blocks another pause on the same bleed. If it is the user, the reply is cancelled and the new transcript is kept. RMS barge-in (`puppet.barge_in_enabled`) stays off on ReSpeaker because AEC residual is loud.
- Chunks are buffered internally to Silero's required **512-sample windows** (32 ms at 16 kHz).

Mic-specific VAD/barge-in defaults live in `config/profiles/respeaker.yaml` or `config/profiles/regular-mic.yaml`. Silence timing in `config/vad.yaml`:

| Key | Default | Description |
|-----|---------|-------------|
| `threshold` | 0.25 | Speech probability to open a turn (end at threshold − 0.15) |
| `min_silence_duration_ms` | 450 | Silence before VAD declares speech ended |
| `force_cpu` | true | Keep VAD on CPU (leave GPU for LLM) |

For a regular USB mic, set `profile: regular-mic` in `config/default.yaml` and use PulseAudio/PipeWire AEC.

Download models once: `./scripts/download_models.sh`

### ReSpeaker XVF3800 USB reset

After a warm host reboot, the ReSpeaker can enumerate but deliver broken USB audio until its firmware is rebooted (Seeed workaround). Puppet can do this automatically before opening the mic:

See `config/profiles/respeaker.yaml` for USB reset and interrupt settings.

`usb_cycle` toggles Linux sysfs `authorized` (0→1), which is often closer to a physical unplug/replug than a plain USBDEVFS reset.

During a reply on ReSpeaker, Nemotron is **not** fed while Piper talks. Silero speech after a short holdoff pauses TTS, resets STT, then decodes. The reply is cancelled only when that text does **not** match the current TTS phrase (`puppet.interrupt_min_chars`); otherwise playback resumes with a cooldown so speaker bleed cannot chop the same sentence. After a short canned play line (“found you” / hide-and-seek count), overlapping speech is remembered so a greeting is not dropped. Count-to-ten is not barge-in paused (speaker bleed was chopping the countdown and blocking the search). Cancelling seek aborts that announce so a button restart does not wait on the leftover count.

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
- **Interrupts**: VAD + STT classify noise (restart same context) vs real speech (append to draft, regenerate)

### Streaming config (`config/puppet.yaml`)

| Key | Default | Role |
|-----|---------|------|
| `stt_gap_ms` | 200 | Quiet STT after VAD end before LLM starts |
| `stt_tail_ms` | 800 | Keep feeding STT after VAD end (covers 320ms Nemotron chunk) |
| `min_user_chars` | 3 | Minimum draft length to trigger LLM |
| `restart_on_partial` | true | Restart LLM when new STT words arrive during generation |
| `interrupt_min_chars` | 4 | STT length to treat interrupt as real speech |
| `interrupt_eval_ms` | 700 | Window to capture interrupt speech |
| `barge_in_grace_ms` | 1500 | Ignore barge-in at the start of a reply |
| `fresh_speech_timeout_ms` | 2500 | Reopen mic if a greeting overlapped TTS |

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
