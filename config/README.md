# Configuration guide

Puppet loads YAML from this directory and deep-merges it at startup.

## Start here

1. Pick a **mic profile** in `default.yaml`:
   ```yaml
   profile: respeaker   # ReSpeaker XVF3800 (default)
   # profile: regular-mic
   ```
2. Pick a **language** in Eye (writes `config/language.active`) or run `puppet --language fr_1`. Missing file → German (`de_1`) at brain start. Edit prompts in `config/language/en_1.yaml`, `fr_1.yaml`, `de_1.yaml`. Local overlays live in `../brain/` as `de_2.yaml`, `fr_2.yaml`, … and show up on Eye.
3. Tune hardware in `puppet.yaml` → `mouth` (servo bus, angles, sync).
4. Everything else: only change when you know you need to (see tables below).

Merge order (later wins):

`default.yaml` → `profiles/<profile>.yaml` → `language/*.yaml` → `../brain/de_2.yaml` (and other numbered overlays, if present) → `language.active` (or German `de_1`) → `stt.yaml` → `llm.yaml` → `tts.yaml` → `vad.yaml` → `puppet.yaml` → env (`PUPPET_*`) → `--language`

## Mic profiles

| | `respeaker` | `regular-mic` |
|---|-------------|---------------|
| Hardware | ReSpeaker XVF3800 (AEC + VAD on device) | Generic USB mic |
| STT feed | Continuous (`vad.gate_stt: false`) | Gated on speech (`vad.gate_stt: true`) |
| Barge-in | Off (`puppet.barge_in_enabled: false`) | On |
| USB reset on start | Yes (needs udev permissions — see docs) | No |
| Interrupt while speaking | Pause TTS → confirm with STT | Energy barge-in |

Profile files: `config/profiles/respeaker.yaml`, `config/profiles/regular-mic.yaml`.

## File map

| File | What it controls | Touch when… |
|------|------------------|-------------|
| `default.yaml` | Profile name, core audio I/O, logging, MQTT vision/capture | Changing mic type, log level, ALSA devices, vision capture |
| `profiles/*.yaml` | Mic-specific VAD / barge-in / ReSpeaker USB | Switching between ReSpeaker and regular mic |
| `language/en_1.yaml` `fr_1.yaml` `de_1.yaml` | Persona prompts per locale | Editing Kace's voice / tags in that language |
| `../brain/de_2.yaml` | Optional local overlay (gitignored) | Extra persona; Eye shows **DE 2** |
| `language.active` | One-line `en_1`/`fr_1`/`de_1`/`de_2` (gitignored) | Eye language picker; missing → German `de_1` |
| `play.speeds` | JSON follow/seek/forward (gitignored) | Eye speed sliders; live via `robot/play/speeds` |
| `stt.yaml` | Nemotron model path, streaming chunk size, GPU suspend | STT latency, model swap |
| `llm.yaml` | Model, context, temperature, `binding` (upstream/prism) | Reply style / model swap / LLM build |
| `tts.yaml` | Piper threads, phrase lead-in silence | Clipped first word, CPU load |
| `vad.yaml` | Silero model + silence timing | LLM fires too early / too late |
| `puppet.yaml` | Turn timing, echo guards, jaw servo | Lip sync, conversation flow, servo wiring |

## Vision MQTT (`default.yaml` → `mqtt`)

| Key | Default | Meaning |
|-----|---------|---------|
| `vision_enabled` | `true` | Subscribe to `robot/nav/scene` and inject `CameraJSON` when needed |
| `capture_before_reply` | `false` | If `true`, capture before every reply; if `false`, use the cache and let Gemma emit `<<look>>` |
| `capture_topic` | `robot/nav/capture` | Capture request topic (eyes must be running) |
| `scene_topic` | `robot/nav/scene` | Scene result topic |
| `capture_timeout_s` | `60` | Max wait for a matching scene |
| `capture_view` | `traverse` | Eyes view mode for the capture |
| `drive_cmd_topic` | `robot/drive/cmd` | One-shot turns when `face_speaker` is on |

Requires Mosquitto + Eye (or equivalent) listening for captures.

### Face speaker (ReSpeaker DoA → drive)

In `profiles/respeaker.yaml` → `audio.respeaker`:

| Key | Default | Meaning |
|-----|---------|---------|
| `face_speaker` | `true` | Latch DoA while speaking; turn toward speaker before reply |
| `doa_front_deg` | `60` | DoA when the speaker is ahead (~180° = to the robot's right) |
| `doa_deadband_deg` | `25` | Skip turns smaller than this |
| `doa_ms_per_deg` | `8` | Turn duration scaling (tune on hardware) |
| `doa_invert` | `false` | Flip left/right if mounting is mirrored |

Needs the drive MQTT bridge running.

## Play / movement (`default.yaml` → `play`)

See [movement.md](../../docs/movement.md) for the full follow / hide-and-seek design.

| Key | Default | Meaning |
|-----|---------|---------|
| `enabled` | `true` | Background play supervisor |
| `allow_motion` | `true` | Publish drive nudges (`false` = voice only) |
| `follow.stop_m` | `0.9` | Stop this far from the child |
| `follow.obstacle_m` | `0.8` | Sidestep if something else is closer |
| `follow.forward_dur_ms` | `700` | One roll pulse |
| `follow.backward_dur_ms` | `500` | One reverse pulse |
| `follow.turn_speed` | `125` | Follow / face-voice rotation (Eye **Follow turn**) |
| `follow.seek_turn_speed` | `125` | Hide-and-seek search rotation (Eye **Seek turn**) |
| `follow.forward_speed` | `105` | Roll speed (Eye **Forward**) |
| `follow.alive_jitter` | `0.25` | Extra wander/peek while following or seeking (`0` = metronome) |
| `follow.min_pulse_ms` | `320` | Floor on jittered follow/seek pulses |
| `follow.unstick_after` | `2` | Reverse after this many blocked ticks; reverse immediately if left and right are also tight |
| `follow.uturn_after` | `3` | Reverses without escaping a corner → U-turn toward the freer side |

Voice: Gemma tags (`<<follow>>`, `<<seek>>`, `<<stop>>`, `<<back>>`). MQTT: `robot/play/cmd` (`follow` / `seek` / `idle` / `back`). Eye sliders publish `robot/play/speeds` (live).

## Common tuning

### Lip sync (jaw servo)

`puppet.yaml` → `puppet.mouth`:

- `playback_delay_ms` — main knob (try 250–350 ms)
- `mode: word` — open/close per word
- Idle jaw uses PCA9685 `MODE1` SLEEP over I2C (tie `/OE` to GND); wakes while talking
- `debug: true` or `puppet --mouth-debug` — timeline logs

### Turn detection (when LLM starts)

`puppet.yaml`:

- `stt_gap_ms` — quiet STT after you stop talking
- `stt_tail_ms` — keep decoding trailing syllables

`vad.yaml`:

- `min_silence_duration_ms` — raise if LLM cuts you off mid-sentence

### ReSpeaker USB hang after reboot

`profiles/respeaker.yaml` → `audio.respeaker.usb_reset_on_start`.

If reset logs show **Permission denied**, add the udev rule in [docs/audio-pipeline.md](../docs/audio-pipeline.md).

### List / pick ALSA devices

```bash
python scripts/check_mic.py --list-devices
```

Then set `audio.input_device` in `default.yaml`.

## Environment overrides

`PUPPET_SECTION__KEY=value` maps to nested YAML, e.g.:

```bash
export PUPPET_LOGGING__LEVEL=INFO
export PUPPET_LANGUAGE__ACTIVE=de
```
