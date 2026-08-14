# Configuration guide

Puppet loads YAML from this directory and deep-merges it at startup.

## Start here

1. Pick a **mic profile** in `default.yaml`:
   ```yaml
   profile: respeaker   # ReSpeaker XVF3800 (default)
   # profile: regular-mic
   ```
2. Pick a **language** in the eyes debug web (writes `config/language.active`) or run `puppet --language fr`. Missing file → German at brain start. Edit prompts in `config/language/en.yaml`, `fr.yaml`, `de.yaml`. If `../brain/de.yaml` exists, German uses that file; otherwise `language/de.yaml`.
3. Tune hardware in `puppet.yaml` → `mouth` (servo bus, angles, sync).
4. Everything else: only change when you know you need to (see tables below).

Merge order (later wins):

`default.yaml` → `profiles/<profile>.yaml` → `language/*.yaml` → `../brain/de.yaml` (if present) → `language.active` (or German) → `stt.yaml` → `llm.yaml` → `tts.yaml` → `vad.yaml` → `puppet.yaml` → env (`PUPPET_*`) → `--language`

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
| `language/en.yaml` `fr.yaml` `de.yaml` | Persona prompts per locale | Editing Kace's voice / tags in that language |
| `../brain/de.yaml` | Optional local German overlay (gitignored) | Local German prompt; if missing, `language/de.yaml` |
| `language.active` | One-line `en`/`fr`/`de` (gitignored) | Eyes debug web language picker; missing → German |
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

Requires Mosquitto + eyes debug_web (or equivalent) listening for captures.

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
| `follow.obstacle_m` | `0.5` | Sidestep if something else is closer |
| `follow.forward_dur_ms` | `500` | One roll pulse |
| `follow.backward_dur_ms` | `500` | One reverse pulse |

Voice: Gemma tags (`<<follow>>`, `<<seek>>`, `<<stop>>`, `<<back>>`). MQTT: `robot/play/cmd` (`follow` / `seek` / `idle` / `back`).

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
