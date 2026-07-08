# Configuration guide

Puppet loads YAML from this directory and deep-merges it at startup.

## Start here

1. Pick a **mic profile** in `default.yaml`:
   ```yaml
   profile: respeaker   # ReSpeaker XVF3800 (default)
   # profile: regular-mic
   ```
2. Pick a **language** in `language.yaml` (`language.active`) or run `puppet --language fr`.
3. Tune hardware in `puppet.yaml` → `mouth` (servo bus, angles, sync).
4. Everything else: only change when you know you need to (see tables below).

Merge order (later wins):

`default.yaml` → `profiles/<profile>.yaml` → `language.yaml` → `stt.yaml` → `llm.yaml` → `tts.yaml` → `vad.yaml` → `puppet.yaml` → env (`PUPPET_*`)

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
| `default.yaml` | Profile name, core audio I/O, logging, MQTT | Changing mic type, log level, ALSA devices |
| `profiles/*.yaml` | Mic-specific VAD / barge-in / ReSpeaker USB | Switching between ReSpeaker and regular mic |
| `language.yaml` | Active locale + per-language STT/TTS/LLM prompts | Adding a language or editing Kace's persona |
| `stt.yaml` | Nemotron model path, streaming chunk size, GPU suspend | STT latency, model swap |
| `llm.yaml` | Model, context, temperature, `binding` (upstream/prism) | Reply style / model swap / LLM build |
| `tts.yaml` | Piper threads, phrase lead-in silence | Clipped first word, CPU load |
| `vad.yaml` | Silero model + silence timing | LLM fires too early / too late |
| `puppet.yaml` | Turn timing, echo guards, jaw servo | Lip sync, conversation flow, servo wiring |

## Common tuning

### Lip sync (jaw servo)

`puppet.yaml` → `puppet.mouth`:

- `playback_delay_ms` — main knob (try 250–350 ms)
- `mode: word` — open/close per word
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
