# Architecture

Puppet is a **single-process, modular monolith** for low-latency voice interaction on embedded Linux (Jetson).

## Components

| Module | Backend | Integration |
|--------|---------|-------------|
| `stt/` | parakeet.cpp | pybind11 (`puppet_parakeet`) |
| `llm/` | llama.cpp | llama-cpp-python |
| `tts/` | Piper | piper-tts Python API |
| `orchestrator/` | — | State machine, turn-taking |
| `core/audio/` | — | Capture, playback, AEC, Silero VAD (onnxruntime) |

## Data flow

```
Mic → Silero VAD → AEC → STT (streaming partials → draft_user)
                              ↓ gap (stt_gap_ms)
                         LLM thread (token stream)
                              ↓ phrase boundaries
                         Piper TTS (streaming phrases)
                              ↓
                         Speaker + AEC reference buffer
```

**Interrupts** while THINKING/SPEAKING:
- **Noise** (VAD but no STT text): restore snapshot, restart generation
- **Speech**: append interrupt STT to draft, cancel and regenerate

## Why not MQTT for audio?

MQTT is suitable for **control events** (optional, disabled by default). Real-time PCM and echo-cancellation reference signals stay **in-process** via shared buffers.

## Config

YAML files in `config/` are merged at startup. Environment overrides use `PUPPET_<SECTION>__<KEY>` (e.g. `PUPPET_STT__MODEL_PATH`).

## GPU memory

parakeet.cpp and llama.cpp may both use the GPU. With `stt.suspend_during_llm: true` (default), parakeet unloads from GPU while llama-server decodes, then reloads for the next utterance. Profile on your Jetson and adjust quantization (`Q4` LLM, smaller STT model) if VRAM is tight.
