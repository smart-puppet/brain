# Audio pipeline

## Sample rates

| Stage | Typical rate |
|-------|----------------|
| Mic / STT (parakeet) | 16 kHz mono float32 |
| TTS (Piper) | 22.05 kHz mono int16 |
| AEC reference buffer | 16 kHz (matches mic/STT; TTS resampled on write) |

Resampling happens in `core/audio/aec.py` when writing TTS chunks to the reference buffer.

## Voice activity detection (Silero VAD)

Silero VAD runs on 16 kHz mic audio before STT:

- **STT gating** (`vad.gate_stt: true`): audio is sent to parakeet only during speech segments.
- **Barge-in** (`vad.barge_in: true`): user speech during **TTS playback** interrupts the assistant (not during LLM thinking)
- Chunks are buffered internally to Silero's required **512-sample windows** (32 ms at 16 kHz).

Configure in `config/vad.yaml`:

| Key | Default | Description |
|-----|---------|-------------|
| `threshold` | 0.5 | Speech probability threshold |
| `min_silence_duration_ms` | 700 | Silence before VAD declares speech ended |
| `force_cpu` | true | Keep VAD on CPU (leave GPU for LLM) |

Download models once: `./scripts/download_models.sh`

## Echo cancellation / barge-in

Requires system package: `sudo apt install libspeexdsp-dev`

1. Every TTS chunk is resampled to 16 kHz and written to `AudioReference` as it plays.
2. **SpeexDSP** cancels echo using the mic + speaker reference.
3. AEC runs only during TTS playback; raw mic is used while listening.
4. Clean audio feeds VAD and STT during playback. Echo-like chunks can be dropped (`suppress_stt_on_echo`).

Tune in `config/aec.yaml`:

| Key | Default | Role |
|-----|---------|------|
| `frame_size` | 160 | Speex frame (10 ms @ 16 kHz) |
| `filter_length` | 2048 | Echo tail length (~128 ms) |
| `playback_delay_ms` | 120 | Align reference with mic (80–200) |

## Latency budget

- **STT**: parakeet streaming — partial words appended to `conversation.draft_user` immediately
- **LLM prefill** (`llm.prefill_at_generation`): one sync `max_tokens: 0` call after speech, before decode
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

### LLM prefill (`config/llm.yaml`, llama-server)

| Key | Default | Role |
|-----|---------|------|
| `prefill_at_generation` | false | Off by default — same single-request path as `test_llm` |
| `prefill_during_listen` | false | Background prefill while STT runs (GPU contention on Jetson) |

After each completed reply, `run_puppet` logs one INFO line with a latency bar and LLM perf (same style as `test_llm.py`):

```
latency speech→audio 1100ms [████▓▓▓░░░░░░░░░░░░] wait 400ms | ttft 500ms | buffer 200ms | play 400ms | total 1500ms | llm_wall 2353ms  |  ctx ...
```

- **speech→audio** — user stopped speaking (VAD end) → first audio from the speaker (the main perceived latency)
- **wait** — speech end → LLM request (STT tail + gap)
- **ttft** — LLM request → first token (compare to `test_llm`)
- **buffer** — first token → first speaker (phrase batching + Piper synth)
- **play** — first audio → end of reply playback
- **llm_wall** — raw server stream time (should be ~2–3s like `test_llm`)

At DEBUG, the first `tts playing` trace line also shows ms since speech end.

## Functional test fixtures

Place WAV files in `tests/functional/fixtures/`:

- `sample_utterance.wav` — 16 kHz mono speech
- `tts_reference.wav` — optional reference for AEC tests
