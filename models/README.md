# Models

Download all required models automatically:

```bash
./scripts/download_models.sh
```

Expected paths (see `config/language/*.yaml` for per-language TTS):

| Path | Description |
|------|-------------|
| `models/vad/silero_vad.onnx` | Silero VAD model |
| `models/stt/nemotron-3.5-asr-streaming-0.6b-q8_0.gguf` | parakeet streaming STT (multilingual) |
| `models/llm/Gemma-4-E2B-it-fraQtl-HiFi-Q4_K_M.gguf` | current LLM (`config/llm.yaml`; Gemma 4 E2B) |
| `models/llm/Qwen3.5-2B-Q4_K_M.gguf` | optional Unsloth Qwen3.5 2B |
| `models/llm/Qwen3.5-0.8B-Q4_K_M.gguf` | optional Unsloth Qwen3.5 0.8B |
| `models/llm/Ternary-Bonsai-4B-Q2_0.gguf` | optional Prism fork (`./scripts/build_llama_prism.sh`) |
| `models/llm/Bonsai-8B-Q1_0.gguf` | optional; standard pip `llama-cpp-python` |
| `models/tts/en_US-ryan-medium.onnx` (+ `.json`) | Piper English voice |
| `models/tts/fr_FR-siwis-medium.onnx` (+ `.json`) | Piper French voice |
| `models/tts/de_DE-thorsten-medium.onnx` (+ `.json`) | Piper German voice |

`download_models.sh` skips files that already exist and supports URL/path overrides via environment variables.
