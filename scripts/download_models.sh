#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODELS_DIR="$ROOT/models"
PIPER_BASE="${PIPER_BASE:-https://huggingface.co/rhasspy/piper-voices/resolve/main}"

# Override any URL/path with env vars if needed.
VAD_PATH="${VAD_PATH:-$MODELS_DIR/vad/silero_vad.onnx}"
VAD_URL="${VAD_URL:-https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx}"

STT_PATH="${STT_PATH:-$MODELS_DIR/stt/nemotron-3.5-asr-streaming-0.6b-q8_0.gguf}"
STT_URL="${STT_URL:-https://huggingface.co/mudler/parakeet-cpp-gguf/resolve/main/nemotron-3.5-asr-streaming-0.6b-q8_0.gguf}"

LLM_PATH="${LLM_PATH:-$MODELS_DIR/llm/Bonsai-8B-Q1_0.gguf}"
LLM_URL="${LLM_URL:-https://huggingface.co/prism-ml/Bonsai-8B-gguf/resolve/main/Bonsai-8B-Q1_0.gguf}"

mkdir -p "$MODELS_DIR/vad" "$MODELS_DIR/stt" "$MODELS_DIR/llm" "$MODELS_DIR/tts"

download_if_missing() {
  local dest="$1"
  local url="$2"

  if [[ -s "$dest" ]]; then
    echo "Already present: $dest"
    return 0
  fi

  echo "Downloading: $dest"
  curl -fL --retry 3 --retry-delay 2 "$url" -o "$dest"
}

download_tts_voice() {
  local rel_path="$1"   # e.g. en/en_US/ryan/medium/en_US-ryan-medium
  local name="${rel_path##*/}"
  local model_dest="$MODELS_DIR/tts/${name}.onnx"
  local config_dest="$MODELS_DIR/tts/${name}.onnx.json"
  download_if_missing "$model_dest" "$PIPER_BASE/$rel_path.onnx"
  download_if_missing "$config_dest" "$PIPER_BASE/$rel_path.onnx.json"
}

download_if_missing "$VAD_PATH" "$VAD_URL"
download_if_missing "$STT_PATH" "$STT_URL"
download_if_missing "$LLM_PATH" "$LLM_URL"

# Piper voices for en / fr / de (see config/language.yaml)
download_tts_voice "en/en_US/ryan/medium/en_US-ryan-medium"
download_tts_voice "fr/fr_FR/siwis/medium/fr_FR-siwis-medium"
download_tts_voice "de/de_DE/thorsten/medium/de_DE-thorsten-medium"

echo "All requested models are present."
