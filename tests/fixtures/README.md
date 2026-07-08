# Test fixtures

| File | Source | Format |
|------|--------|--------|
| `jfk.wav` | [whisper.cpp samples](https://github.com/ggerganov/whisper.cpp/tree/master/samples) | 16 kHz mono 16-bit PCM |

Copy if missing:

```bash
cp ../whisper-cpp/samples/jfk.wav tests/fixtures/jfk.wav
```

Used by `tests/functional/test_stt_jfk.py` to verify parakeet streaming STT.
