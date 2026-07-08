# Native C/C++ dependencies

Sources are cloned here by fetch/build scripts.

| Path | Description |
|------|-------------|
| `parakeet.lock` | Pinned repo URL and commit for parakeet.cpp |
| `parakeet.cpp/` | Cloned parakeet source (gitignored) |
| `llama-cpp-python.lock` | Pinned llama-cpp-python tag for LLM builds |
| `prism-llama.lock` | Pinned PrismML llama.cpp fork commit |
| `llama-binding.lock` | Last LLM build: `upstream` or `prism` (gitignored) |

## Build scripts

| Script | Purpose |
|--------|---------|
| `./scripts/build_parakeet.sh` | parakeet.cpp + `puppet_parakeet` binding |
| `./scripts/build_llama.sh` | llama-cpp-python + **upstream** llama.cpp |
| `./scripts/build_llama_prism.sh` | llama-cpp-python + **Prism** fork (Q2_0) |

### LLM builds overwrite each other

> **WARNING:** `build_llama.sh` and `build_llama_prism.sh` share one `.venv` and one `vendor/llama-cpp-python/` tree. Running either script **replaces** the previous `llama_cpp` install — they cannot coexist. Expect ~30–45 min to rebuild when switching bindings.

Both scripts print a large warning before compiling if `llama_cpp` is already present. Set `llm.binding` in `config/llm.yaml` to match whichever script you ran last (`vendor/native/llama-binding.lock`).
