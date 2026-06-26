# Native C/C++ dependencies

Sources are cloned here by `scripts/fetch_native_deps.sh` and built by `scripts/build_native.sh`.

| Path | Description |
|------|-------------|
| `parakeet.lock` | Pinned repo URL and commit for parakeet.cpp |
| `parakeet.cpp/` | Cloned source (gitignored, created on first build) |

To bump parakeet.cpp, update `parakeet.lock` and re-run `./scripts/build_native.sh`.
