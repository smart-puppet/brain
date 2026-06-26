#!/usr/bin/env python3
"""Patch llama-cpp-python ctypes bindings for the PrismML llama.cpp fork."""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LLAMA_CPP_PY = ROOT / "vendor" / "llama-cpp-python" / "llama_cpp" / "llama_cpp.py"


def patch_llama_cpp_py(path: Path) -> None:
  text = path.read_text(encoding="utf-8")
  original = text

  # Prism fork does not export llama_n_rs_seq.
  text = text.replace(
    '@ctypes_function("llama_n_rs_seq", [llama_context_p_ctypes], ctypes.c_uint32)',
    '@ctypes_function("llama_n_rs_seq", [llama_context_p_ctypes], ctypes.c_uint32, enabled=False)',
  )

  # llama_context_params in Prism matches upstream before n_rs_seq / ctx_type / ctx_other.
  old_fields = """    _fields_ = [
        ("n_ctx", ctypes.c_uint32),
        ("n_batch", ctypes.c_uint32),
        ("n_ubatch", ctypes.c_uint32),
        ("n_seq_max", ctypes.c_uint32),
        ("n_rs_seq", ctypes.c_uint32),
        ("n_outputs_max", ctypes.c_uint32),
        ("n_threads", ctypes.c_int32),
        ("n_threads_batch", ctypes.c_int32),
        ("ctx_type", ctypes.c_int),
        ("rope_scaling_type", ctypes.c_int),
        ("pooling_type", ctypes.c_int),
        ("attention_type", ctypes.c_int),
        ("flash_attn_type", ctypes.c_int),
        ("rope_freq_base", ctypes.c_float),
        ("rope_freq_scale", ctypes.c_float),
        ("yarn_ext_factor", ctypes.c_float),
        ("yarn_attn_factor", ctypes.c_float),
        ("yarn_beta_fast", ctypes.c_float),
        ("yarn_beta_slow", ctypes.c_float),
        ("yarn_orig_ctx", ctypes.c_uint32),
        ("defrag_thold", ctypes.c_float),
        ("cb_eval", ggml_backend_sched_eval_callback),
        ("cb_eval_user_data", ctypes.c_void_p),
        ("type_k", ctypes.c_int),
        ("type_v", ctypes.c_int),
        ("abort_callback", ggml_abort_callback),
        ("abort_callback_data", ctypes.c_void_p),
        ("embeddings", ctypes.c_bool),
        ("offload_kqv", ctypes.c_bool),
        ("no_perf", ctypes.c_bool),
        ("op_offload", ctypes.c_bool),
        ("swa_full", ctypes.c_bool),
        ("kv_unified", ctypes.c_bool),
        ("samplers", ctypes.POINTER(llama_sampler_seq_config)),
        ("n_samplers", ctypes.c_size_t),
        ("ctx_other", llama_context_p_ctypes),
    ]"""

  new_fields = """    _fields_ = [
        ("n_ctx", ctypes.c_uint32),
        ("n_batch", ctypes.c_uint32),
        ("n_ubatch", ctypes.c_uint32),
        ("n_seq_max", ctypes.c_uint32),
        ("n_threads", ctypes.c_int32),
        ("n_threads_batch", ctypes.c_int32),
        ("rope_scaling_type", ctypes.c_int),
        ("pooling_type", ctypes.c_int),
        ("attention_type", ctypes.c_int),
        ("flash_attn_type", ctypes.c_int),
        ("rope_freq_base", ctypes.c_float),
        ("rope_freq_scale", ctypes.c_float),
        ("yarn_ext_factor", ctypes.c_float),
        ("yarn_attn_factor", ctypes.c_float),
        ("yarn_beta_fast", ctypes.c_float),
        ("yarn_beta_slow", ctypes.c_float),
        ("yarn_orig_ctx", ctypes.c_uint32),
        ("defrag_thold", ctypes.c_float),
        ("cb_eval", ggml_backend_sched_eval_callback),
        ("cb_eval_user_data", ctypes.c_void_p),
        ("type_k", ctypes.c_int),
        ("type_v", ctypes.c_int),
        ("abort_callback", ggml_abort_callback),
        ("abort_callback_data", ctypes.c_void_p),
        ("embeddings", ctypes.c_bool),
        ("offload_kqv", ctypes.c_bool),
        ("no_perf", ctypes.c_bool),
        ("op_offload", ctypes.c_bool),
        ("swa_full", ctypes.c_bool),
        ("kv_unified", ctypes.c_bool),
        ("samplers", ctypes.POINTER(llama_sampler_seq_config)),
        ("n_samplers", ctypes.c_size_t),
    ]"""

  if old_fields not in text:
    if new_fields in text:
      print(f"Already patched: {path}")
      return
    raise SystemExit(f"Could not find llama_context_params fields block in {path}")

  text = text.replace(old_fields, new_fields)

  # Drop Prism-only TYPE_CHECKING attrs for removed fields.
  for attr in ("n_rs_seq", "n_outputs_max", "ctx_type", "ctx_other"):
    text = re.sub(rf"        {attr}: .*\n", "", text)

  if text == original:
    print(f"No changes needed: {path}")
    return

  path.write_text(text, encoding="utf-8")
  print(f"Patched {path}")


def main() -> int:
  if not LLAMA_CPP_PY.is_file():
    print(f"Missing {LLAMA_CPP_PY}", file=sys.stderr)
    return 1
  patch_llama_cpp_py(LLAMA_CPP_PY)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
