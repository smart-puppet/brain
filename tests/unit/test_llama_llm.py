from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from puppet.llm.llama import LlamaLlm, _resolve_ggml_type


def test_resolve_ggml_type_q4_0() -> None:
  pytest.importorskip("llama_cpp")
  assert _resolve_ggml_type("q4_0") == 2
  assert _resolve_ggml_type(2) == 2


def test_resolve_ggml_type_unknown() -> None:
  pytest.importorskip("llama_cpp")
  with pytest.raises(ValueError, match="Unknown GGML cache type"):
    _resolve_ggml_type("not_a_real_type")


def test_llm_warmup_runs_minimal_completion() -> None:
  llm = LlamaLlm.__new__(LlamaLlm)
  llm._system_prompt = "You are Elmo."
  llm._llm = MagicMock()
  llm.warmup(max_tokens=1, prompt="Hi", stream=False)
  llm._llm.create_chat_completion.assert_called_once_with(
    messages=[
      {"role": "system", "content": "You are Elmo."},
      {"role": "user", "content": "Hi"},
    ],
    stream=False,
    temperature=0.0,
    max_tokens=1,
  )
