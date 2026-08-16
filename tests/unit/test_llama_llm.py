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
  from puppet.llm.llama import _VISION_INSTRUCTIONS

  llm = LlamaLlm.__new__(LlamaLlm)
  llm._system_prompt = "You are Kace."
  llm._vision_hint_fn = None
  llm._llm = MagicMock()
  llm.warmup(max_tokens=1, prompt="Hi", stream=False)
  llm._llm.create_chat_completion.assert_called_once_with(
    messages=[
      {"role": "system", "content": f"You are Kace.\n\n{_VISION_INSTRUCTIONS}"},
      {"role": "user", "content": "Hi"},
    ],
    stream=False,
    temperature=0.0,
    max_tokens=1,
  )


def test_build_messages_keeps_system_frozen_and_appends_camera_to_user() -> None:
  from puppet.core.types import Conversation
  from puppet.llm.llama import _VISION_INSTRUCTIONS

  llm = LlamaLlm.__new__(LlamaLlm)
  llm._system_prompt = "You are Kace."
  llm._vision_hint_fn = lambda: "BodyStatus: standing still.\nCameraJSON: {\"objects\":[]}"
  conversation = Conversation()
  conversation.add_user("Hello")
  conversation.add_assistant("Hi!")
  conversation.append_draft("What do you see?")
  messages = llm._build_messages(conversation)
  assert messages[0] == {
    "role": "system",
    "content": f"You are Kace.\n\n{_VISION_INSTRUCTIONS}",
  }
  assert messages[1] == {"role": "user", "content": "Hello"}
  assert messages[2] == {"role": "assistant", "content": "Hi!"}
  assert messages[3]["role"] == "user"
  assert messages[3]["content"].startswith("What do you see?")
  assert "Private: BodyStatus" in messages[3]["content"]
  assert '{"objects":[]}' in messages[3]["content"]
  assert '{"objects":[]}' not in messages[0]["content"]
