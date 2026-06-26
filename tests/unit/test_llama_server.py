from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock

from puppet.core.types import Conversation
from puppet.llm.llama_server import LlamaServerLlm


def _bare_llm() -> LlamaServerLlm:
  llm = LlamaServerLlm.__new__(LlamaServerLlm)
  llm._system_prompt = "You are Elmo."
  llm._last_perf = None
  llm._prefill_lock = threading.Lock()
  llm._prefill_cancel = threading.Event()
  llm._prefill_thread = None
  llm._prefilled_draft = None
  llm._prefill_inflight_draft = None
  llm._prefill_at_generation = True
  return llm


def test_llama_server_builds_messages() -> None:
  llm = _bare_llm()
  conversation = Conversation()
  conversation.add_user("hello")
  messages = llm._build_messages(conversation)
  assert messages[0] == {"role": "system", "content": "You are Elmo."}
  assert messages[1] == {"role": "user", "content": "hello"}


def test_llama_server_warmup_uses_streaming_path() -> None:
  llm = _bare_llm()
  llm._chat_once = MagicMock(return_value=iter(["Hi"]))
  llm.warmup(max_tokens=4, prompt="Hi", stream=True)
  llm._chat_once.assert_called_once()
  assert llm._chat_once.call_args.kwargs["stream"] is True
  assert llm._last_perf is None


def test_prefill_uses_max_tokens_zero() -> None:
  llm = _bare_llm()
  llm._chat_once = MagicMock(return_value={})
  conversation = Conversation()
  conversation.draft_user = "hello world"
  llm.ensure_prefill(conversation)
  llm._chat_once.assert_called_once()
  assert llm._chat_once.call_args.kwargs["max_tokens"] == 0
  assert llm._chat_once.call_args.kwargs["stream"] is False
  assert llm.prefilled_draft == "hello world"


def test_schedule_prefill_skips_when_already_warm() -> None:
  llm = _bare_llm()
  llm._prefilled_draft = "hello"
  llm._chat_once = MagicMock()
  conversation = Conversation()
  conversation.draft_user = "hello"
  llm.schedule_prefill(conversation)
  llm._chat_once.assert_not_called()


def test_stream_reply_ensures_prefill_before_generation() -> None:
  llm = _bare_llm()
  llm.ensure_prefill = MagicMock()
  llm._chat_once = MagicMock(return_value=iter(["Hi"]))
  conversation = Conversation()
  conversation.draft_user = "hello"
  tokens = list(llm.stream_reply(conversation))
  llm.ensure_prefill.assert_called_once_with(conversation)
  assert tokens == ["Hi"]
