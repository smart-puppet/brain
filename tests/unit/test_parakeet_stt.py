from __future__ import annotations

import pytest

from puppet.stt.parakeet import (
  STREAMING_CHUNK_MS_TO_ATT_CONTEXT,
  att_context_for_chunk_ms,
  strip_lang_tags,
)


@pytest.mark.parametrize(
  ("chunk_ms", "att_context"),
  sorted(STREAMING_CHUNK_MS_TO_ATT_CONTEXT.items()),
)
def test_att_context_for_chunk_ms(chunk_ms: int, att_context: tuple[int, int]) -> None:
  assert att_context_for_chunk_ms(chunk_ms) == att_context


def test_att_context_for_chunk_ms_rejects_unknown() -> None:
  with pytest.raises(ValueError, match="chunk_ms=99"):
    att_context_for_chunk_ms(99)


def test_strip_lang_tags() -> None:
  text = "Hello world. <en-US> How are you?"
  assert strip_lang_tags(text) == "Hello world. How are you?"
  assert strip_lang_tags(" my partial") == " my partial"
