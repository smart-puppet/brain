#!/usr/bin/env python3
"""Test LLM only: type text, stream the reply, print llama.cpp perf counters."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
  sys.path.insert(0, str(_SCRIPTS))

from _runtime import configure_logging, load_puppet_config
from puppet.core.types import Conversation
from puppet.llm import create_llm
from puppet.llm.llama import LlamaLlm
from puppet.llm.perf import format_llama_perf, format_llama_perf_cli


def main(argv: list[str] | None = None) -> int:
  parser = argparse.ArgumentParser(description="Test LLM only (text → llama.cpp)")
  parser.add_argument("--config", default="config", help="Config directory")
  parser.add_argument("--language", "-l", choices=["en", "fr", "de"], default=None)
  parser.add_argument("text", nargs="*", help="User message (omit for interactive mode)")
  args = parser.parse_args(argv)

  config = load_puppet_config(args.config, language=args.language)
  configure_logging(config, trace=True)

  llm = create_llm(config)
  print("LLM ready.", flush=True)

  one_shot = bool(args.text)
  user_text = " ".join(args.text).strip()
  conversation = Conversation()

  def run_turn(prompt: str) -> int:
    conversation.add_user(prompt)
    print("Assistant: ", end="", flush=True)
    reply_parts: list[str] = []
    try:
      for token in llm.stream_reply(conversation):
        reply_parts.append(token)
        print(token, end="", flush=True)
    except RuntimeError as exc:
      print(f"\nLLM error: {exc}", file=sys.stderr)
      return 1
    print()
    conversation.add_assistant("".join(reply_parts))

    perf = getattr(llm, "last_perf", None)
    if perf is not None:
      print(f"[ {format_llama_perf_cli(perf)} ]")
      print(f"perf: {format_llama_perf(perf)}")
    return 0

  if one_shot:
    if not user_text:
      print("No input.", file=sys.stderr)
      return 1
    return run_turn(user_text)

  print("Multi-turn LLM test. Empty line or 'quit' to exit.")
  while True:
    try:
      user_text = input("You: ").strip()
    except EOFError:
      print()
      break
    if not user_text or user_text.lower() in {"quit", "exit", "q"}:
      break
    if run_turn(user_text) != 0:
      return 1
    print()
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
