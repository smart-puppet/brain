from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum, auto
from typing import Any


class PipelineState(Enum):
  IDLE = auto()
  LISTENING = auto()
  THINKING = auto()
  SPEAKING = auto()
  INTERRUPT_CAPTURE = auto()


@dataclass(frozen=True)
class AudioChunk:
  """Mono PCM audio block."""

  samples: Any  # numpy.ndarray float32
  sample_rate: int
  timestamp: float = 0.0


@dataclass
class TranscriptSegment:
  text: str
  is_final: bool = False
  end_of_utterance: bool = False


@dataclass
class LlmMessage:
  role: str
  content: str


@dataclass
class TurnSnapshot:
  """Restorable conversation state for noise-only interrupts."""

  messages: list[LlmMessage]
  draft_user: str


@dataclass
class Conversation:
  messages: list[LlmMessage] = field(default_factory=list)
  draft_user: str = ""

  def add_user(self, text: str) -> None:
    self.messages.append(LlmMessage(role="user", content=text))

  def add_assistant(self, text: str) -> None:
    self.messages.append(LlmMessage(role="assistant", content=text))

  def append_draft(self, text: str) -> None:
    self.draft_user += text

  def commit_draft(self) -> str:
    text = self.draft_user.strip()
    if text:
      self.add_user(text)
      self.draft_user = ""
    return text

  def prompt_messages(self) -> list[LlmMessage]:
    msgs = list(self.messages)
    draft = self.draft_user.strip()
    if draft:
      msgs.append(LlmMessage(role="user", content=draft))
    return msgs

  def snapshot(self) -> TurnSnapshot:
    return TurnSnapshot(
      messages=[replace(m) for m in self.messages],
      draft_user=self.draft_user,
    )

  def restore(self, snap: TurnSnapshot) -> None:
    self.messages = [replace(m) for m in snap.messages]
    self.draft_user = snap.draft_user
