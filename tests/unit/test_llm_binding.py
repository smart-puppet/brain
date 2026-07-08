from __future__ import annotations

from pathlib import Path

import pytest

from puppet.llm.binding import (
  binding_lock_path,
  read_installed_binding,
  validate_llama_binding,
)


def test_read_installed_binding_missing(tmp_path: Path) -> None:
  assert read_installed_binding(tmp_path) is None


def test_read_installed_binding_upstream(tmp_path: Path) -> None:
  lock = tmp_path / "vendor/native/llama-binding.lock"
  lock.parent.mkdir(parents=True)
  lock.write_text("LLAMA_BINDING=upstream\n", encoding="utf-8")
  assert read_installed_binding(tmp_path) == "upstream"


def test_read_installed_binding_prism(tmp_path: Path) -> None:
  lock = tmp_path / "vendor/native/llama-binding.lock"
  lock.parent.mkdir(parents=True)
  lock.write_text("# comment\nLLAMA_BINDING=prism\n", encoding="utf-8")
  assert read_installed_binding(tmp_path) == "prism"


def test_read_installed_binding_ignores_unknown(tmp_path: Path) -> None:
  lock = tmp_path / "vendor/native/llama-binding.lock"
  lock.parent.mkdir(parents=True)
  lock.write_text("LLAMA_BINDING=other\n", encoding="utf-8")
  assert read_installed_binding(tmp_path) is None


def test_validate_llama_binding_unknown_config(tmp_path: Path) -> None:
  with pytest.raises(ValueError, match="llm.binding must be one of"):
    validate_llama_binding({"llm": {"binding": "bogus"}}, repo_root=tmp_path)


def test_validate_llama_binding_no_lock_warns(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
  validate_llama_binding({"llm": {"binding": "upstream"}}, repo_root=tmp_path)
  assert "No llama binding stamp" in caplog.text


def test_validate_llama_binding_match(tmp_path: Path) -> None:
  lock = binding_lock_path(tmp_path)
  lock.parent.mkdir(parents=True)
  lock.write_text("LLAMA_BINDING=upstream\n", encoding="utf-8")
  validate_llama_binding({"llm": {"binding": "upstream"}}, repo_root=tmp_path)


def test_validate_llama_binding_mismatch(tmp_path: Path) -> None:
  lock = binding_lock_path(tmp_path)
  lock.parent.mkdir(parents=True)
  lock.write_text("LLAMA_BINDING=prism\n", encoding="utf-8")
  with pytest.raises(RuntimeError, match="llm.binding is 'upstream'"):
    validate_llama_binding({"llm": {"binding": "upstream"}}, repo_root=tmp_path)
