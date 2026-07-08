"""Component smoke tests (require models + hardware; run via scripts/ or pytest -m component)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"


@pytest.mark.component
def test_stt_script_help() -> None:
  proc = subprocess.run(
    [sys.executable, str(SCRIPTS / "test_stt.py"), "--help"],
    cwd=ROOT,
    capture_output=True,
    text=True,
    check=False,
  )
  assert proc.returncode == 0
  assert "Test STT only" in proc.stdout


@pytest.mark.component
def test_llm_script_help() -> None:
  proc = subprocess.run(
    [sys.executable, str(SCRIPTS / "test_llm.py"), "--help"],
    cwd=ROOT,
    capture_output=True,
    text=True,
    check=False,
  )
  assert proc.returncode == 0
  assert "Test LLM only" in proc.stdout


@pytest.mark.component
def test_tts_script_help() -> None:
  proc = subprocess.run(
    [sys.executable, str(SCRIPTS / "test_tts.py"), "--help"],
    cwd=ROOT,
    capture_output=True,
    text=True,
    check=False,
  )
  assert proc.returncode == 0
  assert "Test TTS only" in proc.stdout
