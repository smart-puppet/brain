from pathlib import Path

from puppet.play.speeds import (
  clamp_speed,
  normalize_speeds,
  read_play_speeds,
  resolve_play_speeds,
  write_play_speeds,
)


def test_clamp_speed() -> None:
  assert clamp_speed(100, 125) == 100
  assert clamp_speed(0, 125) == 20
  assert clamp_speed(400, 125) == 200
  assert clamp_speed("nope", 125) == 125


def test_normalize_keeps_defaults_for_missing_keys() -> None:
  speeds = normalize_speeds(
    {"follow_turn": 90},
    {"follow_turn": 125, "seek_turn": 140, "forward": 80},
  )
  assert speeds == {"follow_turn": 90, "seek_turn": 140, "forward": 80}


def test_write_and_read_roundtrip(tmp_path: Path) -> None:
  path, written = write_play_speeds(
    tmp_path, {"follow_turn": 80, "seek_turn": 150, "forward": 60}
  )
  assert path.name == "play.speeds"
  assert read_play_speeds(tmp_path) == written


def test_resolve_uses_yaml_until_overlay_exists(tmp_path: Path) -> None:
  follow = {"turn_speed": 110, "seek_turn_speed": 160, "forward_speed": 95}
  assert resolve_play_speeds(follow, tmp_path) == {
    "follow_turn": 110,
    "seek_turn": 160,
    "forward": 95,
  }
  write_play_speeds(tmp_path, {"follow_turn": 40, "seek_turn": 50, "forward": 60})
  assert resolve_play_speeds(follow, tmp_path) == {
    "follow_turn": 40,
    "seek_turn": 50,
    "forward": 60,
  }
