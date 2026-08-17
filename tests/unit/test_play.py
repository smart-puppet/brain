import random

from puppet.play.actions import (
  parse_robot_actions,
  strip_robot_actions,
  user_asked_to_stop,
)
from puppet.play.phrases import play_phrase
from puppet.play.policy import PlayConfig, PlayMemory, plan_follow, plan_seek


def test_llm_follow_tag() -> None:
  spoken, actions = parse_robot_actions("Okay! I will follow you.\n<<follow>>")
  assert actions == ["follow"]
  assert "Okay" in spoken
  assert "<<" not in spoken


def test_llm_back_tag() -> None:
  spoken, actions = parse_robot_actions("Okay, I will reverse a little.\n<<back>>")
  assert actions == ["back"]
  assert "reverse" in spoken.lower()
  assert "<<" not in spoken


def test_looks_like_vision_question() -> None:
  from puppet.mqtt.scene import looks_like_vision_followup, looks_like_vision_question

  assert looks_like_vision_question("que vois-tu?")
  assert looks_like_vision_question("What do you see?")
  assert looks_like_vision_question("Was siehst du?")
  assert looks_like_vision_question("Est-ce que tu vois maintenant?")
  assert not looks_like_vision_question("Follow me")
  assert looks_like_vision_followup("Et maintenant.")
  assert looks_like_vision_followup("and now?")
  assert looks_like_vision_followup("Jetzt")
  assert not looks_like_vision_followup("Qu'est-ce que tu veux faire maintenant ?")


def test_glimpse_not_forced_on_reverse() -> None:
  from puppet.mqtt.scene import looks_like_looking_bridge, should_force_object_glimpse

  assert should_force_object_glimpse(
    looked=False,
    inject_context=True,
    has_objects=True,
    mentions_objects=False,
    vision_dump=False,
    suppressed_phrases=False,
    vision_question=False,
    motion=True,
  ) is False
  assert should_force_object_glimpse(
    looked=False,
    inject_context=True,
    has_objects=True,
    mentions_objects=False,
    vision_dump=False,
    suppressed_phrases=False,
    vision_question=True,
    motion=False,
  ) is True
  assert should_force_object_glimpse(
    looked=True,
    inject_context=True,
    has_objects=True,
    mentions_objects=False,
    vision_dump=False,
    suppressed_phrases=False,
    vision_question=True,
    motion=False,
  ) is False
  assert looks_like_looking_bridge("Je regarde...")
  assert looks_like_looking_bridge("I am looking.")
  assert not looks_like_looking_bridge("Je vois une chaise.")
  assert not looks_like_looking_bridge("I see a chair.")


def test_camera_json_only_after_look() -> None:
  import json

  from puppet.mqtt.scene import SceneIngest

  ingest = SceneIngest(name="t")
  ingest.set_inject_context(True)
  ingest.apply_scene(
    {"objects": [{"label": "chair", "bearing": "left", "dist_m": 1.0}]},
    from_look=False,
  )
  assert ingest.context_line() == ""
  ingest.apply_scene(
    {"objects": [{"label": "chair", "bearing": "left", "dist_m": 1.0}]},
    from_look=True,
  )
  line = ingest.context_line()
  assert "chair" in line
  assert "need_look" not in line

  vision = SceneIngest(name="vision")
  vision.set_inject_context(True)
  vision.apply_scene(
    {"objects": [{"label": "chair", "bearing": "left", "dist_m": 1.0}], "req_id": "look1"},
    from_look=True,
  )
  class _Msg:
    def __init__(self, payload: bytes) -> None:
      self.payload = payload

  vision._on_message(
    None,
    None,
    _Msg(json.dumps({"objects": [{"label": "plant"}], "req_id": "nav"}).encode()),
  )
  assert vision.from_look()
  assert "chair" in vision.context_line()


def test_llm_look_and_stop_tags() -> None:
  spoken, actions = parse_robot_actions("I am looking!\n<<look>>\nACTION: stop")
  assert actions == ["look", "idle"]
  assert "looking" in spoken.lower()
  assert "ACTION" not in spoken
  assert "<<" not in spoken


def test_user_asked_to_stop() -> None:
  assert user_asked_to_stop("Stop, stop, stop.")
  assert user_asked_to_stop("Stop.")
  assert user_asked_to_stop("Bitte bleib stehen")
  assert user_asked_to_stop("Arrête !")
  assert not user_asked_to_stop("Follow me")
  assert not user_asked_to_stop("What is a stopwatch?")


def test_strip_robot_actions_leaves_chat() -> None:
  assert strip_robot_actions("Want to hear a story?") == "Want to hear a story?"
  assert strip_robot_actions("<<follow>>") == ""


def test_follow_turns_toward_person() -> None:
  cfg = PlayConfig()
  mem = PlayMemory()
  scene = {
    "closest_m": 2.0,
    "sectors": {"left": 2.0, "center": 2.0, "right": 2.0},
    "objects": [{"label": "person", "dist_m": 2.0, "bearing": "left"}],
  }
  nudge = plan_follow(scene, mem, cfg)
  assert nudge.cmd == "turn_left"
  assert nudge.reason == "turn_to_person"


def test_follow_stops_when_close() -> None:
  cfg = PlayConfig(follow_stop_m=0.9)
  scene = {
    "closest_m": 0.8,
    "sectors": {"left": 1.0, "center": 0.8, "right": 1.0},
    "objects": [{"label": "person", "dist_m": 0.8, "bearing": "center"}],
  }
  nudge = plan_follow(scene, PlayMemory(), cfg)
  assert nudge.cmd == "idle"
  assert nudge.reason == "close"


def test_follow_avoids_closer_obstacle() -> None:
  cfg = PlayConfig(obstacle_m=0.5, person_margin_m=0.2)
  scene = {
    "closest_m": 0.4,
    "sectors": {"left": 1.5, "center": 0.4, "right": 0.6},
    "objects": [{"label": "person", "dist_m": 2.0, "bearing": "center"}],
  }
  mem = PlayMemory()
  nudge = plan_follow(scene, mem, cfg)
  assert nudge.cmd == "turn_left"
  assert nudge.reason == "avoid"
  second = plan_follow(scene, mem, cfg)
  assert second.cmd == "backward"
  assert second.reason == "unstick"


def test_stuck_with_no_side_path_reverses() -> None:
  scene = {
    "closest_m": 0.3,
    "sectors": {"left": 0.4, "center": 0.3, "right": 0.4},
    "objects": [],
  }
  nudge = plan_seek(scene, PlayMemory(), PlayConfig())
  assert nudge.cmd == "backward"
  assert nudge.reason == "unstick"


def test_follow_and_seek_use_separate_turn_speeds() -> None:
  follow_scene = {
    "closest_m": 2.0,
    "sectors": {"left": 2.0, "center": 2.0, "right": 2.0},
    "objects": [{"label": "person", "dist_m": 2.0, "bearing": "left"}],
  }
  seek_scene = {
    "closest_m": 2.0,
    "sectors": {"left": 2.0, "center": 2.0, "right": 2.0},
    "objects": [],
  }
  cfg = PlayConfig(turn_speed=180, seek_turn_speed=70)
  follow = plan_follow(follow_scene, PlayMemory(), cfg)
  seek = plan_seek(seek_scene, PlayMemory(), cfg)
  assert follow.speed == 180
  assert seek.speed == 70
  assert seek.reason == "search"


def test_follow_approaches_when_clear() -> None:
  scene = {
    "closest_m": 2.0,
    "sectors": {"left": 2.0, "center": 2.0, "right": 2.0},
    "objects": [{"label": "person", "dist_m": 2.0, "bearing": "center"}],
  }
  nudge = plan_follow(scene, PlayMemory(), PlayConfig())
  assert nudge.cmd == "forward"
  assert nudge.reason == "approach"


def test_play_found_and_giveup_phrases() -> None:
  assert "Found" in play_phrase("found", "en")
  assert "trouve" in play_phrase("found", "fr").lower()
  assert "finde" in play_phrase("giveup", "de").lower()
  assert "ten" in play_phrase("seek", "en").lower()
  assert "dix" in play_phrase("seek", "fr").lower()
  assert "zehn" in play_phrase("seek", "de").lower()
  assert "stay" in play_phrase("idle", "en").lower()
  assert "reverse" in play_phrase("back", "en").lower()
  assert play_phrase("ack", "en")
  assert "accord" in play_phrase("ack", "fr").lower()


def test_seek_found_when_close() -> None:
  scene = {
    "closest_m": 1.0,
    "sectors": {"left": 1.2, "center": 1.0, "right": 1.2},
    "objects": [{"label": "person", "dist_m": 1.0, "bearing": "center"}],
  }
  nudge = plan_seek(scene, PlayMemory(), PlayConfig(found_m=1.15))
  assert nudge.reason == "found"
  assert nudge.cmd == "idle"


def test_seek_alternates_glance_and_roll() -> None:
  scene = {
    "closest_m": 2.0,
    "sectors": {"left": 2.0, "center": 2.0, "right": 2.0},
    "objects": [],
  }
  mem = PlayMemory()
  cfg = PlayConfig(
    search_turn_ticks=1,
    search_forward_ticks=1,
    search_turn_dur_ms=500,
    search_forward_dur_ms=900,
  )
  nudges = [plan_seek(scene, mem, cfg) for _ in range(4)]
  assert [n.cmd for n in nudges] == ["turn_left", "forward", "turn_left", "forward"]
  assert nudges[0].dur_ms == 500
  assert nudges[1].dur_ms == 900


def test_seek_searches_when_no_person() -> None:
  scene = {
    "closest_m": 2.0,
    "sectors": {"left": 2.0, "center": 2.0, "right": 2.0},
    "objects": [],
  }
  mem = PlayMemory()
  cfg = PlayConfig(
    search_turn_ticks=3,
    search_forward_ticks=3,
    search_turn_dur_ms=1800,
    search_forward_dur_ms=1200,
  )
  nudges = [plan_seek(scene, mem, cfg) for _ in range(6)]
  assert [n.cmd for n in nudges[:3]] == ["turn_left", "turn_left", "turn_left"]
  assert all(n.dur_ms == 1800 for n in nudges[:3])
  assert [n.cmd for n in nudges[3:]] == ["forward", "forward", "forward"]
  assert all(n.dur_ms == 1200 for n in nudges[3:])


def test_seek_turns_when_path_blocked() -> None:
  scene = {
    "closest_m": 0.4,
    "sectors": {"left": 1.5, "center": 0.4, "right": 0.6},
    "objects": [],
  }
  nudge = plan_seek(scene, PlayMemory(), PlayConfig())
  assert nudge.cmd in ("turn_left", "turn_right")
  assert nudge.reason == "search"


def test_seek_does_not_chase_voice() -> None:
  scene = {
    "closest_m": 1.5,
    "sectors": {"left": 1.5, "center": 1.5, "right": 1.5},
    "objects": [],
  }
  nudge = plan_seek(scene, PlayMemory(), PlayConfig(), heading_error_deg=90)
  assert nudge.reason != "turn_to_voice"
  assert nudge.reason != "approach_voice"


def test_seek_gives_up_when_lost_too_long() -> None:
  scene = {"closest_m": 0.7, "sectors": {}, "objects": []}
  mem = PlayMemory()
  cfg = PlayConfig(seek_giveup_ticks=3)
  plan_seek(scene, mem, cfg)
  plan_seek(scene, mem, cfg)
  nudge = plan_seek(scene, mem, cfg)
  assert nudge.cmd == "idle"
  assert nudge.reason == "giveup"


def test_follow_lost_glances_then_rolls() -> None:
  scene = {
    "closest_m": 1.5,
    "sectors": {"left": 1.5, "center": 1.5, "right": 1.5},
    "objects": [],
  }
  mem = PlayMemory()
  cfg = PlayConfig(lost_ticks_max=2, search_turn_ticks=1, search_forward_ticks=1)
  first = plan_follow(scene, mem, cfg, heading_error_deg=90)
  assert first.cmd == "idle"
  assert first.reason == "lost"
  plan_follow(scene, mem, cfg, heading_error_deg=90)
  cmds = [plan_follow(scene, mem, cfg, heading_error_deg=90).cmd for _ in range(4)]
  assert cmds == ["turn_left", "forward", "turn_left", "forward"]


def test_close_furniture_does_not_roll() -> None:
  scene = {
    "closest_m": 0.58,
    "sectors": {"left": 4.0, "center": 4.0, "right": 4.0},
    "floor_ahead_pct": 0.37,
    "objects": [],
  }
  nudge = plan_follow(scene, PlayMemory(), PlayConfig(lost_ticks_max=0))
  assert nudge.cmd != "forward"
  assert nudge.reason in ("unstick", "scan", "avoid")


def test_corner_does_not_roll_into_the_same_wall() -> None:
  blocked = {
    "closest_m": 0.65,
    "sectors": {"left": 0.95, "center": 0.65, "right": 0.9},
    "floor_ahead_pct": 0.05,
    "objects": [],
  }
  barely = {
    "closest_m": 0.90,
    "sectors": {"left": 1.1, "center": 0.9, "right": 1.05},
    "floor_ahead_pct": 0.13,
    "objects": [],
  }
  mem = PlayMemory()
  cfg = PlayConfig(lost_ticks_max=0, uturn_after=3, uturn_ticks=2, uturn_dur_ms=900)
  for _ in range(6):
    plan_follow(blocked, mem, cfg)
  assert mem.retreats >= 1
  nudge = plan_follow(barely, mem, cfg)
  assert nudge.cmd != "forward"


def test_uturn_after_repeated_retreats() -> None:
  blocked = {
    "closest_m": 0.65,
    "sectors": {"left": 0.95, "center": 0.65, "right": 1.2},
    "floor_ahead_pct": 0.04,
    "objects": [],
  }
  mem = PlayMemory()
  cfg = PlayConfig(lost_ticks_max=0, uturn_after=3, uturn_ticks=2, uturn_dur_ms=900)
  nudges = [plan_follow(blocked, mem, cfg) for _ in range(8)]
  uturns = [n for n in nudges if n.reason == "uturn"]
  assert len(uturns) >= 2
  assert uturns[0].cmd == uturns[1].cmd
  assert uturns[0].cmd in ("turn_left", "turn_right")
  assert uturns[0].dur_ms == 900
  open_side = {
    "closest_m": 2.0,
    "sectors": {"left": 2.0, "center": 2.0, "right": 2.0},
    "floor_ahead_pct": 0.5,
    "objects": [],
  }
  escaped = plan_follow(open_side, mem, cfg)
  assert escaped.cmd == "forward"
  assert escaped.reason == "scan"


def test_unstick_insists_on_one_direction() -> None:
  blocked = {
    "closest_m": 0.65,
    "sectors": {"left": 1.2, "center": 0.65, "right": 0.9},
    "floor_ahead_pct": 0.04,
    "objects": [],
  }
  mem = PlayMemory()
  cfg = PlayConfig(lost_ticks_max=0, uturn_after=3, uturn_ticks=2, uturn_dur_ms=900)
  turns = [
    n.cmd
    for n in (plan_follow(blocked, mem, cfg) for _ in range(10))
    if n.cmd.startswith("turn_")
  ]
  assert turns
  assert set(turns) == {turns[0]}


def test_low_floor_ahead_counts_as_blocked() -> None:
  scene = {
    "closest_m": 2.0,
    "sectors": {"left": 1.8, "center": 1.5, "right": 0.6},
    "floor_ahead_pct": 0.05,
    "objects": [],
  }
  nudge = plan_seek(scene, PlayMemory(), PlayConfig(floor_block_pct=0.12))
  assert nudge.reason == "search"
  assert nudge.cmd == "turn_left"


def test_alive_jitter_still_searches() -> None:
  scene = {
    "closest_m": 2.0,
    "sectors": {"left": 2.0, "center": 2.0, "right": 2.0},
    "objects": [],
  }
  mem = PlayMemory(rng=random.Random(7))
  cfg = PlayConfig(alive_jitter=0.4, search_turn_ticks=3, search_forward_ticks=3)
  cmds = [plan_seek(scene, mem, cfg).cmd for _ in range(14)]
  assert set(cmds) <= {"turn_left", "turn_right", "forward"}
  assert "forward" in cmds
  assert any(cmd.startswith("turn_") for cmd in cmds)
