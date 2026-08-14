from puppet.play.actions import parse_robot_actions, strip_robot_actions, user_asked_to_stop
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
  from puppet.mqtt.scene import looks_like_vision_question

  assert looks_like_vision_question("que vois-tu?")
  assert looks_like_vision_question("What do you see?")
  assert looks_like_vision_question("Was siehst du?")
  assert not looks_like_vision_question("Follow me")


def test_glimpse_not_forced_on_reverse() -> None:
  from puppet.mqtt.scene import should_force_object_glimpse

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
  nudge = plan_follow(scene, PlayMemory(), cfg)
  assert nudge.cmd == "turn_left"
  assert nudge.reason == "avoid"


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


def test_seek_found_when_close() -> None:
  scene = {
    "closest_m": 1.0,
    "sectors": {"left": 1.2, "center": 1.0, "right": 1.2},
    "objects": [{"label": "person", "dist_m": 1.0, "bearing": "center"}],
  }
  nudge = plan_seek(scene, PlayMemory(), PlayConfig(found_m=1.15))
  assert nudge.reason == "found"
  assert nudge.cmd == "idle"


def test_seek_searches_when_no_person() -> None:
  scene = {"closest_m": 2.0, "sectors": {}, "objects": []}
  nudge = plan_seek(scene, PlayMemory(), PlayConfig())
  assert nudge.reason == "search"
  assert nudge.cmd in ("turn_left", "turn_right")


def test_seek_does_not_roll_toward_voice() -> None:
  scene = {
    "closest_m": 1.5,
    "sectors": {"left": 1.5, "center": 1.5, "right": 1.5},
    "objects": [],
  }
  nudge = plan_seek(scene, PlayMemory(), PlayConfig(), heading_error_deg=5)
  assert nudge.cmd != "forward"
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


def test_follow_lost_waits_instead_of_hunting() -> None:
  scene = {
    "closest_m": 1.5,
    "sectors": {"left": 1.5, "center": 1.5, "right": 1.5},
    "objects": [],
  }
  mem = PlayMemory()
  cfg = PlayConfig()
  with_heading = plan_follow(scene, mem, cfg, heading_error_deg=90)
  assert with_heading.cmd == "idle"
  assert with_heading.reason == "lost"
  facing = plan_follow(scene, PlayMemory(), cfg, heading_error_deg=5)
  assert facing.cmd == "idle"
  assert facing.reason == "lost"
