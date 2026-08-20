import random

from puppet.play.actions import (
  looks_like_private_state,
  parse_robot_actions,
  strip_private_robot_state,
  strip_robot_actions,
  user_asked_to_stop,
)
from puppet.play.phrases import play_phrase
from puppet.play.policy import PlayConfig, PlayMemory, plan_follow, plan_seek


def test_strips_leaked_body_status() -> None:
  spoken, actions = parse_robot_actions(
    "Life is an excruciatingly long and pointless sequence of biological processes. "
    "It mostly involves suffering, eventually.\n"
    "BodyStatus: standing still. Motion=on."
  )
  assert actions == []
  assert "BodyStatus" not in spoken
  assert "Motion=" not in spoken
  assert "suffering" in spoken
  assert strip_private_robot_state("I am talking now. BodyStatus: standing still. Motion=on.") == (
    "I am talking now."
  )
  assert looks_like_private_state("BodyStatus:")
  assert looks_like_private_state("Motion=on.")
  assert looks_like_private_state("I am already standing still.")
  assert not looks_like_private_state("I will follow you.")
  assert strip_private_robot_state("I am already standing still.") == ""
  assert parse_robot_actions("I am already standing still.") == ("", [])


def test_llm_follow_tag() -> None:
  spoken, actions = parse_robot_actions("Okay! I will follow you.\n<<follow>>")
  assert actions == ["follow"]
  assert "Okay" in spoken
  assert "<<" not in spoken


def test_llm_back_tag() -> None:
  spoken, actions = parse_robot_actions("Okay, I will roll backward a little.\n<<backward>>")
  assert actions == ["back"]
  assert "backward" in spoken.lower()
  assert "<<" not in spoken
  spoken, actions = parse_robot_actions("Okay, I will reverse a little.\n<<back>>")
  assert actions == ["back"]


def test_llm_forward_tag() -> None:
  spoken, actions = parse_robot_actions("Okay, I will roll forward a little. <<forward>>")
  assert actions == ["forward"]
  assert "forward" in spoken.lower()
  assert "<<" not in spoken


def test_infers_nudge_from_canned_speech_without_tag() -> None:
  spoken, actions = parse_robot_actions("D'accord, je recule un peu.")
  assert actions == ["back"]
  assert "recule" in spoken
  spoken, actions = parse_robot_actions("D'accord, j'avance un peu.")
  assert actions == ["forward"]
  spoken, actions = parse_robot_actions("Okay, I will roll forward a little.")
  assert actions == ["forward"]
  spoken, actions = parse_robot_actions("Okay, I will roll backward a little.")
  assert actions == ["back"]
  spoken, actions = parse_robot_actions("Coucou ! Je vais super bien, merci.")
  assert actions == []


def test_infers_seek_from_spoken_hide_and_seek_without_tag() -> None:
  spoken, actions = parse_robot_actions(
    "Oui, j'adore jouer à cache-cache avec Maignon ! C'est très amusant quand on cherche."
  )
  assert actions == ["seek"]
  assert "cache-cache" in spoken
  spoken, actions = parse_robot_actions("D'accord ! Je te cherche.")
  assert actions == ["seek"]
  spoken, actions = parse_robot_actions("Okay! I will look for you.")
  assert actions == ["seek"]
  spoken, actions = parse_robot_actions("Okay! Ich suche dich.")
  assert actions == ["seek"]
  spoken, actions = parse_robot_actions("Let's play hide-and-seek later maybe.")
  assert actions == ["seek"]
  spoken, actions = parse_robot_actions("C'est très amusant quand on cherche.")
  assert actions == []


def test_infers_idle_from_canned_stop_speech_without_tag() -> None:
  spoken, actions = parse_robot_actions("D'accord, je reste ici.")
  assert actions == ["idle"]
  assert "reste ici" in spoken
  spoken, actions = parse_robot_actions("Okay, I will stay here.")
  assert actions == ["idle"]
  spoken, actions = parse_robot_actions("Okay, ich bleibe hier.")
  assert actions == ["idle"]


def test_looks_like_vision_question() -> None:
  from puppet.mqtt.scene import (
    looks_like_vision_followup,
    looks_like_vision_question,
    needs_vision_capture,
  )

  assert looks_like_vision_question("que vois-tu?")
  assert looks_like_vision_question("What do you see?")
  assert looks_like_vision_question("Was siehst du?")
  assert looks_like_vision_question("Est-ce que tu vois maintenant?")
  assert not looks_like_vision_question("Follow me")
  assert looks_like_vision_followup("Et maintenant.")
  assert looks_like_vision_followup("and now?")
  assert looks_like_vision_followup("Jetzt")
  assert not looks_like_vision_followup("Qu'est-ce que tu veux faire maintenant ?")
  assert needs_vision_capture("What do you see right now")
  assert needs_vision_capture("But do you see anything rightnow?")
  assert needs_vision_capture("And what do you see right now")
  assert not needs_vision_capture("What can you tell me about life?")
  assert not needs_vision_capture("Follow me")


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
  assert user_asked_to_stop("tu peux arrêter maintenant, s'il te plaît.")
  assert user_asked_to_stop("Je veux que tu t'arrêtes.")
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
  cfg = PlayConfig(follow_stop_m=1.5)
  scene = {
    "closest_m": 1.4,
    "sectors": {"left": 1.6, "center": 1.4, "right": 1.6},
    "objects": [{"label": "person", "dist_m": 1.4, "bearing": "center"}],
  }
  nudge = plan_follow(scene, PlayMemory(), cfg)
  assert nudge.cmd == "idle"
  assert nudge.reason == "close"


def test_follow_stops_when_legs_lose_person_box() -> None:
  mem = PlayMemory()
  cfg = PlayConfig(follow_stop_m=1.5)
  plan_follow(
    {
      "closest_m": 1.4,
      "sectors": {"left": 1.6, "center": 1.4, "right": 1.6},
      "objects": [{"label": "person", "dist_m": 1.4, "bearing": "center"}],
    },
    mem,
    cfg,
  )
  scene = {
    "closest_m": 1.2,
    "sectors": {"left": 1.4, "center": 1.2, "right": 1.4},
    "objects": [],
  }
  nudge = plan_follow(scene, mem, cfg)
  assert nudge.cmd == "idle"
  assert nudge.reason == "close"


def test_follow_remembers_exit_side_after_centering() -> None:
  cfg = PlayConfig(lost_ticks_max=2, turn_ms_per_deg=10, follow_recover_deg=180, search_turn_dur_ms=400)
  mem = PlayMemory()
  left = {
    "closest_m": 2.0,
    "sectors": {"left": 2.0, "center": 2.0, "right": 2.0},
    "objects": [{"label": "person", "dist_m": 2.0, "bearing": "left"}],
  }
  center = {
    "closest_m": 1.8,
    "sectors": {"left": 1.8, "center": 1.8, "right": 1.8},
    "objects": [{"label": "person", "dist_m": 1.8, "bearing": "center"}],
  }
  assert plan_follow(left, mem, cfg).cmd == "turn_left"
  assert plan_follow(center, mem, cfg).cmd == "forward"
  empty = {
    "closest_m": 2.5,
    "sectors": {"left": 2.5, "center": 2.5, "right": 2.5},
    "objects": [],
  }
  nudge = plan_follow(empty, mem, cfg)
  assert nudge.cmd == "turn_left"
  assert nudge.reason == "recover"


def test_follow_turns_off_frame_even_when_something_is_close() -> None:
  cfg = PlayConfig(follow_stop_m=1.5, lost_ticks_max=2)
  mem = PlayMemory()
  plan_follow(
    {
      "closest_m": 1.6,
      "sectors": {"left": 1.6, "center": 1.6, "right": 1.6},
      "objects": [{"label": "person", "dist_m": 1.6, "bearing": "left"}],
    },
    mem,
    cfg,
  )
  nudge = plan_follow(
    {
      "closest_m": 1.2,
      "sectors": {"left": 1.2, "center": 1.4, "right": 1.4},
      "objects": [],
    },
    mem,
    cfg,
  )
  assert nudge.reason == "recover"
  assert nudge.cmd == "turn_left"


def test_follow_avoids_closer_obstacle() -> None:
  cfg = PlayConfig(obstacle_m=0.5, person_margin_m=0.2)
  scene = {
    "closest_m": 0.4,
    "sectors": {"left": 1.5, "center": 0.4, "right": 0.6},
    "objects": [{"label": "person", "dist_m": 2.0, "bearing": "center"}],
  }
  mem = PlayMemory()
  nudge = plan_follow(scene, mem, cfg)
  assert nudge.reason != "unstick"
  assert nudge.cmd == "forward"
  second = plan_follow(scene, mem, cfg)
  assert second.reason != "unstick"


def test_stuck_signal_is_ignored() -> None:
  scene = {
    "closest_m": 0.3,
    "sectors": {"left": 0.4, "center": 0.3, "right": 0.4},
    "objects": [],
  }
  nudge = plan_seek(scene, PlayMemory(), PlayConfig())
  assert nudge.reason != "unstick"
  assert nudge.cmd in ("turn_left", "turn_right", "forward", "backward")


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
  assert seek.reason == "scan"


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
  assert "personne" in play_phrase("nofollow", "fr").lower()
  assert "anyone" in play_phrase("nofollow", "en").lower()
  assert "ten" in play_phrase("seek", "en").lower()
  assert "dix" in play_phrase("seek", "fr").lower()
  assert "zehn" in play_phrase("seek", "de").lower()
  assert "stay" in play_phrase("idle", "en").lower()
  assert "backward" in play_phrase("back", "en").lower()
  assert play_phrase("ack", "en")
  assert "accord" in play_phrase("ack", "fr").lower()


def test_seek_found_when_person_within_two_meters() -> None:
  scene = {
    "closest_m": 1.8,
    "sectors": {"left": 2.0, "center": 1.8, "right": 2.0},
    "objects": [{"label": "person", "dist_m": 1.9, "bearing": "center"}],
  }
  nudge = plan_seek(scene, PlayMemory(), PlayConfig(found_m=2.0))
  assert nudge.reason == "found"
  assert nudge.cmd == "idle"


def test_seek_station_look_then_faces_and_rolls() -> None:
  scene = {
    "closest_m": 2.0,
    "sectors": {"left": 2.0, "center": 2.0, "right": 2.0},
    "objects": [],
  }
  mem = PlayMemory()
  cfg = PlayConfig(
    turn_ms_per_deg=10,
    follow_spin_deg=90,
    seek_face_deg=90,
    search_turn_dur_ms=300,
    search_forward_dur_ms=900,
    search_forward_ticks=1,
    seek_giveup_ticks=40,
    seek_map=False,
  )
  nudges = [plan_seek(scene, mem, cfg) for _ in range(7)]
  assert [n.reason for n in nudges] == ["scan", "scan", "scan", "search", "search", "search", "search"]
  assert [n.cmd for n in nudges[:6]] == ["turn_left"] * 6
  assert nudges[6].cmd == "forward"
  assert nudges[6].dur_ms == 900


def test_seek_keeps_rolling_after_first_look() -> None:
  scene = {
    "closest_m": 2.5,
    "sectors": {"left": 2.5, "center": 2.5, "right": 2.5},
    "objects": [],
  }
  mem = PlayMemory()
  cfg = PlayConfig(
    turn_ms_per_deg=10,
    follow_spin_deg=90,
    seek_face_deg=0,
    search_turn_dur_ms=300,
    search_forward_dur_ms=800,
    search_forward_ticks=8,
    seek_giveup_ticks=40,
    seek_map=False,
  )
  nudges = [plan_seek(scene, mem, cfg) for _ in range(11)]
  assert all(n.reason == "scan" for n in nudges[:3])
  assert all(n.cmd == "forward" for n in nudges[3:])


def test_seek_second_station_keeps_going_straight() -> None:
  scene = {
    "closest_m": 2.0,
    "sectors": {"left": 2.0, "center": 2.0, "right": 2.0},
    "objects": [],
  }
  mem = PlayMemory()
  cfg = PlayConfig(
    turn_ms_per_deg=10,
    follow_spin_deg=90,
    seek_face_deg=90,
    search_turn_dur_ms=300,
    search_forward_dur_ms=400,
    search_forward_ticks=1,
    seek_giveup_ticks=40,
    seek_map=False,
  )
  cmds = [plan_seek(scene, mem, cfg).cmd for _ in range(10)]
  assert cmds[6] == "forward"
  assert all(cmd == "forward" for cmd in cmds[6:])


def test_seek_recovers_toward_exit_then_scans() -> None:
  cfg = PlayConfig(
    found_m=1.5,
    turn_ms_per_deg=10,
    follow_recover_deg=90,
    follow_spin_deg=90,
    seek_face_deg=90,
    search_turn_dur_ms=300,
    seek_giveup_ticks=40,
    seek_map=False,
  )
  mem = PlayMemory()
  seen = {
    "closest_m": 2.5,
    "sectors": {"left": 2.5, "center": 2.5, "right": 2.5},
    "objects": [{"label": "person", "dist_m": 2.5, "bearing": "right"}],
  }
  empty = {
    "closest_m": 2.5,
    "sectors": {"left": 2.5, "center": 2.5, "right": 2.5},
    "objects": [],
  }
  assert plan_seek(seen, mem, cfg).reason == "turn_to_person"
  recover = plan_seek(empty, mem, cfg)
  assert recover.cmd == "turn_right"
  assert recover.reason == "recover"
  reasons = [recover.reason]
  while reasons[-1] == "recover":
    reasons.append(plan_seek(empty, mem, cfg).reason)
    assert len(reasons) < 10
  assert reasons[-1] == "search"


def test_seek_turns_instead_of_rolling_into_furniture() -> None:
  open_scene = {
    "closest_m": 2.0,
    "sectors": {"left": 2.0, "center": 2.0, "right": 2.0},
    "objects": [],
  }
  blocked = {
    "closest_m": 0.4,
    "sectors": {"left": 1.5, "center": 0.4, "right": 0.6},
    "objects": [],
  }
  mem = PlayMemory()
  cfg = PlayConfig(
    turn_ms_per_deg=10,
    follow_spin_deg=90,
    seek_face_deg=90,
    search_turn_dur_ms=300,
    search_forward_dur_ms=400,
    search_forward_ticks=2,
    seek_giveup_ticks=40,
    seek_map=False,
  )
  nudge = None
  for _ in range(12):
    nudge = plan_seek(open_scene, mem, cfg)
    if nudge.cmd == "forward":
      break
  assert nudge is not None and nudge.cmd == "forward"
  abort = plan_seek(blocked, mem, cfg)
  assert abort.cmd == "backward"
  assert abort.reason != "unstick"


def test_seek_turns_when_path_blocked() -> None:
  scene = {
    "closest_m": 0.4,
    "sectors": {"left": 1.5, "center": 0.4, "right": 0.6},
    "objects": [],
  }
  nudge = plan_seek(scene, PlayMemory(), PlayConfig())
  assert nudge.reason != "unstick"
  assert nudge.cmd in ("turn_left", "turn_right", "forward", "backward")


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


def test_follow_start_spins_then_nofollow() -> None:
  scene = {
    "closest_m": 2.5,
    "sectors": {"left": 2.5, "center": 2.5, "right": 2.5},
    "objects": [],
  }
  mem = PlayMemory()
  cfg = PlayConfig(
    lost_ticks_max=2,
    turn_ms_per_deg=10,
    follow_spin_deg=90,
    search_turn_dur_ms=300,
  )
  first = plan_follow(scene, mem, cfg)
  assert first.cmd == "turn_left"
  assert first.reason == "scan"
  cmds = [first.cmd]
  reasons = [first.reason]
  while reasons[-1] != "nofollow":
    nudge = plan_follow(scene, mem, cfg)
    cmds.append(nudge.cmd)
    reasons.append(nudge.reason)
    assert len(cmds) < 10
  assert all(cmd == "turn_left" for cmd in cmds[:-1])
  assert cmds[-1] == "idle"
  assert mem.spun_ms >= 90 * 10


def test_follow_lost_turns_toward_exit_side_then_full_spin() -> None:
  cfg = PlayConfig(
    lost_ticks_max=1,
    turn_ms_per_deg=10,
    follow_recover_deg=90,
    follow_spin_deg=90,
    search_turn_dur_ms=300,
  )
  mem = PlayMemory()
  seen = {
    "closest_m": 2.0,
    "sectors": {"left": 2.0, "center": 2.0, "right": 2.0},
    "objects": [{"label": "person", "dist_m": 2.0, "bearing": "right"}],
  }
  assert plan_follow(seen, mem, cfg).reason == "turn_to_person"
  empty = {
    "closest_m": 2.5,
    "sectors": {"left": 2.5, "center": 2.5, "right": 2.5},
    "objects": [],
  }
  recover = plan_follow(empty, mem, cfg)
  assert recover.cmd == "turn_right"
  assert recover.reason == "recover"
  reasons = [recover.reason]
  while reasons[-1] == "recover":
    reasons.append(plan_follow(empty, mem, cfg).reason)
    assert len(reasons) < 10
  assert "scan" in reasons
  while reasons[-1] != "nofollow":
    reasons.append(plan_follow(empty, mem, cfg).reason)
    assert len(reasons) < 20
  assert reasons[-1] == "nofollow"


def test_follow_stops_recover_when_person_reappears() -> None:
  cfg = PlayConfig(turn_ms_per_deg=10, follow_recover_deg=180, search_turn_dur_ms=400)
  mem = PlayMemory()
  seen = {
    "closest_m": 2.0,
    "sectors": {"left": 2.0, "center": 2.0, "right": 2.0},
    "objects": [{"label": "person", "dist_m": 2.0, "bearing": "left"}],
  }
  empty = {
    "closest_m": 2.5,
    "sectors": {"left": 2.5, "center": 2.5, "right": 2.5},
    "objects": [],
  }
  assert plan_follow(seen, mem, cfg).reason == "turn_to_person"
  assert plan_follow(empty, mem, cfg).reason == "recover"
  back = plan_follow(seen, mem, cfg)
  assert back.reason == "turn_to_person"
  assert mem.search_phase == ""


def test_close_furniture_still_spins_at_follow_start() -> None:
  scene = {
    "closest_m": 0.58,
    "sectors": {"left": 4.0, "center": 4.0, "right": 4.0},
    "floor_ahead_pct": 0.37,
    "objects": [],
  }
  nudge = plan_follow(scene, PlayMemory(), PlayConfig(lost_ticks_max=0))
  assert nudge.reason == "scan"
  assert nudge.cmd in ("turn_left", "turn_right")


def test_blocked_path_spins_when_follow_starts_empty() -> None:
  blocked = {
    "closest_m": 0.65,
    "sectors": {"left": 0.95, "center": 0.65, "right": 0.9},
    "floor_ahead_pct": 0.05,
    "objects": [],
  }
  mem = PlayMemory()
  cfg = PlayConfig(lost_ticks_max=0)
  nudges = [plan_follow(blocked, mem, cfg) for _ in range(4)]
  assert all(n.reason == "scan" for n in nudges)
  assert all(n.cmd in ("turn_left", "turn_right") for n in nudges)


def test_low_floor_ahead_does_not_unstick() -> None:
  scene = {
    "closest_m": 2.0,
    "sectors": {"left": 1.8, "center": 1.5, "right": 0.6},
    "floor_ahead_pct": 0.05,
    "objects": [],
  }
  nudge = plan_seek(scene, PlayMemory(), PlayConfig(floor_block_pct=0.12))
  assert nudge.reason != "unstick"
  assert nudge.cmd in ("turn_left", "turn_right", "forward", "backward")


def test_alive_jitter_still_searches() -> None:
  scene = {
    "closest_m": 2.0,
    "sectors": {"left": 2.0, "center": 2.0, "right": 2.0},
    "objects": [],
  }
  mem = PlayMemory(rng=random.Random(7))
  cfg = PlayConfig(
    alive_jitter=0.4,
    search_turn_ticks=3,
    search_forward_ticks=3,
    seek_map=False,
  )
  cmds = [plan_seek(scene, mem, cfg).cmd for _ in range(14)]
  assert set(cmds) <= {"turn_left", "turn_right", "forward"}
  assert "forward" in cmds
  assert any(cmd.startswith("turn_") for cmd in cmds)
