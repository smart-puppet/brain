from unittest.mock import MagicMock

from puppet.hardware.mouth import PhonemeMouth
from puppet.hardware.pca9685 import PCA9685


def test_pca_sleep_wake_mode1() -> None:
  bus = MagicMock()
  # MODE1 reads: awake with AI, then after sleep bit set, then awake again.
  bus.read_byte_data.side_effect = [0x01, 0x11, 0x01, 0x01]
  pca = PCA9685(bus, address=0x40)
  pca._enabled = True

  pca.sleep()
  assert bus.write_byte_data.call_args_list[-1].args[1:] == (0x00, 0x11)
  assert pca._enabled is False

  pca.wake()
  # clear sleep, then RESTART|AI
  writes = [c.args for c in bus.write_byte_data.call_args_list[-2:]]
  assert writes[0][1:] == (0x00, 0x01)
  assert writes[1][1:] == (0x00, 0x01 | 0x80 | 0x08)
  assert pca._enabled is True


def test_pca_set_enabled_toggles_sleep() -> None:
  bus = MagicMock()
  bus.read_byte_data.return_value = 0x01
  pca = PCA9685(bus, address=0x40)
  pca.set_enabled(True)
  assert pca._enabled is True
  pca.set_enabled(False)
  assert pca._enabled is False
  pca.set_enabled(False)  # idempotent
  assert bus.write_byte_data.call_count >= 2


def test_mouth_wakes_pca_while_talking() -> None:
  pca = MagicMock()
  mouth = PhonemeMouth(pca, channel=15, closed_deg=0.0, open_deg=25.0, mode="word")
  pca.reset_mock()

  mouth.on_reply_sync_start()
  pca.set_enabled.assert_called_once_with(True)

  pca.reset_mock()
  mouth.close_for_listen()
  assert pca.set_enabled.call_args_list == [
    ((True,),),
    ((False,),),
  ]
  pca.set_servo_angle.assert_called_with(15, 0.0)


def test_mouth_open_at_start_sleeps_after_pose() -> None:
  pca = MagicMock()
  mouth = PhonemeMouth(pca, channel=15, closed_deg=0.0, open_deg=25.0, mode="word")
  pca.reset_mock()
  mouth.open_at_start()
  assert pca.set_enabled.call_args_list == [
    ((True,),),
    ((False,),),
  ]
  pca.set_servo_angle.assert_called_once_with(15, 25.0)


def test_mouth_close_sleeps_pca() -> None:
  pca = MagicMock()
  mouth = PhonemeMouth(pca, channel=15, closed_deg=10.0, open_deg=25.0, mode="word")
  pca.reset_mock()
  mouth.close()
  assert pca.set_enabled.call_args_list == [
    ((True,),),
    ((False,),),
  ]
  pca.set_servo_angle.assert_called_with(15, 10.0)
