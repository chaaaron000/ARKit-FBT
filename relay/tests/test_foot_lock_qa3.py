"""Independent round-three regressions; assert output, not tuning constants."""
import os
import subprocess
import sys

import numpy as np
import pytest

from arkit_fbt_relay.mathutil import Rigid
from arkit_fbt_relay.pipeline import Processor, Settings
from arkit_fbt_relay.protocol import Pose, build_frame, parse_frame
from arkit_fbt_relay.recording import Recorder
from test_foot_lock_qa import CAMERA, LEFT, MODES, REST, RIGHT, feed, feet, packet, planted


def world_frame(i, *, shift=0.0, travel=0.0, hover=0.0, drop=0.0):
    drift = Rigid(CAMERA.rot, np.array([shift, 0., 0.]))
    joints = {}
    for name, p in REST.items():
        foot = name in (LEFT, RIGHT, 'left_toes_joint', 'right_toes_joint')
        dy = hover if foot else -drop
        joints[name] = Pose(drift.apply(p.pos + (0.0 if foot else travel, dy, 0.)), p.rot)
    cam = Pose(drift.apply(CAMERA.pos), CAMERA.rot)
    return parse_frame(build_frame(i + 1, i / 60, cam, joints))


@pytest.mark.parametrize('foot_lock', [False, True])
def test_common_tracking_origin_shift_does_not_jump_feet(foot_lock):
    p = Processor(Settings(foot_lock=foot_lock))
    before = planted(p)
    after = p.process(world_frame(120, shift=0.08))
    # Camera and subject stayed fixed relative to each other; output must stay in the same reference.
    assert np.linalg.norm(feet(after) - feet(before)) < 0.01
    assert 'camera_moved' not in after.events


@MODES
@pytest.mark.parametrize('change', ['switch_mode', 'disable'])
def test_settings_change_does_not_step_both_feet_during_weight_shift(mode, change):
    control, changed = [Processor(Settings(foot_lock_mode=mode)) for _ in range(2)]
    for p in (control, changed):
        planted(p)
        for j in range(43):
            feed(p, 120 + j, travel=0.20 * j / 42)
    if change == 'disable':
        changed.settings.foot_lock = False
    else:
        changed.settings.foot_lock_mode = 'motion' if mode == 'stable' else 'stable'
    expected, actual = [feed(p, 163, travel=0.20) for p in (control, changed)]
    for side in ('left', 'right'):
        switched = feet(actual, side) - actual.head_pos
        unchanged = feet(expected, side) - expected.head_pos
        assert np.linalg.norm(switched - unchanged) < 0.01


def test_seated_hover_cannot_replace_floor_measured_before_tracking_gap():
    p = Processor(Settings())
    for i in range(90):
        feed(p, i)
    floor = p.floor_y
    if floor is None:
        pytest.fail('Standing input failed to establish a floor')
    for i in (90, 720, 1380):
        feed(p, i, masked=True)
    for i in range(1410, 1515):
        p.process(world_frame(i, hover=0.12, drop=0.30))
    assert p.floor_y <= floor + 0.031


@MODES
def test_seated_feet_remain_unplanted_after_floor_history_expires(mode):
    p = Processor(Settings(foot_lock_mode=mode))
    for i in range(240):
        feed(p, i)
    for j in range(24 * 60):
        fraction = min(1., j / 24)
        p.process(world_frame(240 + j, hover=fraction * 0.065, drop=fraction * 0.30))
        if j == 90 and p.foot_locked:
            pytest.fail('Seated feet did not release before the old floor samples expired')
    assert not p.foot_locked


def _offscreen(script, *args):
    # A separate process avoids inheriting an existing QGuiApplication from the video test.
    return subprocess.run([sys.executable, '-c', script, *map(str, args)], check=True,
                          capture_output=True, encoding='utf-8', timeout=25,
                          env={**os.environ, 'QT_QPA_PLATFORM': 'offscreen', 'PYTHONUTF8': '1'}).stdout


@pytest.mark.skipif(sys.platform != 'win32', reason='Reproduced with the Windows Qt layout')
def test_foot_mode_combo_fits_relay_default_viewport():
    clipped = _offscreen("""
import os
from pathlib import Path
os.environ.setdefault('QT_QPA_FONTDIR', str(Path(os.environ['WINDIR']) / 'Fonts'))
from PySide6 import QtCore, QtWidgets
from arkit_fbt_relay.engine import Engine
from arkit_fbt_relay.gui import MainWindow
from arkit_fbt_relay.pipeline import Settings
app = QtWidgets.QApplication([])
win = MainWindow(Engine(Settings(), listen_port=43861, osc_port=43862, dry_run=True), [], lambda: None)
win.show()
app.processEvents()
scroll = win.findChild(QtWidgets.QScrollArea)
combo = next(c for c in win.findChildren(QtWidgets.QComboBox) if c.findData('stable') >= 0)
scroll.verticalScrollBar().setValue(scroll.verticalScrollBar().maximum())
app.processEvents()
viewport = scroll.viewport()
reset = next(b for b in win.findChildren(QtWidgets.QPushButton) if b.text() == '카메라 기준 재설정')
print(max(c.mapTo(viewport, QtCore.QPoint()).x() + c.width() - viewport.width()
          for c in (combo, reset, win.export_button)))
win.close()
""")
    assert int(clipped) <= 0


def test_gui_export_matches_stable_foot_position(tmp_path):
    path = tmp_path / 'stable-drag.afbtrec'
    rec = Recorder(path)
    live = Processor(Settings(foot_lock_mode='stable'))
    for i in range(250):
        dx = 0.14 * min(1., max(0, i - 135) / 35)
        data = packet(i, left=(dx, 0., 0.))
        rec.write(i / 60, data)
        live_pose = live.process(parse_frame(data))
    rec.close()
    shown = _offscreen("""
import os, subprocess, sys, time
from pathlib import Path
if os.name == 'nt':
    os.environ.setdefault('QT_QPA_FONTDIR', str(Path(os.environ['WINDIR']) / 'Fonts'))
from PySide6 import QtWidgets
from arkit_fbt_relay import video
from arkit_fbt_relay.engine import Engine
from arkit_fbt_relay.gui import MainWindow
from arkit_fbt_relay.pipeline import Settings

app = QtWidgets.QApplication([])
path = Path(sys.argv[1])
QtWidgets.QFileDialog.getOpenFileName = lambda *a: (str(path), '')
spawn = subprocess.Popen
subprocess.Popen = lambda argv, *a, **kw: None if argv[0] == 'explorer' else spawn(argv, *a, **kw)
coords = []
draw = video._draw_output
def capture(gui, core, painter, state, x0, y0, locked):
    foot = next(t.pos for t in state.trackers if t.name == 'left_foot')
    coords.append(float((foot - state.head_pos)[0]))
    return draw(gui, core, painter, state, x0, y0, locked)
video._draw_output = capture
engine = Engine(Settings(foot_lock_mode='stable'), listen_port=43861, osc_port=43862, dry_run=True)
win = MainWindow(engine, [], lambda: None, path.parent)
win.export_button.click()
deadline = time.monotonic() + 15
while not win.export_button.isEnabled():
    if time.monotonic() > deadline:
        raise RuntimeError('Export did not finish')
    app.processEvents()
    time.sleep(.01)
if not path.with_suffix('.mp4').exists() or not coords:
    raise RuntimeError(engine.snapshot().logs)
print(coords[-1])
win.close()
""", path)
    exported = float(shown.strip().splitlines()[-1])
    assert exported == pytest.approx(float((feet(live_pose) - live_pose.head_pos)[0]), abs=0.02)


def test_tracking_origin_drift_during_torso_shift_does_not_move_camera_reference():
    p = Processor(Settings(foot_lock=False))
    planted(p)
    events = []
    for j in range(60):
        t = j / 60
        out = p.process(world_frame(120 + j, shift=0.16 * t, travel=-0.14 * t))
        events.extend(out.events)
    assert 'camera_moved' not in events


@MODES
def test_missing_foot_does_not_confirm_a_lower_plant_height(mode):
    p = Processor(Settings(foot_lock_mode=mode))
    for i in range(240):
        before = feed(p, i)
    for j in range(142):
        visible = j < 60
        y = -0.044 * min(1., j / 15) if visible else 0.
        frame = parse_frame(packet(240 + j, left=(0., y, 0.)))
        if not visible:
            frame = parse_frame(build_frame(frame.seq, frame.timestamp, frame.camera, frame.joints,
                                            untracked={LEFT}))
        out = p.process(frame)
    assert feet(out)[1] == pytest.approx(feet(before)[1], abs=0.01)
    assert 'left_foot' in p.foot_locked
