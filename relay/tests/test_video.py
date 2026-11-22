"""녹화(관절 + 카메라 미리보기) → MP4 내보내기."""

import os

import numpy as np

from arkit_fbt_relay.fake_phone import synth_joints
from arkit_fbt_relay.preview import build_chunks
from arkit_fbt_relay.protocol import JOINT_NAMES, Pose, build_frame
from arkit_fbt_relay.recording import Recorder, analyze

CAM = Pose(np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 0.0, 1.0]))


def _fake_jpeg() -> bytes:
    from PySide6 import QtCore, QtGui
    img = QtGui.QImage(384, 288, QtGui.QImage.Format.Format_RGB32)
    img.fill(QtGui.QColor("#334455"))
    buf = QtCore.QBuffer()
    buf.open(QtCore.QIODevice.OpenModeFlag.WriteOnly)
    img.save(buf, "JPG", 60)
    return bytes(buf.data())


def test_export_video(tmp_path, monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", os.environ.get("QT_QPA_PLATFORM", "offscreen"))
    from arkit_fbt_relay.video import _ensure_qt, export_video
    _ensure_qt()
    jpeg = _fake_jpeg()

    path = tmp_path / "r.afbtrec"
    rec = Recorder(path)
    for i in range(120):  # 2초, 60fps
        t = i / 60
        rec.write(100 + t, build_frame(i + 1, t, CAM, synth_joints(t)))
        if i % 8 == 0:
            uv = [(0.5, 0.2 + k * 0.03) for k in range(len(JOINT_NAMES))]
            for c in build_chunks(i // 8 + 1, 384, 288, t, uv, jpeg):
                rec.write(100 + t, c)
    rec.close()

    # 분석은 미리보기 조각을 무시해야 한다
    assert "깨짐 0" in analyze(path)

    out = export_video(path, tmp_path / "r.mp4", fps=10)
    assert out.exists() and out.stat().st_size > 1000
