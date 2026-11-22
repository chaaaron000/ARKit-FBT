"""녹화 파일(.afbtrec) → MP4 영상.

왼쪽: iPhone 카메라 미리보기 + ARKit 뼈대 (원본)
오른쪽: VRChat이 실제로 받는 자세 (머리 기준 정면/위에서 본 모습) + 상태

    uv run python -m arkit_fbt_relay.recording export-video recordings/xxx.afbtrec
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np

from .pipeline import TRACKERS, FrameOut, Processor, Settings, foot_lock_status
from .preview import Preview, Reassembler, display_rotation, is_preview_chunk
from .protocol import BONES, JOINT_NAMES, ProtocolError, parse_frame
from .recording import read_records

W, H = 1280, 720
CAM_W = 720  # 왼쪽 카메라 영역 폭
TRACKER_COLORS = {
    "hip": "#ff9f1c", "left_foot": "#2ec4b6", "right_foot": "#e71d36",
    "left_knee": "#7bdff2", "right_knee": "#f7a1a1", "chest": "#c77dff",
    "left_elbow": "#b5e48c", "right_elbow": "#ffd166",
}
GATE_LABELS = {j: label for _i, _n, j, label in TRACKERS}
GATE_LABELS.update({"head_joint": "머리", "neck_1_joint": "목", "spine_7_joint": "가슴",
                    "left_upLeg_joint": "왼허벅지", "right_upLeg_joint": "오른허벅지"})


def _ensure_qt():
    import os
    if os.name == "nt" and "QT_QPA_FONTDIR" not in os.environ:
        # 화면 없는(offscreen) 모드에서도 한글 글꼴을 찾도록
        os.environ["QT_QPA_FONTDIR"] = os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts")
    from PySide6 import QtGui
    if QtGui.QGuiApplication.instance() is None:
        _ensure_qt.app = QtGui.QGuiApplication(["arkit-fbt-video"])  # 글꼴 렌더링에 필요
    return QtGui


def export_video(path: Path, out: Path | None = None, fps: float = 15.0, settings: Settings | None = None,
                 progress: Callable[[float], None] | None = None) -> Path:
    """녹화를 MP4로 만든다. 반환: 만든 파일 경로."""
    import imageio_ffmpeg

    QtGui = _ensure_qt()
    from PySide6 import QtCore

    out = out or path.with_suffix(".mp4")
    records = list(read_records(path))
    if not records:
        raise ValueError("빈 녹화 파일입니다")
    settings = settings or Settings(enabled_trackers=[n for _i, n, _j, _l in TRACKERS[:5]])
    proc = Processor(settings)
    reasm = Reassembler()

    t0, t_end = records[0][0], records[-1][0]
    n_out = max(1, int((t_end - t0) * fps))
    writer = imageio_ffmpeg.write_frames(str(out), (W, H), fps=fps, codec="libx264",
                                         pix_fmt_in="rgb24", pix_fmt_out="yuv420p", quality=7,
                                         macro_block_size=16)
    writer.send(None)

    idx = 0
    preview: Preview | None = None
    state: FrameOut | None = None
    raw_tracked = 0
    cam_rot = None
    has_video = False
    try:
        for k in range(n_out):
            now = t0 + k / fps
            while idx < len(records) and records[idx][0] <= now:
                _t, data = records[idx]
                idx += 1
                if is_preview_chunk(data):
                    p = reasm.add(data)
                    if p is not None:
                        preview, has_video = p, True
                    continue
                try:
                    frame = parse_frame(data)
                except ProtocolError:
                    continue
                state = proc.process(frame)
                raw_tracked = len(frame.tracked_joints) if frame.tracked_joints is not None else len(frame.joints)
                if state.camera_rot is not None:
                    cam_rot = state.camera_rot
            img = _render(QtGui, QtCore, now - t0, preview, state, raw_tracked, cam_rot, proc)
            img = img.convertToFormat(QtGui.QImage.Format.Format_RGB888)
            writer.send(bytes(img.constBits())[: W * H * 3])
            if progress and k % 30 == 0:
                progress(k / n_out)
    finally:
        writer.close()
    if progress:
        progress(1.0)
    if not has_video:
        print("참고: 이 녹화에는 카메라 미리보기가 없어 뼈대만 그렸습니다.")
    return out


def _render(QtGui, QtCore, t: float, preview: Preview | None, state: FrameOut | None,
            raw_tracked: int, cam_rot, proc: Processor):
    img = QtGui.QImage(W, H, QtGui.QImage.Format.Format_RGB32)
    img.fill(QtGui.QColor("#15161b"))
    p = QtGui.QPainter(img)
    p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)

    # ---- 왼쪽: 카메라 + 원본 뼈대
    if preview is not None:
        cam = QtGui.QImage.fromData(preview.jpeg, "JPG")
        if not cam.isNull():
            cam = cam.convertToFormat(QtGui.QImage.Format.Format_RGB32)
            _draw_raw_skeleton(QtGui, QtCore, cam, preview.joints)
            rot = display_rotation(cam_rot) if cam_rot is not None else 0
            if rot:
                cam = cam.transformed(QtGui.QTransform().rotate(-90 * rot))
            cam = cam.scaled(CAM_W, H, QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                             QtCore.Qt.TransformationMode.SmoothTransformation)
            p.drawImage((CAM_W - cam.width()) // 2, (H - cam.height()) // 2, cam)
    else:
        p.setPen(QtGui.QColor("#777"))
        p.drawText(QtCore.QRectF(0, 0, CAM_W, H), QtCore.Qt.AlignmentFlag.AlignCenter, "카메라 영상 없음")

    # ---- 오른쪽: VRChat이 받는 자세
    x0 = CAM_W + 20
    p.setPen(QtGui.QColor("#e6e6e6"))
    font = p.font()
    font.setPointSize(13)
    font.setBold(True)
    p.setFont(font)
    p.drawText(x0, 34, f"{int(t // 60):02d}:{t % 60:05.2f}")
    font.setPointSize(11)
    font.setBold(False)
    p.setFont(font)

    lines = [f"ARKit 추적 관절: {raw_tracked}/21"]
    if state is not None and state.tracked:
        gated = [GATE_LABELS.get(g, g) for g in sorted(proc.gated)]
        lines.append("고정 중: " + (", ".join(gated) if gated else "없음"))
        lines.append("발 고정: " + foot_lock_status(proc.settings, proc.foot_locked, proc.floor_y is not None))
        lines.append(f"자세 기울기 보정: {proc.lean_deg:.1f}°" if proc.lean_deg is not None else "자세 기울기 보정: -")
        lines.append(f"카메라 정면: {state.facing:+.2f}")
    else:
        lines.append("몸 인식 안 됨")
    for i, line in enumerate(lines):
        color = "#c8c8c8"
        if line.startswith("고정 중") and not line.endswith("없음"):
            color = "#ffd166"
        elif line.startswith("발 고정") and proc.foot_locked:
            color = "#2ec4b6"
        p.setPen(QtGui.QColor(color))
        p.drawText(x0, 64 + i * 24, line)

    if state is not None and state.tracked and state.head_pos is not None:
        _draw_output(QtGui, QtCore, p, state, x0, 64 + len(lines) * 24, proc.foot_locked)
    p.end()
    return img


def _draw_raw_skeleton(QtGui, QtCore, img, joints) -> None:
    if not joints:
        return
    w, h = img.width(), img.height()
    idx = {n: i for i, n in enumerate(JOINT_NAMES)}

    def pt(name):
        i = idx[name]
        v = joints[i] if i < len(joints) else None
        return None if v is None else QtCore.QPointF(v[0] * w, v[1] * h)

    p = QtGui.QPainter(img)
    p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
    p.setPen(QtGui.QPen(QtGui.QColor("#d0d6e0"), 2))
    for a, b in BONES:
        pa, pb = pt(a), pt(b)
        if pa is not None and pb is not None:
            p.drawLine(pa, pb)
    p.end()


def _draw_output(QtGui, QtCore, p, state: FrameOut, x0: int, y0: int, locked=frozenset()) -> None:
    """머리를 원점으로, VRChat이 받는 트래커 위치를 정면(X-Y)과 위(X-Z)에서 그린다. 고정된 발은 굵은 테두리."""
    head = np.asarray(state.head_pos)
    fwd = state.forward if state.forward is not None else np.array([0.0, 0.0, 1.0])
    # 몸 정면 기준 좌표로 돌려서 그린다 (몸이 돌아도 그림은 정면)
    f = np.array([fwd[0], 0.0, fwd[2]])
    f /= max(1e-6, float(np.linalg.norm(f)))
    right = np.array([f[2], 0.0, -f[0]])
    scale = 170.0  # px / m
    panel_w = 250

    def local(pos):
        d = np.asarray(pos) - head
        return float(d @ right), float(d[1]), float(d @ f)

    for col, title in enumerate(("정면 (VRChat이 받는 값)", "위에서 (정면 = 위쪽)")):
        px = x0 + col * (panel_w + 20)
        p.setPen(QtGui.QColor("#555"))
        p.setBrush(QtCore.Qt.BrushStyle.NoBrush)
        p.drawRect(QtCore.QRectF(px, y0, panel_w, 520))
        p.setPen(QtGui.QColor("#aaa"))
        p.drawText(px + 6, y0 - 6, title)
        cx = px + panel_w / 2
        cy = y0 + 60 if col == 0 else y0 + 260
        # 머리
        p.setPen(QtCore.Qt.PenStyle.NoPen)
        p.setBrush(QtGui.QColor("white"))
        p.drawEllipse(QtCore.QPointF(cx, cy), 8, 8)
        for t in state.trackers:
            lx, ly, lz = local(t.pos)
            pt = QtCore.QPointF(cx + lx * scale, cy - ly * scale) if col == 0 else \
                QtCore.QPointF(cx + lx * scale, cy - lz * scale)
            p.setBrush(QtGui.QColor(TRACKER_COLORS.get(t.name, "#ccc")))
            locked_foot = t.name in locked
            p.setPen(QtGui.QPen(QtGui.QColor("#b8ff3c" if locked_foot else "white"), 3 if locked_foot else 1))
            p.drawEllipse(pt, 9, 9)
        # 1m 눈금
        p.setPen(QtGui.QColor("#444"))
        p.drawLine(QtCore.QPointF(px + 10, y0 + 505), QtCore.QPointF(px + 10 + scale * 0.5, y0 + 505))
        p.drawText(QtCore.QPointF(px + 14 + scale * 0.5, y0 + 509), "50cm")
