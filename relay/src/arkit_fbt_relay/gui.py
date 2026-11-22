"""상태 확인과 실시간 튜닝용 창 (PySide6 + pyqtgraph)."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from .engine import Engine
from .pipeline import TRACKERS
from .preview import display_rotation, framing_warnings
from .protocol import BONES, JOINT_NAMES

TRACKER_COLORS = {
    "hip": "#ff9f1c", "left_foot": "#2ec4b6", "right_foot": "#e71d36",
    "left_knee": "#7bdff2", "right_knee": "#f7a1a1", "chest": "#c77dff",
    "left_elbow": "#b5e48c", "right_elbow": "#ffd166",
}


class FloatSlider(QtWidgets.QWidget):
    valueChanged = QtCore.Signal(float)

    def __init__(self, lo: float, hi: float, value: float, decimals: int = 2, steps: int = 200):
        super().__init__()
        self.lo, self.hi, self.steps, self.decimals = lo, hi, steps, decimals
        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider.setRange(0, steps)
        self.label = QtWidgets.QLabel()
        self.label.setMinimumWidth(40)
        lay.addWidget(self.slider, 1)
        lay.addWidget(self.label)
        self.slider.valueChanged.connect(self._changed)
        self.set_value(value)

    def set_value(self, v: float) -> None:
        i = round((v - self.lo) / (self.hi - self.lo) * self.steps)
        self.slider.setValue(max(0, min(self.steps, i)))
        self.label.setText(f"{v:.{self.decimals}f}")

    def value(self) -> float:
        return self.lo + (self.hi - self.lo) * self.slider.value() / self.steps

    def _changed(self, _i: int) -> None:
        v = self.value()
        self.label.setText(f"{v:.{self.decimals}f}")
        self.valueChanged.emit(v)


class MainWindow(QtWidgets.QMainWindow):
    exportFinished = QtCore.Signal()

    def __init__(self, engine: Engine, ips: list[str], save: Callable[[], None],
                 recordings_dir: Path = Path("recordings")):
        super().__init__()
        self.exportFinished.connect(lambda: self.export_button.setEnabled(True))
        self.engine = engine
        self.save = save
        self.recordings_dir = recordings_dir
        self.setWindowTitle("ARKit FBT Relay")
        self.resize(1440, 800)
        self.engine.want_preview = True
        self._preview_id = -1
        self._preview_rot = 0
        self._resized = False

        central = QtWidgets.QWidget()
        root = QtWidgets.QHBoxLayout(central)
        self.setCentralWidget(central)

        left_widget = QtWidgets.QWidget()
        left = QtWidgets.QVBoxLayout(left_widget)
        left.addWidget(self._status_box(ips))
        left.addWidget(self._tracker_box())
        left.addWidget(self._action_box())
        left.addWidget(self._tuning_box())
        left.addStretch(1)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidget(left_widget)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setMinimumWidth(500)
        root.addWidget(scroll, 0)

        right = QtWidgets.QVBoxLayout()
        root.addLayout(right, 1)
        views = QtWidgets.QHBoxLayout()
        views.addWidget(self._preview_box(), 1)
        views.addWidget(self._plots(), 1)
        right.addLayout(views, 1)
        self.log_view = QtWidgets.QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(500)
        self.log_view.setFixedHeight(150)
        right.addWidget(self.log_view)
        self._log_count = 0

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.refresh)
        self.timer.start(33)

    # ------------------------------------------------------------------ 구성

    def _status_box(self, ips: list[str]) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("상태")
        form = QtWidgets.QFormLayout(box)
        self.st: dict[str, QtWidgets.QLabel] = {}
        rows = [
            ("ip", "이 PC 주소"), ("phone", "iPhone"), ("rate", "수신 속도"),
            ("jitter", "수신 간격 흔들림"), ("lost", "손실 / 오류"), ("tracked", "몸 인식"),
            ("facing", "카메라 정면"), ("yaw", "방향 정렬"), ("lean", "자세 기울기 보정"),
            ("gate", "불안정해서 고정 중"), ("quality", "ARKit 추적 품질 (3초)"),
            ("gesture", "손 들기 제스처"),
            ("osc", "VRChat 전송"),
        ]
        for key, title in rows:
            lab = QtWidgets.QLabel("-")
            lab.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
            form.addRow(title, lab)
            self.st[key] = lab
        port = self.engine.bound_port
        self.st["ip"].setText("<br>".join(f"<b>{ip}</b> : {port}" for ip in ips) or "(찾지 못함)")
        return box

    def _tracker_box(self) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("보낼 트래커 (처음엔 골반 + 양발 추천)")
        grid = QtWidgets.QGridLayout(box)
        enabled = set(self.engine.settings.enabled_trackers)
        self.tracker_checks: dict[str, QtWidgets.QCheckBox] = {}
        for i, (_idx, name, _joint, label) in enumerate(TRACKERS):
            cb = QtWidgets.QCheckBox(label)
            cb.setChecked(name in enabled)
            cb.setStyleSheet(f"QCheckBox {{ color: {TRACKER_COLORS[name]}; font-weight: bold; }}")
            cb.toggled.connect(self._trackers_changed)
            grid.addWidget(cb, i // 3, i % 3)
            self.tracker_checks[name] = cb
        return box

    def _tuning_box(self) -> QtWidgets.QGroupBox:
        s = self.engine.settings
        box = QtWidgets.QGroupBox("필터 (떨림 ↔ 반응 속도)")
        form = QtWidgets.QFormLayout(box)

        def slider(attr: str, lo: float, hi: float, tip: str) -> FloatSlider:
            w = FloatSlider(lo, hi, getattr(s, attr))
            w.setToolTip(tip)
            w.valueChanged.connect(lambda v: self.engine.update_settings(**{attr: v}))
            return w

        form.addRow("위치 안정도", slider("min_cutoff", 0.05, 8.0, "낮을수록 떨림↓ 지연↑ (기본 3.0)"))
        form.addRow("위치 반응성", slider("beta", 0.0, 8.0, "높을수록 빠른 동작에서 지연↓ 떨림↑ (기본 3.0)"))
        form.addRow("회전 안정도", slider("rot_min_cutoff", 0.05, 5.0, "낮을수록 떨림↓ 지연↑"))
        form.addRow("회전 반응성", slider("rot_beta", 0.0, 3.0, "높을수록 지연↓ 떨림↑"))
        form.addRow("깊이 방향 필터 비율", slider("depth_ratio", 0.05, 1.0,
                                                "카메라 앞뒤 방향만 더 강하게 거릅니다. 낮을수록 앞뒤 흔들림↓ (기본 0.2, 1 = 같게)"))
        form.addRow("관절 최대 속도(m/s)", slider("max_rel_speed", 0.0, 10.0,
                                               "골반 기준으로 이보다 빠른 움직임은 잘라 튐을 누릅니다. "
                                               "낮으면 빠른 발차기가 늦게 따라옵니다 (기본 3, 0 = 끔)"))
        form.addRow("몸 전체 순간이동 무시(m/s)", slider("max_speed", 2.0, 30.0, "몸 전체가 이보다 빠르게 이동하면 무시 (기본 20)"))
        form.addRow("예측 (ms)", slider("prediction_ms", 0.0, 120.0,
                                         "움직이는 방향으로 미리 내다봐서 지연을 줄입니다. 크면 동작이 튀어나갑니다 (0 = 끔)"))

        spike = QtWidgets.QCheckBox("순간 튐 제거 (3프레임 중앙값, 지연 +17ms)")
        spike.setChecked(s.spike_window > 1)
        spike.toggled.connect(lambda v: self.engine.update_settings(spike_window=3 if v else 1))
        form.addRow(spike)

        head = QtWidgets.QComboBox()
        options = [("chest", "가슴 기준 (추천)"), ("neck", "목 기준"), ("head", "ARKit 머리 그대로")]
        for key, text in options:
            head.addItem(text, key)
        keys = [k for k, _ in options]
        head.setCurrentIndex(keys.index(s.head_horizontal) if s.head_horizontal in keys else 0)
        head.setToolTip("헤드셋을 쓰면 ARKit이 머리를 앞으로 쏠리게 잡아 몸이 밀려 보입니다. 가슴 기준이 가장 정확했습니다.")
        head.currentIndexChanged.connect(lambda _i: self.engine.update_settings(head_horizontal=head.currentData()))
        form.addRow("머리 위치(수평)", head)

        def check(attr: str, text: str, tip: str = "") -> QtWidgets.QCheckBox:
            cb = QtWidgets.QCheckBox(text)
            cb.setChecked(bool(getattr(s, attr)))
            cb.setToolTip(tip)
            cb.toggled.connect(lambda v: self.engine.update_settings(**{attr: v}))
            form.addRow(cb)
            return cb

        check("osc_enabled", "VRChat으로 전송")
        check("quality_gate", "불안정할 때 고정 (ARKit이 몸을 놓치면 마지막 안정 자세 유지)")
        check("lock_yaw_to_camera", "몸 방향을 카메라 정면으로 고정 (허리 회전 안정화)",
              "ARKit 몸 방향 추정은 자주 틀어집니다. 끄면 ARKit 방향을 따르지만 허리가 흔들릴 수 있습니다")
        check("upright_correction", "자세 기울기 보정 (카메라 쪽으로 기우는 오차 제거)")
        check("flat_feet", "발·무릎을 평평하게 (ARKit 발 각도 무시, 몸 방향만 따름)")
        check("send_head_position", "머리 위치 전송 (VRChat 자동 위치 정렬)", "끄면 트래커가 따로 놉니다")
        check("stabilize_camera", "카메라 고정 보정 (ARKit 드리프트 제거)")
        check("auto_yaw_on_start", "시작 시 자동 방향 정렬")
        check("gesture_yaw", "두 손 머리 위로 1.5초 → 방향 정렬")
        check("flip_yaw", "방향 180° 뒤집기 (정렬 후 뒤를 보고 있을 때)")
        return box

    def _action_box(self) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("동작")
        lay = QtWidgets.QGridLayout(box)
        b1 = QtWidgets.QPushButton("지금 방향 정렬")
        b1.setToolTip("카메라를 정면으로 바라보고 누르세요")
        b1.clicked.connect(lambda: self.engine.request_yaw_align("버튼"))
        b2 = QtWidgets.QPushButton("카메라 기준 재설정")
        b2.clicked.connect(self.engine.reset_camera)
        b3 = QtWidgets.QPushButton("설정 저장")
        b3.clicked.connect(self._save)
        self.rec_button = QtWidgets.QPushButton("● 녹화 시작")
        self.rec_button.setToolTip("iPhone에서 오는 원본 데이터와 카메라 미리보기를 파일로 저장합니다 (분석/튜닝용)")
        self.rec_button.clicked.connect(self._toggle_recording)
        self.export_button = QtWidgets.QPushButton("🎬 녹화 → 영상(MP4)")
        self.export_button.setToolTip("녹화 파일을 카메라 영상 + VRChat이 받는 자세가 담긴 MP4로 만듭니다")
        self.export_button.clicked.connect(self._export_video)
        lay.addWidget(b1, 0, 0)
        lay.addWidget(b2, 0, 1)
        lay.addWidget(self.rec_button, 1, 0)
        lay.addWidget(self.export_button, 1, 1)
        lay.addWidget(b3, 2, 0, 1, 2)
        return box

    def _export_video(self) -> None:
        start = str(self.engine.last_recording or self.recordings_dir)
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "영상으로 만들 녹화 파일", start, "녹화 (*.afbtrec)")
        if not path:
            return
        self.export_button.setEnabled(False)
        self.engine.log(f"영상 만드는 중: {Path(path).name} (녹화 길이에 따라 수십 초 걸립니다)")

        def work() -> None:
            try:
                from .video import export_video
                last = [-1]

                def progress(f: float) -> None:
                    step = int(f * 4)
                    if step != last[0]:
                        last[0] = step
                        self.engine.log(f"영상 만드는 중… {f * 100:.0f}%")

                out = export_video(Path(path), progress=progress)
                self.engine.log(f"영상 저장: {out}")
                import subprocess
                subprocess.Popen(["explorer", "/select,", str(out)])
            except Exception as e:  # 버튼이 다시 살아나도록
                self.engine.log(f"영상 만들기 실패: {e!r}")
            finally:
                self.exportFinished.emit()  # 다른 스레드에서 보내도 GUI 스레드에서 처리된다

        threading.Thread(target=work, name="video-export", daemon=True).start()

    def _toggle_recording(self) -> None:
        if self.engine.recording is None:
            name = time.strftime("%Y%m%d-%H%M%S") + ".afbtrec"
            self.engine.start_recording(self.recordings_dir / name)
        else:
            self.engine.stop_recording()

    def _preview_box(self) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("iPhone 카메라")
        lay = QtWidgets.QVBoxLayout(box)
        self.preview_check = QtWidgets.QCheckBox("카메라 미리보기 받기 (거치할 때 켜 두세요)")
        self.preview_check.setChecked(True)
        self.preview_check.toggled.connect(self._preview_toggled)
        lay.addWidget(self.preview_check)
        self.preview_label = QtWidgets.QLabel("iPhone 연결을 기다리는 중…")
        self.preview_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setMinimumSize(320, 320)
        self.preview_label.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored,
                                         QtWidgets.QSizePolicy.Policy.Ignored)
        self.preview_label.setStyleSheet("background:#000; color:#888;")
        lay.addWidget(self.preview_label, 1)
        self.framing_label = QtWidgets.QLabel("")
        self.framing_label.setWordWrap(True)
        lay.addWidget(self.framing_label)
        return box

    def _preview_toggled(self, on: bool) -> None:
        self.engine.want_preview = on
        if not on:
            self.preview_label.clear()
            self.preview_label.setText("미리보기 꺼짐")
            self.framing_label.setText("")
            self._preview_id = -1

    def _plots(self) -> pg.GraphicsLayoutWidget:
        pg.setConfigOptions(antialias=True)
        w = pg.GraphicsLayoutWidget()
        self.views = {}
        for row, (key, title, xl, yl) in enumerate([
            ("front", "정면 (카메라 쪽에서 본 모습)", "X (m)", "높이 (m)"),
            ("top", "위에서 본 모습 (▲ = 카메라)", "X (m)", "Z (m)"),
        ]):
            plot = w.addPlot(row=row, col=0, title=title)
            plot.setAspectLocked(True)
            plot.showGrid(x=True, y=True, alpha=0.25)
            plot.setLabel("bottom", xl)
            plot.setLabel("left", yl)
            bones = pg.PlotDataItem(pen=pg.mkPen("#8d99ae", width=2), connect="pairs")
            joints = pg.ScatterPlotItem(size=6, brush=pg.mkBrush("#8d99ae"), pen=None)
            trackers = pg.ScatterPlotItem(size=16, pen=pg.mkPen("w", width=1))
            head = pg.ScatterPlotItem(size=14, symbol="o", brush=pg.mkBrush("#ffffff"), pen=None)
            facing = pg.PlotDataItem(pen=pg.mkPen("#ffd166", width=3))
            cam = pg.ScatterPlotItem(size=18, symbol="t1", brush=pg.mkBrush("#ffd166"), pen=None)
            for item in (bones, joints, trackers, head, facing, cam):
                plot.addItem(item)
            self.views[key] = dict(plot=plot, bones=bones, joints=joints, trackers=trackers,
                                   head=head, facing=facing, cam=cam)
        self.views["front"]["plot"].setXRange(-1.2, 1.2)
        self.views["front"]["plot"].setYRange(-0.2, 2.0)
        self.views["top"]["plot"].setXRange(-2, 2)
        self.views["top"]["plot"].setYRange(-4, 0.5)
        return w

    # ------------------------------------------------------------------ 이벤트

    def _trackers_changed(self) -> None:
        names = [n for n, cb in self.tracker_checks.items() if cb.isChecked()]
        self.engine.update_settings(enabled_trackers=names)

    def _save(self) -> None:
        try:
            self.save()
            self.engine.log("설정을 저장했습니다")
        except OSError as e:
            self.engine.log(f"설정 저장 실패: {e}")

    # ------------------------------------------------------------------ 갱신

    def refresh(self) -> None:
        snap = self.engine.snapshot()
        st = self.st
        ok, bad = "color:#2ec4b6", "color:#e71d36"

        def mark(label: QtWidgets.QLabel, good: bool, text: str) -> None:
            label.setText(text)
            label.setStyleSheet(ok if good else bad)

        mark(st["phone"], snap.phone_alive, snap.phone_addr if snap.phone_alive else "대기 중")
        st["rate"].setText(f"{snap.packet_rate:.1f} /s")
        st["jitter"].setText(f"{snap.interval_jitter_ms:.1f} ms")
        st["lost"].setText(f"{snap.lost_packets} / {snap.bad_packets}")
        mark(st["tracked"], snap.tracked, "인식 중" if snap.tracked else "없음")
        mark(st["facing"], snap.facing > 0.85, f"{snap.facing:+.2f}")
        if snap.yaw_aligned_at:
            ago = time.time() - snap.yaw_aligned_at
            mark(st["yaw"], True, f"{snap.last_yaw:.0f}° ({ago:.0f}초 전)")
        else:
            mark(st["yaw"], False, "아직 안 함")
        names = {j: label for _i, _n, j, label in TRACKERS}
        names.update({"head_joint": "머리", "neck_1_joint": "목", "spine_7_joint": "가슴",
                      "left_upLeg_joint": "왼허벅지", "right_upLeg_joint": "오른허벅지"})
        gated = [names.get(g, g) for g in snap.gated]
        mark(st["gate"], not gated, ", ".join(gated) if gated else "없음")
        q = snap.track_quality
        mark(st["quality"], q is not None and q >= 0.9, "-" if q is None else f"{q * 100:.0f}%")
        if snap.lean_deg is not None:
            mark(st["lean"], True, f"{snap.lean_deg:.1f}° 바로 세우는 중")
        else:
            mark(st["lean"], not self.engine.settings.upright_correction,
                 "꺼짐" if not self.engine.settings.upright_correction else "전신이 보이면 시작")
        st["gesture"].setText("■" * round(snap.gesture_progress * 10) + "□" * (10 - round(snap.gesture_progress * 10)))
        osc_on = self.engine.settings.osc_enabled and not self.engine.dry_run
        mark(st["osc"], osc_on, f"{'켜짐' if osc_on else '꺼짐'} · {snap.osc_sent} msg"
             + (f" · 오류 {snap.osc_errors}" if snap.osc_errors else ""))

        rec = self.engine.recording
        if rec is None:
            self.rec_button.setText("● 녹화 시작")
            self.rec_button.setStyleSheet("")
        else:
            self.rec_button.setText(f"■ 녹화 중지 ({time.monotonic() - rec.started:.0f}초)")
            self.rec_button.setStyleSheet("color:#e71d36; font-weight:bold;")

        new_logs = snap.logs[self._log_count:] if len(snap.logs) >= self._log_count else snap.logs
        for line in new_logs:
            self.log_view.appendPlainText(line)
        self._log_count = len(snap.logs)

        self._draw(snap.frame)
        self._draw_preview(snap)

    def _draw_preview(self, snap) -> None:
        if not self.engine.want_preview:
            return
        if snap.frame is not None and snap.frame.camera_rot is not None:
            self._preview_rot = display_rotation(snap.frame.camera_rot)
        p = snap.preview
        if p is None:
            if self._preview_id != -1 or not snap.phone_alive:
                self.preview_label.clear()
                self.preview_label.setText("iPhone 연결을 기다리는 중…" if not snap.phone_alive
                                           else "미리보기를 받는 중…")
                self._preview_id = -1
            return
        if p.frame_id == self._preview_id and not self._resized:
            return
        self._preview_id = p.frame_id
        self._resized = False

        img = QtGui.QImage.fromData(p.jpeg, "JPG")
        if img.isNull():
            return
        img = img.convertToFormat(QtGui.QImage.Format.Format_RGB32)
        self._paint_skeleton(img, p.joints)
        if self._preview_rot:
            img = img.transformed(QtGui.QTransform().rotate(-90 * self._preview_rot))
        pix = QtGui.QPixmap.fromImage(img).scaled(
            self.preview_label.size(), QtCore.Qt.AspectRatioMode.KeepAspectRatio,
            QtCore.Qt.TransformationMode.SmoothTransformation)
        self.preview_label.setPixmap(pix)

        warnings = framing_warnings(p.joints, JOINT_NAMES, self._preview_rot)
        q = snap.track_quality
        q_text = f"ARKit 추적 품질 {q * 100:.0f}%" if q is not None else "ARKit 추적 품질 -"
        if warnings:
            self.framing_label.setText(f"⚠ {', '.join(warnings)}  ({q_text})")
            self.framing_label.setStyleSheet("color:#ffd166; font-weight:bold;")
        elif q is not None and q < 0.9:
            self.framing_label.setText(f"⚠ {q_text} — 정면을 보고, 배경이 단순하고 밝은 곳에 서 보세요")
            self.framing_label.setStyleSheet("color:#ffd166; font-weight:bold;")
        else:
            self.framing_label.setText(f"✔ 위치 좋음  ({q_text})")
            self.framing_label.setStyleSheet("color:#2ec4b6; font-weight:bold;")

    def _paint_skeleton(self, img: QtGui.QImage, joints) -> None:
        if not joints:
            return
        w, h = img.width(), img.height()
        idx = {n: i for i, n in enumerate(JOINT_NAMES)}

        def pt(name: str):
            i = idx[name]
            p = joints[i] if i < len(joints) else None
            return None if p is None else QtCore.QPointF(p[0] * w, p[1] * h)

        painter = QtGui.QPainter(img)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        painter.setPen(QtGui.QPen(QtGui.QColor("#d0d6e0"), 2))
        for a, b in BONES:
            pa, pb = pt(a), pt(b)
            if pa is not None and pb is not None:
                painter.drawLine(pa, pb)
        enabled = set(self.engine.settings.enabled_trackers)
        for _i, name, joint, _label in TRACKERS:
            if name not in enabled:
                continue
            p = pt(joint)
            if p is not None:
                painter.setPen(QtGui.QPen(QtGui.QColor("white"), 1))
                painter.setBrush(QtGui.QColor(TRACKER_COLORS[name]))
                painter.drawEllipse(p, 6, 6)
        head = pt("head_joint")
        if head is not None:
            painter.setPen(QtCore.Qt.PenStyle.NoPen)
            painter.setBrush(QtGui.QColor("white"))
            painter.drawEllipse(head, 5, 5)
        painter.end()

    def resizeEvent(self, event) -> None:
        self._resized = True
        super().resizeEvent(event)

    def _draw(self, frame) -> None:
        for key, v in self.views.items():
            if frame is None or not frame.tracked:
                for item in ("bones", "joints", "trackers", "head", "facing"):
                    v[item].setData([], [])
                if frame is not None and key == "top":
                    v["cam"].setData(*self._xy(key, [frame.camera_pos]))
                continue
            sk = frame.skeleton
            pts = [p for a, b in BONES if a in sk and b in sk for p in (sk[a], sk[b])]
            v["bones"].setData(*self._xy(key, pts))
            v["joints"].setData(*self._xy(key, list(sk.values())))
            if frame.trackers:
                xs, ys = self._xy(key, [t.pos for t in frame.trackers])
                v["trackers"].setData(xs, ys, brush=[pg.mkBrush(TRACKER_COLORS[t.name]) for t in frame.trackers])
            else:
                v["trackers"].setData([], [])
            if frame.head_pos is not None:
                v["head"].setData(*self._xy(key, [frame.head_pos]))
                if frame.forward is not None:
                    tip = frame.head_pos + frame.forward * 0.4
                    v["facing"].setData(*self._xy(key, [frame.head_pos, tip]))
            if key == "top":
                v["cam"].setData(*self._xy(key, [frame.camera_pos]))

    @staticmethod
    def _xy(key: str, pts) -> tuple[np.ndarray, np.ndarray]:
        if not pts:
            return np.array([]), np.array([])
        a = np.asarray(pts, dtype=float)
        if key == "front":
            return a[:, 0], a[:, 1]
        return a[:, 0], a[:, 2]


def run_gui(engine: Engine, ips: list[str], save: Callable[[], None], recordings_dir: Path) -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    app.setStyle("Fusion")
    pal = QtGui.QPalette()
    pal.setColor(QtGui.QPalette.ColorRole.Window, QtGui.QColor("#1e1f26"))
    pal.setColor(QtGui.QPalette.ColorRole.WindowText, QtGui.QColor("#e6e6e6"))
    pal.setColor(QtGui.QPalette.ColorRole.Base, QtGui.QColor("#15161b"))
    pal.setColor(QtGui.QPalette.ColorRole.Text, QtGui.QColor("#e6e6e6"))
    pal.setColor(QtGui.QPalette.ColorRole.Button, QtGui.QColor("#2b2d36"))
    pal.setColor(QtGui.QPalette.ColorRole.ButtonText, QtGui.QColor("#e6e6e6"))
    app.setPalette(pal)
    win = MainWindow(engine, ips, save, recordings_dir)
    win.show()
    app.exec()
