"""UDP 수신 -> 처리 -> OSC 송신 루프 (백그라운드 스레드)."""

from __future__ import annotations

import socket
import statistics
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .osc_out import OscOut
from .pipeline import FrameOut, Processor, Settings, unity_yaw_from_forward
from .preview import Preview, Reassembler, is_preview_chunk
from .recording import Recorder
from .protocol import (
    STATUS_DRY_RUN, STATUS_OSC_SENDING, STATUS_WANT_PREVIEW, STATUS_YAW_ALIGNED, ProtocolError,
    build_ack, parse_frame,
)

ACK_INTERVAL = 0.2
AUTO_YAW_STABLE_SEC = 1.5
AUTO_YAW_MIN_FACING = 0.85
GESTURE_HOLD_SEC = 1.5
GESTURE_COOLDOWN_SEC = 5.0


@dataclass
class Snapshot:
    phone_addr: str | None = None
    phone_alive: bool = False
    packet_rate: float = 0.0
    interval_jitter_ms: float = 0.0
    lost_packets: int = 0
    bad_packets: int = 0
    tracked: bool = False
    facing: float = 0.0
    yaw_aligned_at: float | None = None
    last_yaw: float | None = None
    gesture_progress: float = 0.0
    osc_sent: int = 0
    osc_errors: int = 0
    frame: FrameOut | None = None
    preview: Preview | None = None
    logs: list[str] = field(default_factory=list)


class Engine:
    def __init__(self, settings: Settings, listen_host: str = "0.0.0.0", listen_port: int = 39570,
                 osc_host: str = "127.0.0.1", osc_port: int = 9000, dry_run: bool = False,
                 osc_recorder: Callable[[str, list[float]], None] | None = None,
                 log: Callable[[str], None] | None = None):
        self.settings = settings
        self.processor = Processor(settings)
        self.osc = OscOut(osc_host, osc_port, dry_run=dry_run, recorder=osc_recorder)
        self.dry_run = dry_run
        self.listen = (listen_host, listen_port)
        self._extra_log = log

        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._sock: socket.socket | None = None

        self._logs: deque[str] = deque(maxlen=300)
        self._yaw_request: str | None = None
        self._yaw_aligned_at: float | None = None
        self._last_yaw: float | None = None
        self._auto_yaw_done = False
        self._tracked_since: float | None = None
        self._hands_up_since: float | None = None
        self._last_gesture = -1e9
        self._gesture_armed = True

        self._arrivals: deque[float] = deque(maxlen=120)
        self._last_seq: int | None = None
        self._lost = 0
        self._bad = 0
        self._phone_addr: tuple[str, int] | None = None
        self._last_packet = 0.0
        self._last_ack = 0.0
        self._last_frame: FrameOut | None = None
        self._gesture_progress = 0.0
        self._recorder: Recorder | None = None
        # GUI가 켜져 있을 때 iPhone에 카메라 미리보기를 요청한다
        self.want_preview = False
        self._reassembler = Reassembler()
        self._preview: Preview | None = None

    # ------------------------------------------------------------------ 제어

    def start(self) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        if hasattr(socket, "SIO_UDP_CONNRESET"):
            # Windows: 상대 포트가 닫혀 있을 때 recvfrom 이 예외를 던지지 않게 한다
            sock.ioctl(socket.SIO_UDP_CONNRESET, False)
        sock.bind(self.listen)
        sock.settimeout(0.25)
        self._sock = sock
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="relay", daemon=True)
        self._thread.start()
        self.log(f"수신 대기: {self.listen[0]}:{self.listen[1]} → OSC {self.osc.host}:{self.osc.port}"
                 + (" (dry-run)" if self.dry_run else ""))

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
        if self._sock is not None:
            self._sock.close()
        self.stop_recording()

    def start_recording(self, path: Path) -> None:
        with self._lock:
            if self._recorder is not None:
                return
            self._recorder = Recorder(path)
        self.log(f"녹화 시작: {path}")

    def stop_recording(self) -> Path | None:
        with self._lock:
            rec, self._recorder = self._recorder, None
        if rec is None:
            return None
        rec.close()
        self.log(f"녹화 저장: {rec.path} ({rec.count}개, {time.monotonic() - rec.started:.0f}초)")
        return rec.path

    @property
    def recording(self) -> Recorder | None:
        return self._recorder

    @property
    def bound_port(self) -> int:
        return self._sock.getsockname()[1] if self._sock else self.listen[1]

    def log(self, msg: str) -> None:
        line = time.strftime("%H:%M:%S ") + msg
        with self._lock:
            self._logs.append(line)
        if self._extra_log:
            self._extra_log(line)

    def request_yaw_align(self, reason: str = "수동") -> None:
        with self._lock:
            self._yaw_request = reason

    def reset_camera(self) -> None:
        with self._lock:
            self.processor.reset_camera()
            self._auto_yaw_done = False
        self.log("카메라 기준을 다시 잡습니다")

    def update_settings(self, **changes) -> None:
        with self._lock:
            for k, v in changes.items():
                setattr(self.settings, k, v)
            self.processor.apply_filter_settings()

    def snapshot(self) -> Snapshot:
        now = time.monotonic()
        with self._lock:
            arrivals = list(self._arrivals)
            rate = 0.0
            jitter = 0.0
            recent = [a for a in arrivals if now - a < 1.0]
            if len(recent) >= 2:
                rate = (len(recent) - 1) / max(recent[-1] - recent[0], 1e-6)
            if len(arrivals) >= 3:
                intervals = np.diff(arrivals[-60:]) * 1000
                jitter = float(statistics.pstdev(intervals))
            alive = now - self._last_packet < 1.0
            frame = self._last_frame
            return Snapshot(
                phone_addr=f"{self._phone_addr[0]}:{self._phone_addr[1]}" if self._phone_addr else None,
                phone_alive=alive,
                packet_rate=rate if alive else 0.0,
                interval_jitter_ms=jitter if alive else 0.0,
                lost_packets=self._lost,
                bad_packets=self._bad,
                tracked=bool(alive and frame and frame.tracked),
                facing=frame.facing if frame else 0.0,
                yaw_aligned_at=self._yaw_aligned_at,
                last_yaw=self._last_yaw,
                gesture_progress=self._gesture_progress,
                osc_sent=self.osc.sent,
                osc_errors=self.osc.errors,
                frame=frame if alive else None,
                preview=self._preview if alive and self.want_preview else None,
                logs=list(self._logs),
            )

    # ------------------------------------------------------------------ 루프

    def _loop(self) -> None:
        assert self._sock is not None
        while not self._stop.is_set():
            try:
                data, addr = self._sock.recvfrom(65535)
            except TimeoutError:
                continue
            except OSError:
                if self._stop.is_set():
                    break
                continue
            try:
                self.handle_packet(data, addr)
            except Exception as e:  # 루프가 죽지 않도록
                self.log(f"처리 오류: {e!r}")

    def handle_packet(self, data: bytes, addr: tuple[str, int]) -> None:
        now = time.monotonic()
        if is_preview_chunk(data):
            if self.want_preview:
                preview = self._reassembler.add(data)
                if preview is not None:
                    with self._lock:
                        self._preview = preview
            return
        with self._lock:
            if self._recorder is not None:
                self._recorder.write(now, data)
        try:
            frame = parse_frame(data)
        except ProtocolError as e:
            with self._lock:
                self._bad += 1
                bad = self._bad
            if bad in (1, 10, 100) or bad % 1000 == 0:
                self.log(f"잘못된 패킷 ({addr[0]}): {e} [누적 {bad}]")
            return

        with self._lock:
            if self._phone_addr != addr:
                self._phone_addr = addr
                self._last_seq = None
                self.log(f"iPhone 연결: {addr[0]}:{addr[1]}")
            if self._last_seq is not None:
                gap = (frame.seq - self._last_seq) & 0xFFFFFFFF
                if 1 < gap < 1000:
                    self._lost += gap - 1
            self._last_seq = frame.seq
            self._arrivals.append(now)
            self._last_packet = now

            out = self.processor.process(frame)
            self._last_frame = out
            for ev in out.events:
                if ev == "camera_moved":
                    self._auto_yaw_done = False
                    self.log("폰이 움직인 것 같습니다 → 카메라 기준 재설정")

            self._update_triggers(out, now)
            self._send_osc(out)

        if now - self._last_ack >= ACK_INTERVAL and self._sock is not None:
            self._last_ack = now
            try:
                self._sock.sendto(build_ack(frame.seq, frame.timestamp, self._status()), addr)
            except OSError:
                pass

    def _status(self) -> int:
        s = 0
        if self.settings.osc_enabled and not self.dry_run:
            s |= STATUS_OSC_SENDING
        if self._yaw_aligned_at is not None:
            s |= STATUS_YAW_ALIGNED
        if self.dry_run:
            s |= STATUS_DRY_RUN
        if self.want_preview:
            s |= STATUS_WANT_PREVIEW
        return s

    def _update_triggers(self, out: FrameOut, now: float) -> None:
        s = self.settings
        if not out.tracked:
            self._tracked_since = None
            self._hands_up_since = None
            self._gesture_progress = 0.0
            return
        if self._tracked_since is None:
            self._tracked_since = now

        if (s.auto_yaw_on_start and not self._auto_yaw_done and self._yaw_request is None
                and now - self._tracked_since >= AUTO_YAW_STABLE_SEC
                and out.facing >= AUTO_YAW_MIN_FACING):
            self._auto_yaw_done = True
            self._yaw_request = "자동 (카메라 정면)"

        if s.gesture_yaw:
            if out.hands_up:
                if not self._gesture_armed:
                    self._gesture_progress = 0.0
                elif self._hands_up_since is None:
                    self._hands_up_since = now
                else:
                    held = now - self._hands_up_since
                    self._gesture_progress = min(1.0, held / GESTURE_HOLD_SEC)
                    if held >= GESTURE_HOLD_SEC and now - self._last_gesture >= GESTURE_COOLDOWN_SEC:
                        self._last_gesture = now
                        self._gesture_armed = False  # 손을 내렸다 다시 들어야 재발동
                        self._yaw_request = "두 손 들기 제스처"
            else:
                self._hands_up_since = None
                self._gesture_progress = 0.0
                self._gesture_armed = True

    def _send_osc(self, out: FrameOut) -> None:
        s = self.settings
        if not s.osc_enabled or not out.tracked:
            return
        for t in out.trackers:
            self.osc.send_tracker(t)
        if s.send_head_position and out.head_pos is not None:
            self.osc.send_head_position(out.head_pos)
        if self._yaw_request is not None and out.forward is not None:
            yaw = unity_yaw_from_forward(out.forward, s.flip_yaw)
            self.osc.send_head_yaw(yaw)
            self._yaw_aligned_at = time.time()
            self._last_yaw = yaw
            self.log(f"방향 정렬 [{self._yaw_request}] yaw={yaw:.1f}°")
            self._yaw_request = None
