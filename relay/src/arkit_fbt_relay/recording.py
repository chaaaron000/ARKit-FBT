"""iPhone 패킷 녹화/재생/분석.

녹화 파일(.afbtrec): 헤더 b"AFBTREC1" 다음에 레코드가 반복된다.
    f64 수신 시각(PC monotonic, 초) | u32 길이 | 원본 패킷 바이트

    uv run python -m arkit_fbt_relay.recording analyze recordings/xxx.afbtrec
    uv run python -m arkit_fbt_relay.recording replay  recordings/xxx.afbtrec [--speed 1.0]
"""

from __future__ import annotations

import argparse
import socket
import struct
import time
from collections.abc import Iterator
from pathlib import Path
from typing import BinaryIO

import numpy as np

from .mathutil import Rigid, quat_angle, quat_normalize
from .pipeline import Processor, Settings
from .protocol import DEFAULT_PORT, JOINT_NAMES, ProtocolError, parse_frame

FILE_MAGIC = b"AFBTREC1"
RECORD = struct.Struct("<dI")


class Recorder:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._f: BinaryIO | None = path.open("wb")
        self._f.write(FILE_MAGIC)
        self.count = 0
        self.started = time.monotonic()

    def write(self, t: float, data: bytes) -> None:
        if self._f is None:
            return
        self._f.write(RECORD.pack(t, len(data)))
        self._f.write(data)
        self.count += 1

    def close(self) -> None:
        if self._f is not None:
            self._f.close()
            self._f = None


def read_records(path: Path) -> Iterator[tuple[float, bytes]]:
    with path.open("rb") as f:
        if f.read(len(FILE_MAGIC)) != FILE_MAGIC:
            raise ValueError(f"녹화 파일이 아닙니다: {path}")
        while True:
            head = f.read(RECORD.size)
            if len(head) < RECORD.size:
                return
            t, n = RECORD.unpack(head)
            data = f.read(n)
            if len(data) < n:
                return
            yield t, data


# ---------------------------------------------------------------------- 분석

def _pct(a: float, b: float) -> str:
    return f"{100 * a / b:5.1f}%" if b else "  -  "


def analyze(path: Path) -> str:
    """녹화 파일에서 튜닝에 필요한 숫자를 뽑는다."""
    recv_t, frames, bad = [], [], 0
    for t, data in read_records(path):
        if data[:4] != b"AFBT":  # 카메라 미리보기 조각 등
            continue
        try:
            frames.append(parse_frame(data))
            recv_t.append(t)
        except ProtocolError:
            bad += 1
    lines = [f"파일: {path.name}"]
    if len(frames) < 2:
        return "\n".join(lines + ["프레임이 너무 적습니다."])

    recv = np.array(recv_t)
    cap = np.array([f.timestamp for f in frames])
    seqs = np.array([f.seq for f in frames], dtype=np.int64)
    dur = recv[-1] - recv[0]
    lost = int(np.sum(np.clip(np.diff(seqs) - 1, 0, None)))
    tracked = [f for f in frames if f.tracked and f.joints]
    lines += [
        f"길이 {dur:.1f}초, 패킷 {len(frames)}개 (평균 {len(frames) / dur:.1f}/s), 손실 {lost}, 깨짐 {bad}",
        f"몸 인식 비율 {_pct(len(tracked), len(frames))}",
    ]

    # 캡처 간격 = iPhone 쪽 프레임 속도, (수신 - 캡처) 변동 = 처리+네트워크 지연의 흔들림
    cap_dt = np.diff(cap) * 1000
    offset = (recv - cap) * 1000
    offset -= offset.min()
    lines += [
        f"iPhone 프레임 간격: 중앙값 {np.median(cap_dt):.1f}ms, 95% {np.percentile(cap_dt, 95):.1f}ms",
        "지연 변동(캡처→PC 수신, 최소 대비): "
        f"중앙값 +{np.median(offset):.1f}ms, 95% +{np.percentile(offset, 95):.1f}ms, 최대 +{offset.max():.1f}ms",
    ]

    # 카메라 드리프트
    cams = [Rigid(quat_normalize(f.camera.rot), f.camera.pos) for f in frames]
    moves = [np.linalg.norm(c.pos - cams[0].pos) for c in cams]
    turn = max(np.degrees(quat_angle((cams[0].inverse() @ c).rot)) for c in cams)
    lines.append(f"ARKit 카메라 위치 변화(드리프트): 최대 {max(moves) * 100:.1f}cm, 회전 최대 {turn:.2f}°")

    if not tracked:
        return "\n".join(lines)

    lines.append(f"스케일 추정값: {np.median([f.scale for f in tracked]):.3f}")
    lines.append("")
    lines.append("관절                     인식률   떨림(2차차분 std, mm)")
    for name in JOINT_NAMES:
        present = [f.joints[name].pos for f in tracked if name in f.joints]
        jitter = ""
        if len(present) >= 5:
            p = np.array(present)
            jitter = f"{np.std(np.linalg.norm(np.diff(p, 2, axis=0), axis=1)) * 1000:7.2f}"
        lines.append(f"  {name:<22} {_pct(len(present), len(tracked))}  {jitter}")

    # 파이프라인 통과 결과: 몸 방향, 정면 학습
    proc = Processor(Settings())
    facing, hands = [], 0
    for f in frames:
        out = proc.process(f)
        if out.tracked:
            facing.append(out.facing)
            hands += out.hands_up
    if facing:
        fa = np.array(facing)
        lines += [
            "",
            f"카메라 정면 점수: 중앙값 {np.median(fa):+.2f} (1에 가까울수록 정면), "
            f"0.85 이상 {_pct(np.sum(fa >= 0.85), len(fa))}",
            f"좌우 규약 학습값: {proc._sign_score:+.1f} (양수 = ARKit left_ 가 사람 기준 왼쪽)",
            f"손 들기 감지 프레임: {hands}",
        ]
    root = tracked[len(tracked) // 2]
    hips = root.joints.get("hips_joint")
    if hips is not None:
        to_cam = root.camera.pos - hips.pos
        lines.append(f"(중간 프레임) 카메라까지 거리 {np.linalg.norm(to_cam):.2f}m, 골반 높이(카메라 기준) {-to_cam[1]:+.2f}m")
    return "\n".join(lines)


# ---------------------------------------------------------------------- 재생

def replay(path: Path, host: str, port: int, speed: float = 1.0, loop: bool = False) -> int:
    records = list(read_records(path))
    if not records:
        print("빈 파일입니다.")
        return 1
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    while True:
        t0 = records[0][0]
        start = time.perf_counter()
        for t, data in records:
            delay = (t - t0) / speed - (time.perf_counter() - start)
            if delay > 0:
                time.sleep(delay)
            sock.sendto(data, (host, port))
        print(f"{len(records)}개 재생 완료")
        if not loop:
            return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="녹화 파일 분석/재생")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("analyze")
    a.add_argument("file", type=Path)
    r = sub.add_parser("replay")
    r.add_argument("file", type=Path)
    r.add_argument("--host", default="127.0.0.1")
    r.add_argument("--port", type=int, default=DEFAULT_PORT)
    r.add_argument("--speed", type=float, default=1.0)
    r.add_argument("--loop", action="store_true")
    v = sub.add_parser("export-video", help="카메라 영상 + VRChat이 받는 자세를 MP4로")
    v.add_argument("file", type=Path)
    v.add_argument("-o", "--out", type=Path)
    v.add_argument("--fps", type=float, default=15.0)
    args = ap.parse_args(argv)
    if args.cmd == "analyze":
        print(analyze(args.file))
        return 0
    if args.cmd == "export-video":
        from .video import export_video
        out = export_video(args.file, args.out, args.fps,
                           progress=lambda f: print(f"\r영상 만드는 중 {f * 100:3.0f}%", end="", flush=True))
        print(f"\n저장: {out}")
        return 0
    return replay(args.file, args.host, args.port, args.speed, args.loop)


if __name__ == "__main__":
    raise SystemExit(main())
