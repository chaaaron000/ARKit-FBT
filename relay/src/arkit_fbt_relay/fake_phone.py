"""가짜 iPhone: 움직이는 사람의 관절 데이터를 만들어 중계 프로그램으로 보낸다.

    uv run python -m arkit_fbt_relay.fake_phone --seconds 10
"""

from __future__ import annotations

import argparse
import math
import socket
import time

import numpy as np

from .mathutil import Rigid, quat_from_axis_angle, quat_mul
from .protocol import DEFAULT_PORT, JOINT_NAMES, Pose, build_frame, parse_ack

# 카메라(+Z)를 바라보고 선 사람의 관절 (골반 기준, 미터). 사람의 왼쪽 = +X
REST: dict[str, tuple[float, float, float]] = {
    "hips_joint": (0.0, 0.0, 0.0), "spine_4_joint": (0.0, 0.2, 0.0), "spine_7_joint": (0.0, 0.4, 0.0),
    "neck_1_joint": (0.0, 0.55, 0.0), "head_joint": (0.0, 0.65, 0.0),
    "left_shoulder_1_joint": (0.08, 0.47, 0.0), "left_arm_joint": (0.19, 0.45, 0.0),
    "left_forearm_joint": (0.24, 0.17, 0.0), "left_hand_joint": (0.27, -0.08, 0.02),
    "right_shoulder_1_joint": (-0.08, 0.47, 0.0), "right_arm_joint": (-0.19, 0.45, 0.0),
    "right_forearm_joint": (-0.24, 0.17, 0.0), "right_hand_joint": (-0.27, -0.08, 0.02),
    "left_upLeg_joint": (0.1, -0.03, 0.0), "left_leg_joint": (0.1, -0.45, 0.02),
    "left_foot_joint": (0.1, -0.87, 0.0), "left_toes_joint": (0.1, -0.93, 0.13),
    "right_upLeg_joint": (-0.1, -0.03, 0.0), "right_leg_joint": (-0.1, -0.45, 0.02),
    "right_foot_joint": (-0.1, -0.87, 0.0), "right_toes_joint": (-0.1, -0.93, 0.13),
}
HIP_HEIGHT = 0.95
DISTANCE = 2.6


def synth_joints(t: float, body_yaw: float = 0.0, hands_up: bool = False,
                 noise: float = 0.0, rng: np.random.Generator | None = None) -> dict[str, Pose]:
    """시각 t의 관절 자세 (ARKit 월드, 카메라는 원점에서 -Z를 본다)."""
    sway = 0.15 * math.sin(t * 0.8)
    step = math.sin(t * 2.0)
    yaw_q = quat_mul(quat_from_axis_angle([0, 1, 0], math.pi), quat_from_axis_angle([0, 1, 0], body_yaw))
    body = Rigid(yaw_q, np.array([sway, HIP_HEIGHT, -DISTANCE]))
    # yaw_q: 기본 정면(-Z)을 180° 돌려 카메라(+Z)를 보게 한다 -> 로컬 +X(왼쪽)이 월드 -X가 된다
    # REST는 '사람 왼쪽 = 로컬 +X' 기준이므로 회전 전에 X를 뒤집어 로컬로 바꾼다
    joints: dict[str, Pose] = {}
    for name in JOINT_NAMES:
        x, y, z = REST[name]
        local = np.array([-x, y, -z])
        if name.startswith("left_") and ("foot" in name or "toes" in name or "leg_" in name):
            local[1] += 0.08 * max(0.0, step)
        if name.startswith("right_") and ("foot" in name or "toes" in name or "leg_" in name):
            local[1] += 0.08 * max(0.0, -step)
        if hands_up and ("forearm" in name or "hand" in name):
            local[1] = 0.85 if "hand" in name else 0.72
        p = body.apply(local)
        if noise and rng is not None:
            p = p + rng.normal(0, noise, 3)
        joints[name] = Pose(p, body.rot.copy())
    return joints


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="가짜 iPhone 송신기")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--rate", type=float, default=60.0)
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--noise", type=float, default=0.01, help="관절 위치 잡음 (m)")
    ap.add_argument("--drift", type=float, default=0.0, help="ARKit 월드 드리프트 (m/s)")
    ap.add_argument("--hands-up-at", type=float, default=-1, help="이 시각(초)부터 2초간 두 손 들기")
    args = ap.parse_args(argv)

    rng = np.random.default_rng(0)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(0)
    start = time.perf_counter()
    seq = 0
    rtts: list[float] = []
    status = 0
    cam_local = Rigid(np.array([0.0, 0.0, 0.0, 1.0]), np.array([0.0, 1.0, 0.0]))
    while (now := time.perf_counter() - start) < args.seconds:
        seq += 1
        hands = args.hands_up_at >= 0 and args.hands_up_at <= now < args.hands_up_at + 2.0
        drift = Rigid(quat_from_axis_angle([0, 1, 0], 0.002 * now * (args.drift > 0)),
                      np.array([args.drift * now, 0.0, 0.0]))
        cam = drift @ cam_local
        joints = {n: Pose(drift.apply(p.pos), quat_mul(drift.rot, p.rot))
                  for n, p in synth_joints(now, hands_up=hands, noise=args.noise, rng=rng).items()}
        ts = time.perf_counter()
        sock.sendto(build_frame(seq, ts, Pose(cam.pos, cam.rot), joints), (args.host, args.port))
        try:
            while True:
                data, _ = sock.recvfrom(64)
                status, _seq, echo = parse_ack(data)
                rtts.append((time.perf_counter() - echo) * 1000)
        except (BlockingIOError, OSError):
            pass
        time.sleep(max(0.0, seq / args.rate - (time.perf_counter() - start)))
    print(f"보낸 패킷 {seq}, 받은 응답 {len(rtts)}, 상태 0x{status:02x}"
          + (f", 평균 왕복 {np.mean(rtts):.2f} ms" if rtts else ""))
    return 0 if rtts else 1


if __name__ == "__main__":
    raise SystemExit(main())
