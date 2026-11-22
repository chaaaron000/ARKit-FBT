"""ARKit 관절 데이터 -> VRChat 트래커 자세 변환 파이프라인.

1. 카메라 고정 보정: 폰은 삼각대에 고정되어 있다고 보고, ARKit 월드 좌표의
   미끄러짐(VIO 드리프트)을 카메라 자세로 되돌린다.
2. 튀는 값 제거 + One Euro 필터.
3. 몸 방향(yaw) 계산: VRChat 방향 정렬에 쓴다.

위치 정렬은 VRChat이 /tracking/trackers/head/position 으로 매 프레임 해 준다.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field

import numpy as np

from .filters import OneEuroFilter, QuatOneEuroFilter
from .mathutil import Rigid, quat_angle, quat_mul, quat_normalize
from .protocol import BodyFrame

# (OSC 번호, 이름, ARKit 관절, 표시 이름)
TRACKERS: tuple[tuple[int, str, str, str], ...] = (
    (1, "hip", "hips_joint", "골반"),
    (2, "left_foot", "left_foot_joint", "왼발"),
    (3, "right_foot", "right_foot_joint", "오른발"),
    (4, "left_knee", "left_leg_joint", "왼무릎"),
    (5, "right_knee", "right_leg_joint", "오른무릎"),
    (6, "chest", "spine_7_joint", "가슴"),
    (7, "left_elbow", "left_forearm_joint", "왼팔꿈치"),
    (8, "right_elbow", "right_forearm_joint", "오른팔꿈치"),
)
HEAD_JOINT = "head_joint"

UP = np.array([0.0, 1.0, 0.0])


@dataclass
class Settings:
    enabled_trackers: list[str] = field(default_factory=lambda: ["hip", "left_foot", "right_foot"])
    # 위치 필터 (2026-11-22 iPhone XR 실측 녹화로 고른 값: 원본 대비 떨림 절반, 오차도 감소)
    min_cutoff: float = 3.0
    beta: float = 3.0
    # 회전 필터
    rot_min_cutoff: float = 1.0
    rot_beta: float = 0.3
    # 한 프레임 이동 속도가 이보다 빠르면 오인식으로 보고 잠깐 무시 (m/s).
    # 걷거나 발을 찰 때 발은 실제로 6m/s 이상 나오므로, 순간이동 수준만 거른다.
    max_speed: float = 20.0
    # 지연을 줄이기 위해 속도 방향으로 미리 내다보는 시간 (ms). 크면 동작이 튀어나간다.
    prediction_ms: float = 0.0
    stabilize_camera: bool = True
    send_head_position: bool = True
    auto_yaw_on_start: bool = True
    gesture_yaw: bool = True
    flip_yaw: bool = False
    osc_enabled: bool = True

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> Settings:
        names = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})


@dataclass
class TrackerOut:
    index: int
    name: str
    pos: np.ndarray  # ARKit 월드(보정 후)
    rot: np.ndarray


@dataclass
class FrameOut:
    tracked: bool
    camera_pos: np.ndarray
    camera_rot: np.ndarray | None = None  # 보정 후 카메라 회전 (미리보기 방향 계산용)
    events: list[str] = field(default_factory=list)
    skeleton: dict[str, np.ndarray] = field(default_factory=dict)
    trackers: list[TrackerOut] = field(default_factory=list)
    head_pos: np.ndarray | None = None
    forward: np.ndarray | None = None  # 몸 정면 방향 (수평, ARKit 월드)
    facing: float = 0.0  # 1 = 카메라를 정면으로 봄
    hands_up: bool = False


@dataclass
class _Accepted:
    t: float
    pos: np.ndarray
    rot: np.ndarray


def _horizontal(v: np.ndarray) -> np.ndarray | None:
    h = np.array([v[0], 0.0, v[2]])
    n = float(np.linalg.norm(h))
    return h / n if n > 1e-6 else None


class Processor:
    CAMERA_JUMP_POS = 0.05  # m / frame
    CAMERA_JUMP_ROT = math.radians(3.0)  # rad / frame
    MAX_PREDICTION = 0.15  # m

    def __init__(self, settings: Settings):
        self.settings = settings
        self.reset()

    def reset(self) -> None:
        self._cam_ref: Rigid | None = None
        self._cam_prev: Rigid | None = None
        self._sign_score = 0.0
        self._reset_filters()

    def reset_camera(self) -> None:
        self._cam_ref = None
        self._cam_prev = None
        self._reset_filters()

    def _reset_filters(self) -> None:
        self._pos_f: dict[str, OneEuroFilter] = {}
        self._rot_f: dict[str, QuatOneEuroFilter] = {}
        self._accepted: dict[str, _Accepted] = {}

    def apply_filter_settings(self) -> None:
        s = self.settings
        for f in self._pos_f.values():
            f.min_cutoff, f.beta = s.min_cutoff, s.beta
        for f in self._rot_f.values():
            f.min_cutoff, f.beta = s.rot_min_cutoff, s.rot_beta

    # ------------------------------------------------------------------

    def process(self, frame: BodyFrame) -> FrameOut:
        s = self.settings
        events: list[str] = []

        cam = Rigid(quat_normalize(frame.camera.rot), np.asarray(frame.camera.pos, float))
        if self._cam_prev is not None:
            delta = self._cam_prev.inverse() @ cam
            if (np.linalg.norm(delta.pos) > self.CAMERA_JUMP_POS
                    or quat_angle(delta.rot) > self.CAMERA_JUMP_ROT):
                # 드리프트가 아니라 폰이 실제로 움직였다: 새 위치를 기준으로 삼는다
                self._cam_ref = None
                self._reset_filters()
                events.append("camera_moved")
        if self._cam_ref is None:
            self._cam_ref = cam
        self._cam_prev = cam

        fix = self._cam_ref @ cam.inverse() if s.stabilize_camera else Rigid.identity()
        cam_world = fix @ cam
        out = FrameOut(tracked=False, camera_pos=cam_world.pos, camera_rot=cam_world.rot, events=events)
        if not frame.tracked or not frame.joints:
            return out

        world: dict[str, tuple[np.ndarray, np.ndarray]] = {
            name: (fix.apply(p.pos), quat_normalize(quat_mul(fix.rot, p.rot)))
            for name, p in frame.joints.items()
        }
        out.tracked = True
        out.skeleton = {name: v[0] for name, v in world.items()}

        self._body_direction(world, out)

        t = frame.timestamp
        enabled = set(s.enabled_trackers)
        for index, name, joint, _label in TRACKERS:
            if name not in enabled or joint not in world:
                continue
            pos, rot = self._smooth(joint, *world[joint], t)
            out.trackers.append(TrackerOut(index, name, pos, rot))

        if HEAD_JOINT in world:
            out.head_pos, _ = self._smooth(HEAD_JOINT, *world[HEAD_JOINT], t)
            lh = world.get("left_hand_joint")
            rh = world.get("right_hand_joint")
            if lh is not None and rh is not None:
                head_y = world[HEAD_JOINT][0][1]
                out.hands_up = lh[0][1] > head_y + 0.10 and rh[0][1] > head_y + 0.10
        return out

    def _body_direction(self, world, out: FrameOut) -> None:
        """좌우 어깨/골반으로 몸 정면 방향을 구한다 (관절 축 규약과 무관)."""
        rights = []
        for l, r in (("left_arm_joint", "right_arm_joint"), ("left_upLeg_joint", "right_upLeg_joint")):
            if l in world and r in world:
                rights.append(world[r][0] - world[l][0])
        if not rights:
            return
        right = _horizontal(np.mean(rights, axis=0))
        if right is None:
            return
        forward = np.cross(UP, right)

        hips = world.get("hips_joint", world.get("spine_4_joint"))
        to_cam = _horizontal(out.camera_pos - hips[0]) if hips is not None else None
        if to_cam is not None:
            d = float(np.dot(forward, to_cam))
            # ARKit 좌/우 규약이 예상과 반대여도 맞도록, 카메라를 향하는 쪽을 정면으로 학습한다
            if abs(d) > 0.5:
                self._sign_score = max(-50.0, min(50.0, self._sign_score + d))
        sign = -1.0 if self._sign_score < 0 else 1.0
        out.forward = forward * sign
        if to_cam is not None:
            out.facing = float(np.dot(out.forward, to_cam))

    def _smooth(self, name: str, pos: np.ndarray, rot: np.ndarray, t: float):
        s = self.settings
        acc = self._accepted.get(name)
        if acc is not None:
            dt = t - acc.t
            if 0 < dt < 0.3 and np.linalg.norm(pos - acc.pos) / dt > s.max_speed:
                pos, rot = acc.pos, acc.rot  # 튐: 직전 값 유지
            else:
                self._accepted[name] = _Accepted(t, pos, rot)
        else:
            self._accepted[name] = _Accepted(t, pos, rot)

        pf = self._pos_f.get(name)
        if pf is None:
            pf = self._pos_f[name] = OneEuroFilter(s.min_cutoff, s.beta)
        rf = self._rot_f.get(name)
        if rf is None:
            rf = self._rot_f[name] = QuatOneEuroFilter(s.rot_min_cutoff, s.rot_beta)
        out_pos = pf(pos, t)
        if s.prediction_ms > 0:
            v = pf.velocity
            if v is not None:
                lead = v * (s.prediction_ms / 1000.0)
                n = float(np.linalg.norm(lead))
                if n > self.MAX_PREDICTION:
                    lead *= self.MAX_PREDICTION / n
                out_pos = out_pos + lead
        return out_pos, rf(rot, t)


def unity_yaw_from_forward(forward: np.ndarray, flip: bool = False) -> float:
    """ARKit 월드의 정면 방향 -> Unity yaw(도). Unity 정면은 +Z."""
    fx, fz = float(forward[0]), -float(forward[2])  # Unity로 변환 (z 반전)
    yaw = math.degrees(math.atan2(fx, fz))
    if flip:
        yaw += 180.0
    return (yaw + 180.0) % 360.0 - 180.0
