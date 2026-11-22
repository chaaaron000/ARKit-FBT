"""ARKit 관절 데이터 -> VRChat 트래커 자세 변환 파이프라인.

1. 카메라 고정 보정: 폰은 삼각대에 고정되어 있다고 보고, ARKit 월드 좌표의
   미끄러짐(VIO 드리프트)을 카메라 자세로 되돌린다.
2. 자세 기울기 보정: 카메라 한 대로 추정한 뼈대는 카메라 쪽으로 통째로 기울어지는
   경향이 있다 (실측 약 12°). 정면을 보고 선 순간 기울기를 재서 이후 계속 바로 세운다.
3. 골반 기준 상대 위치 필터링: VRChat은 머리 위치로 공간을 맞추므로 상대 위치만 중요하다.
   몸 전체가 앞뒤로 출렁이는 잡음은 상대 위치에서 사라진다.
   - 순간 튐 제거: 최근 N프레임 중앙값
   - One Euro 필터, 카메라 깊이 방향만 더 강하게
4. 발/무릎 회전: ARKit 발 각도는 믿기 어려워 (발끝이 아래를 향함) 몸 방향(yaw)만 따르게 한다.
5. 몸 방향(yaw) 계산: VRChat 방향 정렬에 쓴다.
"""

from __future__ import annotations

import dataclasses
import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from .filters import AnisoFilter, QuatOneEuroFilter
from .mathutil import IDENTITY_Q, Rigid, quat_angle, quat_from_axis_angle, quat_mul, quat_normalize, quat_rotate
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
ANCHOR_JOINT = "hips_joint"
# 회전을 몸 방향(yaw)만으로 만드는 트래커
FLAT_TRACKERS = {"left_foot", "right_foot", "left_knee", "right_knee"}

UP = np.array([0.0, 1.0, 0.0])
MAX_LEAN_DEG = 35.0


@dataclass
class Settings:
    enabled_trackers: list[str] = field(default_factory=lambda: ["hip", "left_foot", "right_foot"])
    # 위치 필터 (좌우·상하). 2026-11-22 iPhone XR 실측 녹화로 고른 값
    min_cutoff: float = 3.0
    beta: float = 3.0
    # 카메라 깊이 방향은 이 비율만큼 더 강하게 거른다 (1 = 같게)
    depth_ratio: float = 0.2
    # 회전 필터
    rot_min_cutoff: float = 1.0
    rot_beta: float = 0.3
    # 순간 튐 제거: 최근 N프레임 중앙값 (1 = 끔, 3이면 60fps에서 지연 약 17ms)
    spike_window: int = 3
    # 몸 전체가 이보다 빠르게 순간 이동하면 오인식으로 보고 잠깐 무시 (m/s)
    max_speed: float = 20.0
    # 골반 기준으로 관절이 움직일 수 있는 최대 속도 (m/s). 넘는 만큼은 잘라서 튐을 누른다.
    max_rel_speed: float = 3.0
    # 지연을 줄이기 위해 속도 방향으로 미리 내다보는 시간 (ms). 크면 동작이 튀어나간다.
    prediction_ms: float = 0.0
    upright_correction: bool = True
    # 몸 기울기 추정의 시간 상수 (초). 이보다 오래 지속되는 기울기는 오차로 보고 바로 세운다.
    upright_tau: float = 8.0
    flat_feet: bool = True
    # VRChat에 보낼 머리 위치의 수평 성분: "chest"(가슴, 추천) / "neck"(목) / "head"(ARKit 머리)
    # 헤드셋을 쓰면 ARKit 머리가 앞으로 쏠려 몸 전체가 밀려 보인다 (실측 골반 기준 17cm).
    head_horizontal: str = "chest"
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


def _horizontal(v: np.ndarray) -> np.ndarray | None:
    h = np.array([v[0], 0.0, v[2]])
    n = float(np.linalg.norm(h))
    return h / n if n > 1e-6 else None


class _Despike:
    """최근 N개 값의 성분별 중앙값."""

    def __init__(self, n: int):
        self.buf: deque[np.ndarray] = deque(maxlen=max(1, n))

    def __call__(self, v: np.ndarray, n: int) -> np.ndarray:
        n = max(1, int(n))
        if self.buf.maxlen != n:
            self.buf = deque(self.buf, maxlen=n)
        self.buf.append(v)
        if n == 1 or len(self.buf) < n:
            return v
        return np.median(np.array(self.buf), axis=0)


class Processor:
    CAMERA_JUMP_POS = 0.05  # m / frame
    CAMERA_JUMP_ROT = math.radians(3.0)  # rad / frame
    MAX_PREDICTION = 0.15  # m

    def __init__(self, settings: Settings):
        self.settings = settings
        self.lean_deg: float | None = None
        self._lean: np.ndarray | None = None
        self.clear_upright()
        self.reset()

    def reset(self) -> None:
        self._cam_ref: Rigid | None = None
        self._cam_prev: Rigid | None = None
        self._sign_score = 0.0
        self._last_world: dict[str, tuple[np.ndarray, np.ndarray]] | None = None
        self._reset_filters()

    def reset_camera(self) -> None:
        self._cam_ref = None
        self._cam_prev = None
        self.clear_upright()
        self._reset_filters()

    def clear_upright(self) -> None:
        self._lean = None
        self.lean_deg = None
        self._axis_avg: np.ndarray | None = None
        self._axis_t: float | None = None

    def _reset_filters(self) -> None:
        self._anchor_f = AnisoFilter()
        self._anchor_last: tuple[float, np.ndarray] | None = None
        self._rel_f: dict[str, AnisoFilter] = {}
        self._despike: dict[str, _Despike] = {}
        self._rot_f: dict[str, QuatOneEuroFilter] = {}
        self._yaw_f = QuatOneEuroFilter()
        # 추적이 끊긴 관절은 마지막 정상 값을 유지한다
        self._rel_out: dict[str, np.ndarray] = {}
        self._rot_out: dict[str, np.ndarray] = {}
        self._slew: dict[str, tuple[float, np.ndarray]] = {}
        self._last_out: FrameOut | None = None
        self._last_forward: np.ndarray | None = None
        self._axis_hips: np.ndarray | None = None
        self._axis_hips_t: float | None = None

    def apply_filter_settings(self) -> None:
        """설정은 매 프레임 필터에 반영되므로 따로 할 일이 없다 (호환용)."""

    # ------------------------------------------------------------------

    @staticmethod
    def _body_axis(w) -> np.ndarray | None:
        """두 발 가운데 → 목·가슴 방향 (단위 벡터). 서 있을 때만 의미가 있다."""
        need = ("spine_7_joint", "neck_1_joint", "left_foot_joint", "right_foot_joint")
        if w is None or any(n not in w for n in need):
            return None
        top = (w["spine_7_joint"][0] + w["neck_1_joint"][0]) / 2
        bottom = (w["left_foot_joint"][0] + w["right_foot_joint"][0]) / 2
        v = top - bottom
        n = float(np.linalg.norm(v))
        if n < 0.9:  # 앉거나 웅크림: 기울기 추정에 쓰지 않는다
            return None
        v = v / n
        if math.degrees(math.acos(max(-1.0, min(1.0, float(v[1]))))) > MAX_LEAN_DEG:
            return None
        return v

    def _set_lean_from_axis(self, v: np.ndarray) -> None:
        angle = math.acos(max(-1.0, min(1.0, float(np.dot(v, UP)))))
        axis = np.cross(v, UP)
        an = float(np.linalg.norm(axis))
        self._lean = IDENTITY_Q.copy() if an < 1e-6 else quat_from_axis_angle(axis / an, angle)
        self.lean_deg = math.degrees(angle)

    def calibrate_upright(self) -> float | None:
        """정면을 보고 똑바로 선 순간에 호출: 지금 기울기로 즉시 보정한다 (이후에도 천천히 계속 추적).

        반환: 보정 각도(도). 전신이 추적되지 않으면 None.
        """
        v = self._body_axis(self._last_world)
        if v is None:
            return None
        self._axis_avg = v
        self._set_lean_from_axis(v)
        self._rel_f.clear()
        self._despike.clear()
        self._slew.clear()
        return self.lean_deg

    def _track_upright(self, t: float) -> None:
        """몸 기울기를 시간 상수 upright_tau 로 천천히 따라간다 (카메라 위치·거리에 따라 바뀌므로)."""
        v = self._body_axis(self._last_world)
        if v is None:
            return
        if self._axis_avg is None or self._axis_t is None:
            self._axis_avg = v
        else:
            dt = max(0.0, min(0.5, t - self._axis_t))
            a = 1.0 - math.exp(-dt / max(0.1, self.settings.upright_tau))
            avg = self._axis_avg + a * (v - self._axis_avg)
            self._axis_avg = avg / float(np.linalg.norm(avg))
        self._axis_t = t
        self._set_lean_from_axis(self._axis_avg)

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
                self.clear_upright()
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

        # ARKit이 '추적 중'으로 표시한 관절만 믿는다 (나머지 추정값은 실측상 크게 흔들림).
        good = frame.joints.keys() if frame.tracked_joints is None else frame.tracked_joints
        world: dict[str, tuple[np.ndarray, np.ndarray]] = {
            name: (fix.apply(p.pos), quat_normalize(quat_mul(fix.rot, p.rot)))
            for name, p in frame.joints.items() if name in good
        }
        if ANCHOR_JOINT not in world:
            # 골반이 추적되지 않음: 직전 출력을 그대로 유지한다
            if self._last_out is not None:
                held = dataclasses.replace(self._last_out, events=events, camera_pos=out.camera_pos,
                                           camera_rot=out.camera_rot, hands_up=False)
                return held
            return out
        if all(n in world for n in ("spine_7_joint", "neck_1_joint", "left_foot_joint", "right_foot_joint")):
            self._last_world = world
            if s.upright_correction:
                self._track_upright(frame.timestamp)

        if s.upright_correction and self._lean is not None:
            pivot = world[ANCHOR_JOINT][0]
            lean = self._lean
            world = {n: (pivot + quat_rotate(lean, p - pivot), quat_normalize(quat_mul(lean, q)))
                     for n, (p, q) in world.items()}

        out.tracked = True
        out.skeleton = {name: v[0] for name, v in world.items()}
        self._body_direction(world, out)

        t = frame.timestamp
        hips = world[ANCHOR_JOINT][0]
        # 깊이 축은 천천히 따라가는 골반 위치로 정한다. 흔들리는 골반으로 축을 잡으면
        # 축 자체가 흔들려 먼 관절에 가짜 잡음이 생긴다 (2.5m에서 8cm 출렁임 ≈ 축 2°).
        if self._axis_hips is None or self._axis_hips_t is None:
            self._axis_hips = hips.copy()
        else:
            dt = max(0.0, min(0.5, t - self._axis_hips_t))
            self._axis_hips = self._axis_hips + (1.0 - math.exp(-dt / 1.0)) * (hips - self._axis_hips)
        self._axis_hips_t = t
        depth = self._axis_hips - cam_world.pos
        dn = float(np.linalg.norm(depth))
        depth = depth / dn if dn > 1e-6 else np.array([0.0, 0.0, -1.0])
        pred = s.prediction_ms / 1000.0
        anchor = self._filter_anchor(hips, t, depth, pred)

        cache: dict[str, np.ndarray | None] = {}

        def joint_pos(name: str) -> np.ndarray | None:
            if name not in cache:
                cache[name] = _joint_pos(name)
            return cache[name]

        def _joint_pos(name: str) -> np.ndarray | None:
            if name == ANCHOR_JOINT:
                return anchor
            if name not in world:
                held = self._rel_out.get(name)  # 추적이 끊김: 마지막 정상 상대 위치 유지
                return None if held is None else anchor + held
            rel = world[name][0] - hips
            ds = self._despike.get(name)
            if ds is None:
                ds = self._despike[name] = _Despike(s.spike_window)
            rel = ds(rel, s.spike_window)
            rel = self._slew_limit(name, rel, t)
            f = self._rel_f.get(name)
            if f is None:
                f = self._rel_f[name] = AnisoFilter()
            r = f(rel, t, depth, s.min_cutoff, s.beta, s.depth_ratio, pred, self.MAX_PREDICTION)
            self._rel_out[name] = r
            return anchor + r

        if out.forward is not None:
            self._last_forward = out.forward
        else:
            out.forward = self._last_forward
        yaw_q = None
        if out.forward is not None:
            self._yaw_f.min_cutoff, self._yaw_f.beta = s.rot_min_cutoff, s.rot_beta
            raw = quat_from_axis_angle(UP, math.atan2(out.forward[0], out.forward[2]))
            yaw_q = self._yaw_f(raw, t)

        enabled = set(s.enabled_trackers)
        for index, name, joint, _label in TRACKERS:
            if name not in enabled:
                continue
            pos = joint_pos(joint)
            if pos is None:
                continue
            if s.flat_feet and name in FLAT_TRACKERS and yaw_q is not None:
                rot = yaw_q
            elif joint in world:
                rf = self._rot_f.get(joint)
                if rf is None:
                    rf = self._rot_f[joint] = QuatOneEuroFilter()
                rf.min_cutoff, rf.beta = s.rot_min_cutoff, s.rot_beta
                rot = self._rot_out[joint] = rf(world[joint][1], t)
            elif joint in self._rot_out:
                rot = self._rot_out[joint]
            else:
                continue
            out.trackers.append(TrackerOut(index, name, pos, rot))

        out.head_pos = joint_pos(HEAD_JOINT)
        horiz_src = {"chest": "spine_7_joint", "neck": "neck_1_joint"}.get(s.head_horizontal)
        if out.head_pos is not None and horiz_src is not None:
            src = joint_pos(horiz_src)
            if src is not None:
                out.head_pos = np.array([src[0], out.head_pos[1], src[2]])
        lh, rh, hd = world.get("left_hand_joint"), world.get("right_hand_joint"), world.get(HEAD_JOINT)
        if lh is not None and rh is not None and hd is not None:
            out.hands_up = lh[0][1] > hd[0][1] + 0.10 and rh[0][1] > hd[0][1] + 0.10
        self._last_out = out
        return out

    def _slew_limit(self, name: str, rel: np.ndarray, t: float) -> np.ndarray:
        """골반 기준 상대 위치가 물리적으로 불가능하게 빨리 변하면 그만큼 잘라낸다."""
        s = self.settings
        prev = self._slew.get(name)
        if prev is not None and s.max_rel_speed > 0:
            pt, pr = prev
            dt = t - pt
            if 0 < dt < 0.5:
                d = rel - pr
                n = float(np.linalg.norm(d))
                lim = s.max_rel_speed * dt
                if n > lim:
                    rel = pr + d * (lim / n)
        self._slew[name] = (t, rel)
        return rel

    def _filter_anchor(self, hips: np.ndarray, t: float, depth: np.ndarray, pred: float) -> np.ndarray:
        """골반 절대 위치. 몸 전체가 순간 이동하는 오인식은 잠깐 무시한다."""
        s = self.settings
        if self._anchor_last is not None:
            lt, lp = self._anchor_last
            dt = t - lt
            if 0 < dt < 0.3 and np.linalg.norm(hips - lp) / dt > s.max_speed:
                hips = lp
            else:
                self._anchor_last = (t, hips)
        else:
            self._anchor_last = (t, hips)
        return self._anchor_f(hips, t, depth, s.min_cutoff, s.beta, s.depth_ratio, pred, self.MAX_PREDICTION)

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

        hips = world.get(ANCHOR_JOINT, world.get("spine_4_joint"))
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


def unity_yaw_from_forward(forward: np.ndarray, flip: bool = False) -> float:
    """ARKit 월드의 정면 방향 -> Unity yaw(도). Unity 정면은 +Z."""
    fx, fz = float(forward[0]), -float(forward[2])  # Unity로 변환 (z 반전)
    yaw = math.degrees(math.atan2(fx, fz))
    if flip:
        yaw += 180.0
    return (yaw + 180.0) % 360.0 - 180.0
