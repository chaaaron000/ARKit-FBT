"""ARKit 관절 데이터 -> VRChat 트래커 자세 변환 파이프라인.

1. 카메라 고정 보정: 폰은 삼각대에 고정되어 있다고 보고, ARKit 월드 좌표의
   미끄러짐(VIO 드리프트, 한 프레임 원점 이동)을 카메라 자세로 되돌린다. 폰이 실제로 움직이면(몸이 카메라에서
   본 위치로는 움직였는데 ARKit 월드에서는 그대로: 한 프레임에 크게, 또는 사람은 가만히 있는데 폰만 1초 사이
   10cm 넘게 움직임) 새 위치를 기준으로 다시 시작한다. 드리프트를 폰 이동으로 오인하지 않는 것이 우선이라,
   그보다 느리거나 애매한 이동은 '카메라 기준 재설정'에 맡긴다.
2. 자세 기울기 보정: 카메라 한 대로 추정한 뼈대는 카메라 쪽으로 통째로 기울어지는
   경향이 있다 (실측 약 12°). 정면을 보고 선 순간 기울기를 재서 이후 계속 바로 세운다.
3. 골반 기준 상대 위치 필터링: VRChat은 머리 위치로 공간을 맞추므로 상대 위치만 중요하다.
   몸 전체가 앞뒤로 출렁이는 잡음은 상대 위치에서 사라진다.
   - 순간 튐 제거: 최근 N프레임 중앙값
   - One Euro 필터, 카메라 깊이 방향만 더 강하게
4. 발/무릎 회전: ARKit 발 각도는 믿기 어려워 (발끝이 아래를 향함) 몸 방향(yaw)만 따르게 한다.
5. 몸 방향(yaw) 계산: VRChat 방향 정렬에 쓴다.
6. 발 고정: 발이 바닥 높이에 있고 거의 안 움직이면 디딘 것으로 보고 그 자리(출력 좌표)에 고정한다
   (PhysCap의 접지 제약을 발 높이·속도 규칙으로 단순화). 몸이 움직여 앵커가 늦는 만큼은 고정 위치를
   옮겨 (발 - 머리)가 틀어지지 않게 한다. 들면 풀고 짧게 섞어 따라간다. 고정 중 동작은 두 가지:
   "motion"은 작은 여유 안의 떨림과 바닥에서의 빠른 튐만 무시하고 그 밖으로는 따라가며 끌어 옮기면 푼다,
   "stable"은 들지 않은 20cm 안의 미끄러짐을 무시한다.
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
# 품질 게이트를 적용하는 관절 (트래커 관절 + 머리 위치 계산 + 몸 방향 계산에 쓰는 것)
GATED_JOINTS = tuple(sorted({j for _, _, j, _ in TRACKERS if j != ANCHOR_JOINT}
                            | {HEAD_JOINT, "neck_1_joint", "spine_7_joint", "left_upLeg_joint", "right_upLeg_joint"}))

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
    # 품질 게이트: 최근 gate_window 초 동안 추적 비율이 낮거나 원본이 심하게 흔들리면
    # 그 관절은 마지막 안정 위치로 고정한다 (ARKit이 몸을 못 보는 구간에서 출렁임 방지).
    quality_gate: bool = True
    gate_window: float = 0.5
    gate_min_tracked: float = 0.8
    gate_max_jitter: float = 0.025  # 원본 상대 위치의 프레임간 변화 중앙값 (m)
    # 지연을 줄이기 위해 속도 방향으로 미리 내다보는 시간 (ms). 크면 동작이 튀어나간다.
    prediction_ms: float = 0.0
    upright_correction: bool = True
    # 몸 기울기 추정의 시간 상수 (초). 이보다 오래 지속되는 기울기는 오차로 보고 바로 세운다.
    upright_tau: float = 8.0
    flat_feet: bool = True
    # 골반·발 방향을 '카메라를 바라보는 방향'으로 고정한다. ARKit 몸 방향 추정은 정면을 봐도
    # 70° 이상 틀어지는 일이 잦고(실측), 몸을 많이 돌리면 어차피 추적을 놓친다.
    lock_yaw_to_camera: bool = True
    # 발 고정: 발이 바닥 높이에 있고 거의 안 움직이면 그 자리에 고정한다. 발을 들면(높이 +2cm, 또는 조금 들고
    # 빨리 옮김) 풀어서 짧게 섞어 따라간다. foot_lock_mode:
    # - "motion"(움직임 우선): 고정된 발은 여유(수평 foot_lock_slack, 위로 Processor.LOCK_SLACK_Y) 안의 떨림만
    #   무시하고 그 밖으로는 따라간다. 0.15초 넘게 끌어 옮기면 풀린다 (발 끌기·작은 걸음이 보임).
    # - "stable"(정지 안정 우선): 수평 20cm 안의 미끄러짐과 들지 않는 작은 움직임은 무시한다 (서 있을 때 더 단단함).
    foot_lock: bool = True
    foot_lock_mode: str = "motion"
    foot_lock_height: float = 0.04  # 바닥(발 관절이 쉬는 높이) 위 이 안이면 디딤 후보 (m)
    foot_lock_speed: float = 0.25  # 최근 0.2초 수평 이동 속도가 이보다 느리면 디딤 후보 (m/s)
    foot_lock_slack: float = 0.05  # "motion": 고정된 발이 수평으로 이 안의 떨림은 무시한다 (m)
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


FOOT_JOINTS = {"left_foot": "left_foot_joint", "right_foot": "right_foot_joint"}
# Settings.foot_lock_mode 값과 GUI 이름
FOOT_LOCK_MODES = {"motion": "움직임 우선", "stable": "정지 안정 우선"}


def foot_lock_status(settings: Settings, foot_locked, floor_ready: bool) -> str:
    """발 고정 상태 한 줄 (예: '왼발 고정 · 오른발 따라감')."""
    if not settings.foot_lock:
        return "꺼짐"
    labels = {n: label for _i, n, _j, label in TRACKERS if n in FOOT_JOINTS and n in settings.enabled_trackers}
    if not labels:
        return "발 트래커 없음"
    if not floor_ready:
        return "바닥 높이 재는 중"
    return " · ".join(f"{label} {'고정' if n in foot_locked else '따라감'}" for n, label in labels.items())


@dataclass
class _FootLock:
    """발 하나의 고정 상태. 위치는 출력 좌표 (앵커 + 상대 위치)."""

    locked: bool = False
    pos: np.ndarray | None = None  # 고정 위치 (앵커 지연 보정 전, Processor._lock_target 참고)
    base_y: float = 0.0  # 고정할 때 높이: 여유로 따라 올라갔다가 이 아래로는 돌아오지 않는다
    since: float | None = None  # 디딤 조건이 이어지기 시작한 시각
    samples: list[np.ndarray] = field(default_factory=list)  # 그동안의 후보 위치 (평균 = 고정 위치)
    release_since: float | None = None  # 들기 조건이 이어지기 시작한 시각
    taut_since: float | None = None  # 고정 위치가 여유 밖으로 끌려가기 시작한 시각
    taut_jump: bool = False  # 끌림이 무시한 튐 때문인가 (풀 때 빨리 섞는다)
    sink_since: float | None = None  # 발이 고정 높이보다 LOCK_SINK 넘게 낮아지기 시작한 시각
    resid: np.ndarray | None = None  # 고정 중 마지막 수평 차이 (튐을 뺀 것): 다른 발이 함께 밀렸나 볼 때
    resid_t: float | None = None
    jump: np.ndarray = field(default_factory=lambda: np.zeros(3))  # 고정 중 무시한 빠른 수평 튐의 합
    jump_t: float | None = None  # 마지막으로 튐을 더한 시각
    last: np.ndarray | None = None  # 직전 프레임의 (c - 앵커 지연 보정)
    release_pending: bool = False  # 기능을 끄거나 모드를 바꿔 풀었다: 다음 처리 프레임에 섞기 시작
    recent: deque[tuple[float, np.ndarray]] = field(default_factory=deque)  # 최근 (시각, 후보 위치): 속도용
    speed: float = 0.0  # 최근 SPEED_WINDOW 초의 수평 이동 속도 (m/s)
    vspeed: float = 0.0  # 같은 시간의 수직 이동 속도 (m/s)
    height: float | None = None  # 이번 프레임 원본 발 높이 (바닥 기준, 모르면 None)
    offset: np.ndarray | None = None  # 고정·풀림 순간의 (직전 출력 - 새 목표), 섞는 동안 0으로 줄인다
    blend: float = 1.0  # 섞기 진행 (0 → 1)
    blend_sec: float = 0.0  # 이번 섞기 시간 (초)
    out: np.ndarray | None = None  # 직전 출력


def _agree(u: np.ndarray, v: np.ndarray, basis, dead=(0.0, 0.0, 0.0)) -> np.ndarray:
    """basis 축마다 u, v 가 같은 쪽을 가리키면 크기가 작은 쪽을, 아니면 0을 모은 벡터 (둘 다 보일 때만
    믿는 움직임). dead: 축마다 이만큼은 잡음으로 버린다."""
    out = np.zeros(3)
    for e, d in zip(basis, dead):
        a, b = float(np.dot(u, e)), float(np.dot(v, e))
        m = min(abs(a), abs(b)) - d
        if a * b > 0.0 and m > 0.0:
            out += math.copysign(m, a) * e
    return out


def _deadzone(v: np.ndarray, basis, dead) -> np.ndarray:
    """basis 축마다 v 성분을 dead 만큼 0 쪽으로 줄인다."""
    out = np.zeros(3)
    for e, d in zip(basis, dead):
        a = float(np.dot(v, e))
        if abs(a) > d:
            out += math.copysign(abs(a) - d, a) * e
    return out


def _clip(v: np.ndarray, n: float) -> np.ndarray:
    """v 의 길이를 n 이하로 자른다."""
    m = float(np.linalg.norm(v))
    return v * (n / m) if m > n else v


class Processor:
    # 폰이 실제로 움직였다는 판정 (Processor._phone_moved, _camera_relocated). ARKit 월드 좌표는 카메라와 몸이
    # 같이 옮겨지기도 하므로(드리프트, 한 프레임 원점 이동) 카메라 이동만으로는 판단하지 않는다: 몸(골반·양발)이
    # 카메라에서 본 위치로는 움직였는데 ARKit 월드에서는 그대로였을 때만 폰이 움직였다고 본다.
    CAMERA_JUMP_POS = 0.05  # m / frame
    CAMERA_JUMP_ROT = math.radians(3.0)  # rad / frame
    # 천천히 옮긴 카메라: 최근 CAMERA_MOVE_SEC 초 사이에 카메라가 이만큼 움직였거나 돌았고, 카메라 기준 골반도
    # 이만큼 바뀌었는데, ARKit 월드의 골반·양발은 모두 CAMERA_MOVE_BODY_STILL 안에 머물렀다
    CAMERA_MOVE_SEC = 1.0
    CAMERA_MOVE_POS = 0.10  # m
    CAMERA_MOVE_ROT = math.radians(5.0)
    CAMERA_MOVE_BODY_STILL = 0.03  # m
    CAMERA_BODY_JOINTS = (ANCHOR_JOINT, *FOOT_JOINTS.values())
    MAX_PREDICTION = 0.15  # m
    REENTRY_SEC = 0.4  # 고정이 풀린 뒤 천천히 옮겨가는 시간
    REENTRY_SPEED = 0.6  # 그동안의 최대 상대 속도 (m/s)
    # 발 고정
    FLOOR_WINDOW = 20.0  # 바닥 높이 추정에 쓰는 최근 시간 (초)
    FLOOR_PERCENTILE = 10.0  # 추적된 발 높이의 이 백분위 = 발 관절이 바닥에서 쉬는 높이
    FLOOR_MIN_SAMPLES = 60  # 이만큼 모이기 전에는 고정하지 않는다 (두 발 합쳐 약 0.5초)
    FLOOR_SETTLE = 3.0  # 재기 시작하고 이 시간 뒤로는 바닥을 FLOOR_RISE 넘게 올리지 않는다 (초)
    FLOOR_RISE = 0.03  # 그동안 잰 가장 낮은 바닥보다 이만큼까지만 올라갈 수 있다 (자세 보정 변화 등, m)
    # 골반이 발보다 높은 정도가 디딘 채 서 있을 때보다 이만큼 넘게 작으면 (의자에 앉음 약 30cm 이상, 쪼그림) 그
    # 발 높이로 바닥을 올리지 않는다 (m). 녹화 6개: 0.10이면 서 있는 표본까지 빠져 정지 표준편차 합이 0.1~0.2cm
    # 늘고, 0.15부터 수정 2회차와 같다.
    FLOOR_STAND_DROP = 0.15
    LOCK_DWELL = 0.2  # 디딤 조건이 이만큼 이어져야 고정 (초)
    RELEASE_CONFIRM = 0.03  # 들기 조건이 이만큼(3프레임) 이어져야 풂 (한 프레임 튐 무시)
    RELEASE_HEIGHT_EXTRA = 0.02  # 풀림 높이 = 디딤 높이 + 이 값
    RELEASE_SPEED_RATIO = 2.0  # 디딤 높이를 넘었고 수평 속도가 디딤 속도의 이 배보다 빠르면 풂 (걷기)
    LOCK_MAX_VSPEED = 0.05  # 최근 0.2초 수직 이동 속도가 이보다 느려야 디딤 후보 (내려오는 중에 고정하지 않게, m/s)
    LOCK_SLACK_Y = 0.005  # "motion": 고정된 발이 무시하는 위쪽 떨림 (m). 수평은 Settings.foot_lock_slack
    LOCK_FOLLOW_SPEED = 1.0  # 여유 밖으로 따라갈 때의 최대 속도 (m/s, 몇 프레임짜리 튐은 덜 옮긴다)
    DRAG_CONFIRM = 0.15  # 이만큼 계속 여유 밖으로 끌려가면 (발을 끌어 옮김) 풀고 그대로 따라간다 (초)
    LOCK_SINK = 0.025  # 발이 고정 높이보다 이만큼 넘게 낮은 채로
    LOCK_SINK_SEC = 1.5  # 이만큼 이어지면 풀고 새 높이에 다시 고정한다 (바닥이 내려갔다: 폰을 천천히 올림)
    LOCK_BLEND = 0.15  # 고정·풀림 때 섞는 시간 (초)
    LOCK_CREEP_TAU = 4.0  # 고정 중 고정 위치가 원본 쪽으로 천천히 옮겨가는 시간 상수 (초)
    BLEND_MAX_SPEED = 0.6  # 차이가 크면 섞는 시간을 늘려 이 평균 속도를 넘지 않게 한다 (m/s, 튐 방지)
    BLEND_DRAG_SPEED = 0.12  # 끌어 옮기다 풀 때의 섞기 최저 평균 속도 (발 속도의 2배까지 빨라진다, m/s)
    BLEND_BUDGET_SPEED = 1.5  # 발이 이 속도에 가까울수록 섞기를 늦춘다 (m/s)
    BLEND_SLOW_MIN = 0.25  # 그때도 섞기 진행은 이 비율보다 느려지지 않는다
    LOCK_MAX_LAG = 0.10  # 앵커가 몸보다 이만큼 넘게 뒤처져 있으면 새로 고정하지 않는다 (m)
    # 앵커 지연(보정·고정할 때)에서 이만큼은 골반 잡음으로 보고 버린다 (m, 깊이·위·좌우). 골반 추정이 흔들리면
    # 보정이 그 흔들림을 고정된 발에 옮긴다: 원본 골반 잡음은 카메라 깊이 방향이 가장 크다 (실측 프레임간 변화
    # 중앙값 깊이 0.4cm, 좌우 0.1cm, 위 0.05cm). 한 발로 디딘 동안(다른 발이 떠서 걷는 중)은 몸이 이 발 위로
    # 옮겨 가므로 좌우를 거의 다 보정하고, 두 발 다 디딘 동안(서 있기·쪼그리기)은 수평 흔들림을 더 버린다
    # (녹화 6개 정지 표준편차 합: 두 발 디딤 데드존 (2.5, 0.5, 1) → (4, 0.5, 4)cm 에서 motion 23.3 → 21.1cm,
    # stable 18.3 → 16.9cm. 수정 2회차 보고서 조정 기록).
    LOCK_LAG_DEAD = (0.025, 0.002, 0.01)
    LOCK_LAG_DEAD_BOTH = (0.04, 0.002, 0.04)
    LOCK_JUMP_SPEED = 0.6  # "motion": 고정된 발이 이보다 빨리 옆으로 가면 ARKit 튐으로 보고 무시한다 (m/s)
    LOCK_JUMP_KEEP = 0.05  # 무시한 튐이 이만큼 넘게 남은 채로
    LOCK_JUMP_SETTLE = 0.1  # 이만큼 더 튀지 않으면 (자리를 옮겼다) 끌려가는 중으로 본다 (초)
    LOCK_COMMON_MAX = 0.04  # "motion": 두 발이 함께 밀린 수평 차이는 이만큼까지 무시한다 (m)
    LOCK_JUMP_RADIUS = 0.20  # 무시한 튐까지 합쳐 이만큼 넘게 벗어나 있으면 끌려가는 중으로 본다 (m)
    SPEED_WINDOW = 0.2  # 디딤 판정 속도 = 이 시간 동안의 수평 순이동 / 시간 (프레임 튐에 둔감)
    GAP_RESET = 0.1  # 처리 프레임 사이가 이보다 벌어지면 디딤 판정을 처음부터 다시 한다 (초)

    def __init__(self, settings: Settings):
        self.settings = settings
        self.lean_deg: float | None = None
        self._lean: np.ndarray | None = None
        self._lock_enabled = settings.foot_lock
        self._lock_mode = settings.foot_lock_mode
        self.clear_upright()
        self.reset()

    def reset(self) -> None:
        self._cam_ref: Rigid | None = None
        self._cam_prev: Rigid | None = None
        # 최근 CAMERA_MOVE_SEC 초: (시각, 카메라, 추적된 골반·양발의 ARKit 월드 위치)
        self._cam_hist: deque[tuple[float, Rigid, dict[str, np.ndarray]]] = deque()
        self._sign_score = 0.0
        self._last_world: dict[str, tuple[np.ndarray, np.ndarray]] | None = None
        self._reset_filters()

    def reset_camera(self) -> None:
        self._cam_ref = None
        self._cam_prev = None
        self._cam_hist = deque()
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
        self._lag = np.zeros(3)  # 앵커 - 몸(One Euro로 거른 골반): 앵커가 몸을 늦게 따라오는 만큼
        self._lag_basis = (np.array([0.0, 0.0, 1.0]), UP, np.array([1.0, 0.0, 0.0]))  # 깊이·위·좌우 (수평 깊이)
        self._qhist: dict[str, deque[tuple[float, np.ndarray | None]]] = {}
        self._held_at: dict[str, float] = {}
        self.gated: set[str] = set()
        self._clear_foot_lock()

    def _clear_foot_lock(self) -> None:
        """발 고정과 바닥 높이 추정을 처음부터 다시 한다 (카메라 이동·재설정·자세 보정 때)."""
        self._feet: dict[str, _FootLock] = {}
        self._clear_floor()
        self.foot_locked: set[str] = set()

    def _clear_floor(self) -> None:
        # (시각, 발 높이, 서 있는 자세였나: 골반 - 발 높이가 디딘 채 섰을 때에 가깝다)
        self._floor_hist: deque[tuple[float, float, bool]] = deque()
        self.floor_y: float | None = None
        self._floor_t0: float | None = None  # 바닥을 재기 시작한 시각
        self._floor_low: float | None = None  # FLOOR_SETTLE 뒤로 잰 가장 낮은 바닥
        # 고정된 발 위로 골반이 있는 높이의 윗선 (서 있을 때). 앉거나 쪼그려도 내려가지 않는다
        self._stand_leg: float | None = None

    def apply_settings(self) -> None:
        """설정 변경을 바로 반영한다 (다음 패킷을 기다리지 않음. 처리할 때도 매번 확인한다): 끈 트래커의 고정
        기록을 버리고, 기능을 끄거나 켜면 바닥을 다시 재며 고정을 섞어서 푼다 (예전 고정 위치·바닥이 되살아나지
        않게). 모드를 바꾸면 고정만 섞어서 풀고 새 모드로 다시 고정한다 (점프 없이)."""
        s = self.settings
        for name in [n for n in self._feet if n not in s.enabled_trackers]:
            del self._feet[name]
        toggled = s.foot_lock != self._lock_enabled
        if toggled:
            self._lock_enabled = s.foot_lock
            self._clear_floor()
        if toggled or s.foot_lock_mode != self._lock_mode:
            self._lock_mode = s.foot_lock_mode
            for st in self._feet.values():
                st.since, st.samples, st.release_since, st.taut_since, st.sink_since = None, [], None, None, None
                if st.locked:
                    st.locked, st.release_pending = False, True
        self.foot_locked = {n for n, st in self._feet.items() if st.locked}

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
        self._clear_foot_lock()  # 기울기가 바뀌면 발 높이·위치도 바뀐다
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

    def _camera_body(self, frame: BodyFrame) -> dict[str, np.ndarray] | None:
        """이번 프레임에 ARKit이 '추적 중'으로 표시한 골반·양발의 ARKit 월드 위치 (골반이 없으면 None)."""
        if not frame.tracked or not frame.joints:
            return None
        good = frame.joints.keys() if frame.tracked_joints is None else frame.tracked_joints
        body = {n: np.asarray(frame.joints[n].pos, float)
                for n in self.CAMERA_BODY_JOINTS if n in frame.joints and n in good}
        return body if ANCHOR_JOINT in body else None

    def _phone_moved(self, cam: Rigid, t: float, body: dict[str, np.ndarray] | None) -> bool:
        """카메라가 한 프레임에 크게 바뀌었을 때: 폰이 움직였는가, ARKit 월드가 통째로 옮겨졌는가. 바로 앞
        (GAP_RESET 안) 프레임과 비교해 몸(골반·양발)이 ARKit 월드에서보다 카메라에서 본 위치로 더 많이
        움직였으면 폰이 움직였다. 월드가 통째로 옮겨지면 카메라에서 본 몸은 그대로다 (카메라 고정 보정이
        흡수한다). 몸을 못 봤으면 증거가 없으므로 아니라고 본다 (오검출 0 우선)."""
        if body is None or not self._cam_hist or t - self._cam_hist[-1][0] > self.GAP_RESET:
            return False
        _t0, cam0, body0 = self._cam_hist[-1]
        names = [n for n in body if n in body0]
        world = sum(float(np.linalg.norm(body[n] - body0[n])) for n in names)
        view = sum(float(np.linalg.norm(cam.inverse().apply(body[n]) - cam0.inverse().apply(body0[n])))
                   for n in names)
        return world < view

    def _camera_relocated(self, cam: Rigid, t: float, body: dict[str, np.ndarray] | None) -> bool:
        """프레임마다는 작아도 폰을 천천히 옮겼는가 (아주 보수적으로, 오검출이 없게): 최근 CAMERA_MOVE_SEC 초
        사이 카메라가 ARKit 월드에서 CAMERA_MOVE_POS 넘게 움직였거나 CAMERA_MOVE_ROT 넘게 돌았고, 카메라 기준
        골반 위치도 CAMERA_MOVE_POS 넘게 바뀌었는데, ARKit 월드의 골반과 양발은 모두 CAMERA_MOVE_BODY_STILL
        안에 머물렀다 (셋 다 처음과 지금 추적돼 있어야 한다).

        ARKit 월드가 미끄러지는 드리프트는 카메라와 몸이 같이 움직여 월드의 몸도 움직인다. 드리프트 중에
        사람도 움직이면 카메라 기준 골반이 바뀌고, 골반이 드리프트를 거슬러 움직여 월드에서 제자리처럼 보일
        수는 있어도 그때 바닥을 디딘 발은 드리프트를 따라 움직인다. 그래서 '사람은 가만히 있고 폰만 옮김'만
        잡는다: 사람이 같이 움직이거나 골반·발을 놓친 채로 옮기면 놓친다 (GUI '카메라 기준 재설정'). 실측 녹화
        6개에서 1초 사이 카메라 이동은 최대 1.4cm·0.05°."""
        if body is None:
            return False
        hist = self._cam_hist
        while hist and t - hist[0][0] > self.CAMERA_MOVE_SEC:
            hist.popleft()
        moved = False
        if hist:
            _t0, cam0, body0 = hist[0]
            h, h0 = body[ANCHOR_JOINT], body0[ANCHOR_JOINT]
            moved = ((np.linalg.norm(cam.pos - cam0.pos) > self.CAMERA_MOVE_POS
                      or quat_angle((cam0.inverse() @ cam).rot) > self.CAMERA_MOVE_ROT)
                     and np.linalg.norm(cam.inverse().apply(h) - cam0.inverse().apply(h0)) > self.CAMERA_MOVE_POS
                     and all(n in body and n in body0
                             and np.linalg.norm(body[n] - body0[n]) < self.CAMERA_MOVE_BODY_STILL
                             for n in self.CAMERA_BODY_JOINTS))
        hist.append((t, cam, body))
        return bool(moved)

    def process(self, frame: BodyFrame) -> FrameOut:
        s = self.settings
        events: list[str] = []
        self.apply_settings()

        cam = Rigid(quat_normalize(frame.camera.rot), np.asarray(frame.camera.pos, float))
        t = frame.timestamp
        body = self._camera_body(frame)
        jumped = False
        if self._cam_prev is not None:
            delta = self._cam_prev.inverse() @ cam
            jumped = ((np.linalg.norm(delta.pos) > self.CAMERA_JUMP_POS
                       or quat_angle(delta.rot) > self.CAMERA_JUMP_ROT)
                      and self._phone_moved(cam, t, body))
        if jumped or self._camera_relocated(cam, t, body):
            # 드리프트가 아니라 폰이 실제로 움직였다: 새 위치를 기준으로 삼는다
            self._cam_ref = None
            self._cam_hist = deque()
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
            # 골반이 추적되지 않음: 직전 출력을 그대로 유지한다 (그 뒤에 끈 트래커는 빼고)
            if self._last_out is not None:
                held = dataclasses.replace(self._last_out, events=events, camera_pos=out.camera_pos,
                                           camera_rot=out.camera_rot, hands_up=False,
                                           trackers=[x for x in self._last_out.trackers
                                                     if x.name in s.enabled_trackers])
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
        body = self._filter_anchor(hips, t, depth, pred)
        anchor = body
        if s.send_head_position:
            # VRChat은 머리 위치로 공간을 맞추므로 절대 위치는 의미가 없다. 머리·트래커 메시지가
            # 같은 프레임에 처리되지 않아도 어긋나지 않도록 기준점을 아주 느리게 움직인다.
            anchor = self._axis_hips.copy()
        self._lag = anchor - body
        flat = _horizontal(depth)
        if flat is not None:
            self._lag_basis = (flat, UP, np.cross(UP, flat))

        # 품질 게이트용 기록: 관절별 (시각, 골반 기준 원본 상대 위치 또는 None=추적 안 됨)
        self.gated = set()
        for name in GATED_JOINTS:
            h = self._qhist.get(name)
            if h is None:
                h = self._qhist[name] = deque(maxlen=90)
            h.append((t, world[name][0] - hips if name in world else None))
        if s.foot_lock:
            self._update_floor(world, t)

        cache: dict[str, np.ndarray | None] = {}

        def joint_pos(name: str) -> np.ndarray | None:
            if name not in cache:
                cache[name] = _joint_pos(name)
            return cache[name]

        def _joint_pos(name: str) -> np.ndarray | None:
            if name == ANCHOR_JOINT:
                return anchor
            if name not in world or not self._quality_ok(name, t):
                if name in world:
                    self.gated.add(name)
                self._held_at[name] = t
                held = self._rel_out.get(name)  # 추적이 끊김/불안정: 마지막 안정 상대 위치 유지
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

        forward_ok = all(self._quality_ok(n, t) for n in ("left_upLeg_joint", "right_upLeg_joint"))
        if out.forward is not None and forward_ok:
            self._last_forward = out.forward
        elif self._last_forward is not None:
            out.forward = self._last_forward
        if s.lock_yaw_to_camera:
            to_cam = _horizontal(cam_world.pos - self._axis_hips)
            if to_cam is not None:
                out.forward = to_cam  # 방향 정렬·발 방향 모두 '카메라 정면' 기준 (facing 은 ARKit 값 유지)
        yaw_q = None
        if out.forward is not None:
            self._yaw_f.min_cutoff, self._yaw_f.beta = s.rot_min_cutoff, s.rot_beta
            raw = quat_from_axis_angle(UP, math.atan2(out.forward[0], out.forward[2]))
            yaw_q = self._yaw_f(raw, t)

        enabled = set(s.enabled_trackers)
        yaw_only = FLAT_TRACKERS | ({"hip"} if s.lock_yaw_to_camera else set())
        for index, name, joint, _label in TRACKERS:
            if name not in enabled:
                continue
            pos = joint_pos(joint)
            if pos is None:
                continue
            if name in FOOT_JOINTS and (s.foot_lock or name in self._feet):
                # 높이는 ARKit이 '추적 중'으로 표시했으면 품질 게이트에 걸려도 쓴다: 게이트가 붙잡아 둔 발도
                # 바닥에 있으면 그 자리에 고정해, 게이트가 풀릴 때의 출렁임이 지나가지 않게 한다.
                # 기능을 끈 직후에는 풀림 섞기만 마저 한다.
                pos = self._foot_lock(name, pos, world[joint][0][1] if joint in world else None, t)
            if (s.flat_feet or s.lock_yaw_to_camera) and name in yaw_only and yaw_q is not None:
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
        self.foot_locked = {n for n, f in self._feet.items() if f.locked}

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

    def _update_floor(self, world, t: float) -> None:
        """바닥 높이 = 최근 FLOOR_WINDOW 초 동안 추적된 두 발 관절 높이의 낮은 백분위. 처음 재기 시작하고
        FLOOR_SETTLE 초가 지나면 그 뒤로는 그동안 잰 가장 낮은 바닥보다 올라가는 것은 서 있는 자세의 발 높이로만,
        FLOOR_RISE 까지만 따라간다.

        발 관절은 발목 근처라 실제 바닥보다 몇 cm 위에 있다. 여기서 구하는 값은 '발 관절이 바닥에서
        쉬는 높이'이고, 디딤 판정도 이 높이를 기준으로 하므로 발목 높이를 따로 빼지 않는다.
        발이 바닥보다 높게 머무는 것(앉아서 발을 띄움)은 바닥이 바뀐 증거가 아니므로 다시 잡지 않는다: 처음부터
        다시 재는 것은 카메라 기준 재설정·camera_moved·기능 켜고 끄기 때뿐이다 (camera_moved 직후 폰이 아직
        움직이는 동안의 값으로 굳지 않게 처음 FLOOR_SETTLE 초는 백분위를 그대로 쓴다). 선 채로 몇 cm 오르내림
        (자세 기울기 보정이 천천히 바뀜)은 따라간다. 앉으면 골반이 발에 가까워지므로(FLOOR_STAND_DROP) 그동안의
        발 높이로는 바닥을 올리지 않는다: 오래 앉아 디딘 기록이 창 밖으로 빠져도 띄운 발이 다시 디딤 높이에
        들어오지 않게. 발이 일관되게 더 낮게 내려가면 (바닥을 너무 높게 잡았거나 폰을 천천히 올림) 백분위가
        내려가는 만큼 낮춘다 (창의 10% = 약 2초).
        """
        hist = self._floor_hist
        hips_y = float(world[ANCHOR_JOINT][0][1])
        for name, joint in FOOT_JOINTS.items():
            if joint in world and self._quality_ok(joint, t):
                y = float(world[joint][0][1])
                st = self._feet.get(name)
                if st is not None and st.locked:
                    leg = hips_y - y
                    if self._stand_leg is None or leg > self._stand_leg:  # 한 프레임 튐에 끌려 올라가지 않게 천천히
                        self._stand_leg = leg if self._stand_leg is None else self._stand_leg + 0.02 * (leg - self._stand_leg)
                standing = self._stand_leg is None or hips_y - y > self._stand_leg - self.FLOOR_STAND_DROP
                hist.append((t, y, standing))
                if self._floor_t0 is None:
                    self._floor_t0 = t
        while hist and t - hist[0][0] > self.FLOOR_WINDOW:
            hist.popleft()
        if len(hist) >= self.FLOOR_MIN_SAMPLES:
            y = float(np.percentile(np.fromiter((y for _t, y, _s in hist), float, len(hist)), self.FLOOR_PERCENTILE))
            if t - self._floor_t0 >= self.FLOOR_SETTLE:
                # 그 전에 잰 바닥도 넣는다: 추적이 끊긴 사이 처음 기록이 창 밖으로 빠졌어도 상한이 남게
                low = self._floor_low = min(v for v in (self._floor_low, self.floor_y, y) if v is not None)
                stand = [y for _t, y, standing in hist if standing]
                rise = (float(np.percentile(stand, self.FLOOR_PERCENTILE))
                        if len(stand) >= self.FLOOR_MIN_SAMPLES else low)
                y = min(y, max(low, min(rise, low + self.FLOOR_RISE)))
            self.floor_y = y

    def _foot_lock(self, name: str, c: np.ndarray, raw_y: float | None, t: float) -> np.ndarray:
        """발 고정. c: 필터·품질 게이트를 거친 발 위치 (출력 좌표 = 앵커 + 상대 위치), raw_y: 이번 프레임
        원본 높이 (ARKit이 '추적 중'으로 표시했을 때만, 아니면 None). 반환: 보낼 발 위치 (출력 좌표).

        고정 위치는 출력 좌표(앵커 = 아주 느리게 따라가는 골반)에 둔다. VRChat이 보는 값은 (발 - 머리)이므로
        고정 중에는 발의 상대 위치 잡음 대신 앵커의 느린 움직임만 남는다. 앵커가 몸보다 늦는 동안(쪼그려
        앉기, 디딘 발 위로 몸이 나아가는 첫 걸음)은 그만큼을 보정한다 (_lock_target).
        """
        s = self.settings
        st = self._feet.get(name)
        if st is None:
            st = self._feet[name] = _FootLock()
        dt = t - st.recent[-1][0] if st.recent else 0.0
        vc = float(np.linalg.norm(c - st.recent[-1][1])) / dt if st.recent and 0.0 < dt <= self.GAP_RESET else 0.0
        if dt > self.GAP_RESET:
            # 골반을 놓쳐 한동안 처리하지 못했다: 그 전 기록으로 디딤·풀림을 판단하지 않는다
            st.recent.clear()
            st.since, st.samples, st.release_since, st.taut_since, st.sink_since = None, [], None, None, None
        st.recent.append((t, c))
        if st.offset is not None:
            # 고정·풀림 순간의 차이를 blend_sec 동안 0으로 줄인다. 진행은 처리한 프레임 시간으로 센다
            # (골반을 놓친 동안 건너뛰어 한 번에 튀지 않게). 발이 빨리 움직이는 동안은 천천히 줄여, 따라잡는 몫이
            # 발 자체의 움직임에 얹혀 한 프레임 이동이 커지지 않게 한다 (발을 빨리 들 때, 추적이 깜빡일 때).
            slow = min(1.0, max(self.BLEND_SLOW_MIN, 1.0 - vc / self.BLEND_BUDGET_SPEED))
            st.blend = min(1.0, st.blend + slow * min(dt, 1.0 / 30) / st.blend_sec)
        while len(st.recent) > 2 and t - st.recent[1][0] >= self.SPEED_WINDOW:
            st.recent.popleft()
        t_old, c_old = st.recent[0]
        span = t - t_old
        st.speed = float(np.linalg.norm((c - c_old)[[0, 2]])) / span if span >= 0.05 else 0.0
        st.vspeed = abs(float(c[1] - c_old[1])) / span if span >= 0.05 else 0.0
        height = None if raw_y is None or self.floor_y is None else raw_y - self.floor_y
        st.height = height
        if st.release_pending:  # 기능을 끄거나 모드를 바꿨다: 직전에 실제로 보낸 위치에서 c로 섞는다
            st.release_pending = False
            self._start_blend(st, st.out, c)

        target = c
        if st.locked:
            # 발을 들면(높이, 또는 조금 들고 빨리 옮김) 바로 푼다. 여유 밖으로 계속 끌려가면(DRAG_CONFIRM,
            # 발을 끌어 옮김) 풀어서 그대로 따라간다. ARKit이 발을 놓치면(height None) 높이는 판단하지 않고,
            # 그때 c는 품질 게이트가 붙잡아 둔 '앵커 + 마지막 안정 상대 위치'라서 몸이 멀리 가면 끌려가다 풀린다.
            # 발이 고정 높이보다 계속 낮으면(LOCK_SINK_SEC) 바닥이 내려간 것이라 풀고 새 높이에 다시 고정한다.
            other = self._feet.get(next(n for n in FOOT_JOINTS if n != name))
            fresh = other is not None and other.locked and other.resid_t is not None and t - other.resid_t <= 0.05
            corr, taut, sink = self._lock_target(st, c, dt, t, self._lag_dead(name),
                                                 other.resid if fresh else None)
            target = st.pos + corr
            if taut is None:
                st.taut_since = None
            elif st.taut_since is None:
                st.taut_since = t
            if taut is not None:
                st.taut_jump = taut == "jump"
            if not sink or raw_y is None:  # 발을 놓친 프레임(캐시된 c)은 낮아졌다는 증거가 아니다
                st.sink_since = None
            elif st.sink_since is None:
                st.sink_since = t
            lifted = height is not None and (
                height > s.foot_lock_height + self.RELEASE_HEIGHT_EXTRA
                or (height > s.foot_lock_height and st.speed > s.foot_lock_speed * self.RELEASE_SPEED_RATIO))
            if not lifted:
                st.release_since = None
            elif st.release_since is None:
                st.release_since = t
            dragged = st.taut_since is not None and t - st.taut_since >= self.DRAG_CONFIRM
            if ((st.release_since is not None and t - st.release_since >= self.RELEASE_CONFIRM)
                    or dragged or (st.sink_since is not None and t - st.sink_since >= self.LOCK_SINK_SEC)):
                # 끌어 옮기다 풀면 여유만큼 남은 차이를 발 속도에 맞춰 천천히 따라잡는다 (느린 끌기에서 풀 때마다
                # 몰아서 움직이지 않게)
                drag = dragged and not st.taut_jump and st.release_since is None
                speed = (min(self.BLEND_MAX_SPEED, max(self.BLEND_DRAG_SPEED, 2.0 * st.speed)) if drag
                         else self.BLEND_MAX_SPEED)
                st.locked = False
                st.release_since = st.taut_since = st.sink_since = None
                self._start_blend(st, self._blended(st, target), c, speed)
                target = c
            else:
                # 여유 안에서도 아주 천천히 원본 쪽으로 옮겨, 오래 남는 어긋남(걷다가 멈춘 직후 고정해 앵커가
                # 따라잡으며 생긴 것, 기울기 보정 변화, 무시한 튐 뒤에 그대로 남은 위치)을 몇 초에 걸쳐 없앤다.
                # 고정 높이 아래로는 옮기지 않는다 (아래쪽 떨림은 무시).
                a = 1.0 - math.exp(-min(dt, 0.1) / self.LOCK_CREEP_TAU)
                st.pos = st.pos + (c - target) * a
                st.pos[1] = max(float(st.pos[1]), st.base_y)
                st.jump = st.jump * (1.0 - a)
                target = st.pos + corr
        else:
            planted = (height is not None and height < s.foot_lock_height and st.speed < s.foot_lock_speed
                       and st.vspeed < self.LOCK_MAX_VSPEED and float(np.linalg.norm(self._lag)) < self.LOCK_MAX_LAG)
            if planted:
                if st.since is None:
                    st.since, st.samples = t, []
                # 앵커가 아직 몸을 따라오는 중이면(멈춘 직후) 따라잡은 뒤의 자리에 고정한다. 골반 잡음과 구별이
                # 안 되는 만큼(LOCK_LAG_DEAD)은 빼지 않는다 (남는 어긋남은 천천히 없앤다).
                st.samples.append(c - _deadzone(self._lag, self._lag_basis, self._lag_dead(name)))
                if t - st.since >= self.LOCK_DWELL:
                    st.locked = True
                    st.pos = np.mean(st.samples, axis=0)
                    st.base_y = float(st.pos[1])
                    st.jump, st.jump_t, st.last = np.zeros(3), None, None
                    st.since, st.samples = None, []
                    target = st.pos + self._lock_target(st, c, 0.0, t, self._lag_dead(name))[0]
                    self._start_blend(st, self._blended(st, c), target)
            else:
                st.since, st.samples = None, []

        # 목표(고정 위치 또는 c)에 줄어드는 차이를 더한다. 풀릴 때도 c가 움직이는 속도는 그대로 살리므로
        # 따라잡느라 튀지 않는다.
        st.out = self._blended(st, target)
        if not s.foot_lock and not st.locked and st.offset is None:
            del self._feet[name]  # 끈 뒤 섞기를 마쳤다
        return st.out

    def _lag_dead(self, name: str) -> tuple[float, float, float]:
        """두 발 다 디딘 동안(다른 발이 고정돼 있고 들리기 시작하지 않음)은 LOCK_LAG_DEAD_BOTH."""
        other = self._feet.get(next(n for n in FOOT_JOINTS if n != name))
        down = (other is not None and other.locked
                and (other.height is None or other.height < self.settings.foot_lock_height / 2))
        return self.LOCK_LAG_DEAD_BOTH if down else self.LOCK_LAG_DEAD

    def _lock_target(self, st: _FootLock, c: np.ndarray, dt: float, t: float, dead,
                     other: np.ndarray | None = None) -> tuple[np.ndarray, str | None, bool]:
        """고정된 발의 앵커 지연 보정(보낼 위치 = st.pos + 보정), 끌려가는 중인가 ("drag" / "jump" / None,
        계속되면 푼다), 고정 높이보다 LOCK_SINK 넘게 낮은가. dead: 앵커 지연에서 버릴 잡음 (Processor._lag_dead).
        other: 다른 발도 고정 중이면 그 발의 수평 차이.

        1. 앵커 지연 보정 (두 모드): 몸이 움직이면 앵커(1초 시간 상수)가 늦는 만큼 고정된 발이 (발 - 머리)로
           틀어진다 (쪼그려 앉기 20cm 가까이). 그 지연은 두 곳에서 보인다: 골반 절대 위치(앵커 - One Euro 골반)와
           이 발(c - 고정 위치). 축(깊이·위·좌우)마다 둘이 같은 쪽을 가리키는 만큼만(작은 쪽) 옮기고, 그중
           dead 까지는 버린다. 골반 추정이 흔들리면 두 값이 같이 흔들리므로(둘 다 골반 기준) 같은 쪽을
           가리키는 잡음도 있다 (LOCK_LAG_DEAD).
        "stable": 고정 위치는 그대로 두고, 수평으로 LOCK_JUMP_RADIUS 를 넘게 벗어나면 끌려가는 중으로 본다.
        "motion":
        2. 빠른 수평 튐: 바닥에 붙은 발이 LOCK_JUMP_SPEED 보다 빨리 옆으로 가는 것은 ARKit 오류로 보고
           무시한다 (발을 빨리 옮기면 들게 되어 높이로 풀린다). 무시한 튐이 LOCK_JUMP_KEEP 넘게 남은 채로
           LOCK_JUMP_SETTLE 동안 더 튀지 않거나(빠르게 끌어 옮김), 모두 합쳐 LOCK_JUMP_RADIUS 를 넘으면 끌려가는
           중으로 본다.
        3. 여유: 남는 차이가 수평으로 foot_lock_slack, 위로 LOCK_SLACK_Y 를 넘으면 그 밖으로는 따라간다
           (낮은 제자리걸음·발 끌기·느린 이동이 여유만큼만 덜 움직임). 아래로는 고정할 때 높이까지만 돌아온다:
           디딘 발은 바닥 아래로 갈 수 없으니 아래쪽 떨림은 무시한다. 두 발이 다 고정돼 있으면 수평 차이 중
           두 발이 같은 쪽으로 함께 밀린 만큼은 LOCK_COMMON_MAX 까지 빼고 본다: ARKit은 두 발을 같이 옮겨 잡는
           일이 잦고(실측 정지 중 두 발 사이 벡터의 흔들림이 각 발보다 작다), 한 발을 끌면 그 발만 움직인다.
        """
        corr = _agree(c - st.pos, self._lag, self._lag_basis, dead)
        cc = c - corr
        sink = float(cc[1]) < st.base_y - self.LOCK_SINK
        d = cc - st.pos
        if self.settings.foot_lock_mode == "stable":
            st.last = cc
            return corr, "jump" if math.hypot(d[0], d[2]) > self.LOCK_JUMP_RADIUS else None, sink
        if st.last is not None and dt > 0.0:
            step = cc - st.last
            step[1] = 0.0
            if float(np.linalg.norm(step)) > self.LOCK_JUMP_SPEED * dt:
                st.jump = st.jump + step
                st.jump_t = t
        st.last = cc
        dh = np.array([d[0], 0.0, d[2]]) - st.jump
        st.resid, st.resid_t = dh, t
        if other is not None:
            dh = dh - _clip(_agree(dh, other, (self._lag_basis[0], self._lag_basis[2])), self.LOCK_COMMON_MAX)
        move = np.zeros(3)
        taut = None
        if (math.hypot(d[0], d[2]) > self.LOCK_JUMP_RADIUS
                or (float(np.linalg.norm(st.jump)) > self.LOCK_JUMP_KEEP and st.jump_t is not None
                    and t - st.jump_t >= self.LOCK_JUMP_SETTLE)):
            taut = "jump"
        h, r = float(np.linalg.norm(dh)), self.settings.foot_lock_slack
        if h > r:
            move = dh * (1.0 - r / h)
            taut = taut or "drag"
        if d[1] > self.LOCK_SLACK_Y:
            move[1] = d[1] - self.LOCK_SLACK_Y
            taut = taut or "drag"
        elif d[1] < 0.0 and st.pos[1] > st.base_y:
            move[1] = max(float(d[1]), st.base_y - float(st.pos[1]))
        m, lim = float(np.linalg.norm(move)), self.LOCK_FOLLOW_SPEED * min(dt, 0.1)
        if m > lim:
            move = move * (lim / m)
        st.pos = st.pos + move
        return corr, taut, sink

    @staticmethod
    def _blended(st: _FootLock, target: np.ndarray) -> np.ndarray:
        """target 에 아직 남은 섞기 차이를 더한 값 (섞기를 마쳤으면 차이를 지운다)."""
        if st.offset is None:
            return target
        if st.blend >= 1.0:
            st.offset = None
            return target
        a = st.blend * st.blend * (3.0 - 2.0 * st.blend)
        return target + st.offset * (1.0 - a)

    def _start_blend(self, st: _FootLock, current: np.ndarray, target: np.ndarray,
                     speed: float | None = None) -> None:
        """이번 프레임에 보냈을 위치(current)에서 새 목표(target)로 섞기 시작한다. speed: 섞기 최대 평균 속도
        (m/s, 기본 BLEND_MAX_SPEED)."""
        if st.out is None:
            st.offset = None
            return
        st.offset = current - target
        st.blend = 0.0
        st.blend_sec = max(self.LOCK_BLEND, float(np.linalg.norm(st.offset)) / (speed or self.BLEND_MAX_SPEED))

    def _quality_ok(self, name: str, t: float) -> bool:
        """최근 gate_window 초 동안 이 관절이 충분히 추적됐고 원본이 심하게 흔들리지 않았는가."""
        s = self.settings
        if not s.quality_gate:
            return True
        h = self._qhist.get(name)
        if not h:
            return True
        window = [r for (ht, r) in h if t - ht <= s.gate_window]
        if len(window) < 3:
            return True
        tracked = [r for r in window if r is not None]
        if len(tracked) / len(window) < s.gate_min_tracked:
            return False
        if len(tracked) >= 4:
            steps = np.linalg.norm(np.diff(np.array(tracked), axis=0), axis=1)
            if float(np.median(steps)) > s.gate_max_jitter:
                return False
        return True

    def _slew_limit(self, name: str, rel: np.ndarray, t: float) -> np.ndarray:
        """골반 기준 상대 위치가 물리적으로 불가능하게 빨리 변하면 그만큼 잘라낸다."""
        s = self.settings
        prev = self._slew.get(name)
        # 고정(유지)이 풀린 직후에는 새 위치로 천천히 옮겨 계단식 점프를 막는다
        held_at = self._held_at.get(name)
        reentry = held_at is not None and t - held_at < self.REENTRY_SEC
        speed = s.max_rel_speed
        if reentry:
            speed = self.REENTRY_SPEED if speed <= 0 else min(speed, self.REENTRY_SPEED)
        if prev is not None and speed > 0:
            pt, pr = prev
            dt = t - pt
            if dt > 0:
                d = rel - pr
                n = float(np.linalg.norm(d))
                lim = speed * min(dt, 0.1)
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
