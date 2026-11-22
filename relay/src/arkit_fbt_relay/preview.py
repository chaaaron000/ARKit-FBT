"""iPhone 카메라 미리보기 조각 재조립. 형식은 ios/ARKitFBT/PreviewSender.swift 주석 참고."""

from __future__ import annotations

import math
import struct
import time
from dataclasses import dataclass

import numpy as np

from .mathutil import quat_conj, quat_rotate

CHUNK_MAGIC = b"AFBC"
CHUNK = struct.Struct("<4sBBHHHI")  # 16 bytes
BLOB = struct.Struct("<HHHHd")  # 16 bytes
CHUNK_PAYLOAD = 1100


@dataclass
class Preview:
    frame_id: int
    width: int
    height: int
    timestamp: float
    joints: list[tuple[float, float] | None]  # JOINT_NAMES 순서, 원본 이미지 0..1
    jpeg: bytes
    received: float


def is_preview_chunk(data: bytes) -> bool:
    return data[:4] == CHUNK_MAGIC


def parse_blob(frame_id: int, blob: bytes) -> Preview:
    if len(blob) < BLOB.size:
        raise ValueError("미리보기 데이터가 너무 짧음")
    w, h, n, _r, ts = BLOB.unpack_from(blob, 0)
    off = BLOB.size
    joints: list[tuple[float, float] | None] = []
    for i in range(n):
        u, v = struct.unpack_from("<2f", blob, off + i * 8)
        joints.append(None if math.isnan(u) or math.isnan(v) else (u, v))
    off += n * 8
    return Preview(frame_id, w, h, ts, joints, bytes(blob[off:]), time.monotonic())


def build_chunks(frame_id: int, width: int, height: int, timestamp: float,
                 joints: list[tuple[float, float] | None], jpeg: bytes) -> list[bytes]:
    """테스트용: Swift PreviewSender 와 같은 조각을 만든다."""
    blob = BLOB.pack(width, height, len(joints), 0, timestamp)
    for p in joints:
        blob += struct.pack("<2f", *(p if p is not None else (math.nan, math.nan)))
    blob += jpeg
    count = (len(blob) + CHUNK_PAYLOAD - 1) // CHUNK_PAYLOAD
    return [CHUNK.pack(CHUNK_MAGIC, 1, 0, i, count, 0, frame_id) + blob[i * CHUNK_PAYLOAD:(i + 1) * CHUNK_PAYLOAD]
            for i in range(count)]


class Reassembler:
    """조각을 모아 완성된 미리보기를 돌려준다. 오래된/불완전한 프레임은 버린다."""

    MAX_PENDING = 4

    def __init__(self) -> None:
        self._pending: dict[int, tuple[int, dict[int, bytes]]] = {}
        self._latest_id = -1
        self.completed = 0
        self.dropped = 0

    def add(self, data: bytes) -> Preview | None:
        if len(data) < CHUNK.size:
            return None
        magic, _ver, _r, index, count, _r2, frame_id = CHUNK.unpack_from(data, 0)
        if magic != CHUNK_MAGIC or count == 0 or index >= count:
            return None
        if frame_id <= self._latest_id:
            if self._latest_id - frame_id < 1000:
                return None  # 이미 더 최신 프레임을 보여 줌
            # 크게 작아졌다 = iPhone 앱이 다시 시작됨
            self._latest_id = -1
            self._pending.clear()
        total, parts = self._pending.setdefault(frame_id, (count, {}))
        if total != count:
            return None
        parts[index] = data[CHUNK.size:]
        if len(parts) < count:
            if len(self._pending) > self.MAX_PENDING:
                oldest = min(self._pending)
                del self._pending[oldest]
                self.dropped += 1
            return None
        del self._pending[frame_id]
        for fid in [f for f in self._pending if f < frame_id]:
            del self._pending[fid]
            self.dropped += 1
        self._latest_id = frame_id
        self.completed += 1
        return parse_blob(frame_id, b"".join(parts[i] for i in range(count)))


def display_rotation(camera_rot) -> int:
    """카메라 자세로 미리보기를 몇 번 90° 반시계 회전해야 똑바로 서는지 구한다 (0..3).

    ARKit 카메라 좌표: 원본 이미지(센서, landscapeRight) 기준 +X = 오른쪽, +Y = 위.
    """
    up_cam = quat_rotate(quat_conj(np.asarray(camera_rot, float)), [0.0, 1.0, 0.0])
    ux, uy = float(up_cam[0]), -float(up_cam[1])  # 이미지 좌표 (y 아래)
    angle = math.degrees(math.atan2(ux, -uy))
    return round(angle / 90.0) % 4


def framing_warnings(joints: list[tuple[float, float] | None], names: tuple[str, ...]) -> list[str]:
    """머리나 발이 화면 밖이면 경고."""
    idx = {n: i for i, n in enumerate(names)}
    checks = [("머리", "head_joint"), ("왼발", "left_foot_joint"), ("오른발", "right_foot_joint")]
    out = []
    if not joints:
        return ["몸이 인식되지 않음"]
    for label, name in checks:
        i = idx.get(name)
        p = joints[i] if i is not None and i < len(joints) else None
        if p is None or not (0.02 <= p[0] <= 0.98 and 0.02 <= p[1] <= 0.98):
            out.append(f"{label}가 화면 밖")
    return out
