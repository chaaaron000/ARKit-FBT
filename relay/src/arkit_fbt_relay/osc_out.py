"""VRChat OSC 트래커 송신."""

from __future__ import annotations

from collections.abc import Callable, Sequence

from pythonosc.udp_client import SimpleUDPClient

from .mathutil import to_unity_pos, to_unity_quat, unity_euler_from_quat
from .pipeline import TrackerOut


class OscOut:
    def __init__(self, host: str = "127.0.0.1", port: int = 9000, dry_run: bool = False,
                 recorder: Callable[[str, list[float]], None] | None = None):
        self.host, self.port = host, port
        self.dry_run = dry_run
        self.recorder = recorder
        self.client = None if dry_run else SimpleUDPClient(host, port)
        self.sent = 0
        self.errors = 0

    def send(self, address: str, values: Sequence[float]) -> None:
        vals = [float(v) for v in values]
        self.sent += 1
        if self.recorder is not None:
            self.recorder(address, vals)
        if self.client is None:
            return
        try:
            self.client.send_message(address, vals)
        except OSError:
            # VRChat이 꺼져 있으면 Windows에서 ConnectionResetError 가 날 수 있다
            self.errors += 1

    def send_tracker(self, t: TrackerOut) -> None:
        base = f"/tracking/trackers/{t.index}"
        self.send(f"{base}/position", to_unity_pos(t.pos))
        self.send(f"{base}/rotation", unity_euler_from_quat(to_unity_quat(t.rot)))

    def send_head_position(self, pos) -> None:
        self.send("/tracking/trackers/head/position", to_unity_pos(pos))

    def send_head_yaw(self, yaw_deg: float) -> None:
        # 한 번만 보내면 VRChat이 즉시 1회 방향 정렬을 한다 (계속 보내면 천천히 따라감)
        self.send("/tracking/trackers/head/rotation", [0.0, yaw_deg, 0.0])
