"""실행: uv run arkit-fbt-relay  (옵션은 --help)"""

from __future__ import annotations

import argparse
import json
import socket
import sys
import time
from pathlib import Path

from .engine import Engine
from .pipeline import Settings
from .protocol import DEFAULT_PORT

RELAY_DIR = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = RELAY_DIR / "config.json"
RECORDINGS_DIR = RELAY_DIR / "recordings"


def local_ipv4_addresses() -> list[str]:
    addrs: set[str] = set()
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            addrs.add(info[4][0])
    except OSError:
        pass
    try:  # 기본 경로로 나가는 주소 (실제로 패킷을 보내지는 않는다)
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("10.255.255.255", 1))
            addrs.add(s.getsockname()[0])
    except OSError:
        pass

    def rank(a: str) -> tuple[int, str]:
        if a.startswith(("192.168.", "10.")):
            return 0, a
        if a.startswith("172.20.10."):  # iPhone USB/핫스팟
            return 1, a
        if a.startswith(("127.", "169.254.")):
            return 9, a
        return 5, a

    return sorted((a for a in addrs if not a.startswith("127.")), key=rank)


def load_settings(path: Path) -> Settings:
    if path.exists():
        try:
            return Settings.from_dict(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError, TypeError) as e:
            print(f"설정 파일을 읽지 못해 기본값을 씁니다: {e}", file=sys.stderr)
    return Settings()


def save_settings(settings: Settings, path: Path) -> None:
    path.write_text(json.dumps(settings.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")


def run_headless(engine: Engine) -> None:
    last_log = 0
    try:
        while True:
            time.sleep(1.0)
            snap = engine.snapshot()
            for line in snap.logs[last_log:]:
                print(line)
            last_log = len(snap.logs)
            if snap.phone_alive:
                n = len(snap.frame.trackers) if snap.frame else 0
                print(f"  수신 {snap.packet_rate:5.1f}/s  흔들림 {snap.interval_jitter_ms:4.1f}ms  "
                      f"몸 {'O' if snap.tracked else 'X'}  정면 {snap.facing:+.2f}  트래커 {n}  "
                      f"OSC {snap.osc_sent}", flush=True)
            else:
                print("  iPhone 대기 중...", flush=True)
    except KeyboardInterrupt:
        pass


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="arkit-fbt-relay", description="iPhone ARKit → VRChat OSC 트래커 중계")
    p.add_argument("--listen-host", default="0.0.0.0", help="iPhone 데이터를 받을 주소 (기본: 모든 네트워크)")
    p.add_argument("--listen-port", type=int, default=DEFAULT_PORT)
    p.add_argument("--osc-host", default="127.0.0.1", help="VRChat 주소")
    p.add_argument("--osc-port", type=int, default=9000, help="VRChat OSC 수신 포트")
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--headless", action="store_true", help="창 없이 콘솔로 실행")
    p.add_argument("--dry-run", action="store_true", help="VRChat으로 보내지 않음")
    p.add_argument("--record", action="store_true",
                   help=f"시작하자마자 iPhone 데이터를 녹화 ({RECORDINGS_DIR})")
    args = p.parse_args(argv)

    settings = load_settings(args.config)
    engine = Engine(settings, args.listen_host, args.listen_port, args.osc_host, args.osc_port,
                    dry_run=args.dry_run, log=print if args.headless else None)
    try:
        engine.start()
    except OSError as e:
        print(f"포트 {args.listen_port} 를 열 수 없습니다: {e}\n이미 실행 중인지 확인하세요.", file=sys.stderr)
        return 1

    ips = local_ipv4_addresses()
    engine.log("iPhone 앱에 입력할 이 PC 주소: " + (", ".join(ips) if ips else "(찾지 못함)")
               + f"  포트 {args.listen_port}")
    if args.record:
        engine.start_recording(RECORDINGS_DIR / (time.strftime("%Y%m%d-%H%M%S") + ".afbtrec"))

    try:
        if args.headless:
            run_headless(engine)
        else:
            from .gui import run_gui
            run_gui(engine, ips, lambda: save_settings(settings, args.config), RECORDINGS_DIR)
    finally:
        engine.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
