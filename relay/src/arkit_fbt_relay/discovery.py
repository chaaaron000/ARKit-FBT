"""Bonjour(mDNS)로 이 PC를 알려서 iPhone 앱이 IP를 몰라도 찾을 수 있게 한다.

서비스 타입은 ios/project.yml 의 NSBonjourServices 와 같아야 한다.
"""

from __future__ import annotations

import socket

from zeroconf import IPVersion, ServiceInfo, Zeroconf

SERVICE_TYPE = "_arkitfbt._udp.local."


class Advertiser:
    def __init__(self, port: int, addresses: list[str], name: str | None = None):
        self.port = port
        self.addresses = addresses
        host = socket.gethostname()
        self.name = name or f"ARKit FBT Relay ({host})"
        self._zc: Zeroconf | None = None
        self._info: ServiceInfo | None = None

    def start(self) -> None:
        if not self.addresses:
            raise RuntimeError("알릴 LAN 주소가 없습니다")
        self._info = ServiceInfo(
            SERVICE_TYPE,
            f"{self.name}.{SERVICE_TYPE}",
            port=self.port,
            addresses=[socket.inet_aton(a) for a in self.addresses],
            properties={"v": "1"},
            server=f"{socket.gethostname()}-arkitfbt.local.",
        )
        self._zc = Zeroconf(ip_version=IPVersion.V4Only)
        self._zc.register_service(self._info, allow_name_change=True)

    def stop(self) -> None:
        if self._zc is None:
            return
        try:
            if self._info is not None:
                self._zc.unregister_service(self._info)
        finally:
            self._zc.close()
            self._zc = None


def lan_addresses(addresses: list[str]) -> list[str]:
    """iPhone이 닿을 수 있는 주소만 고른다 (WSL, Tailscale, 링크 로컬 제외)."""
    # 172.16~31 대역은 WSL/Hyper-V 가상 어댑터가 흔히 쓰므로, iPhone 핫스팟/USB(172.20.10.x)만 허용한다.
    return [a for a in addresses if a.startswith(("192.168.", "10.", "172.20.10."))]
