import Foundation
import Network

/// PC 중계 프로그램으로 UDP 패킷을 보내고 응답(ack)을 받는다.
/// 모든 공개 메서드는 메인 스레드에서 호출한다.
final class UDPSender {
    var onAck: ((Data) -> Void)?
    var onState: ((String) -> Void)?
    /// 전송 실패나 네트워크 경로 문제 (nil = 문제 없음). 메인 스레드에서 호출된다.
    var onProblem: ((String?) -> Void)?

    private let queue = DispatchQueue(label: "arkitfbt.udp")
    private var connection: NWConnection?
    private var endpoint: NWEndpoint?
    private var lastSendErrorReport: TimeInterval = 0
    private var pathProblem: String?

    /// NWError를 사람이 이해할 수 있는 원인으로 바꾼다.
    static func explain(_ error: NWError) -> String {
        switch error {
        case .posix(let code):
            switch code {
            case .EHOSTUNREACH, .ENETUNREACH:
                return "PC에 닿을 수 없음 (\(code)). 로컬 네트워크 권한이나 VPN(Tailscale 등)을 확인하세요"
            case .EPERM, .EACCES:
                return "권한 거부 (\(code)). 설정 → 개인정보 보호 및 보안 → 로컬 네트워크에서 ARKit FBT를 켜세요"
            default:
                return "\(error.localizedDescription)"
            }
        case .dns(let code) where code == -65570:  // kDNSServiceErr_PolicyDenied
            return "로컬 네트워크 권한이 꺼져 있습니다. 설정 → 개인정보 보호 및 보안 → 로컬 네트워크에서 ARKit FBT를 켜세요"
        default:
            return error.localizedDescription
        }
    }

    static func describe(_ path: NWPath) -> String? {
        if path.status != .satisfied {
            if #available(iOS 14.2, *), path.unsatisfiedReason == .localNetworkDenied {
                return "로컬 네트워크 권한이 꺼져 있습니다. 설정 → 개인정보 보호 및 보안 → 로컬 네트워크에서 ARKit FBT를 켜세요"
            }
            return "네트워크 경로 없음 (Wi-Fi 연결을 확인하세요)"
        }
        if !path.usesInterfaceType(.wifi) && !path.usesInterfaceType(.wiredEthernet) {
            if path.usesInterfaceType(.other) {
                return "VPN(Tailscale 등)을 거쳐 나가고 있습니다. VPN을 끄거나 로컬 네트워크 접근을 허용하세요"
            }
            if path.usesInterfaceType(.cellular) {
                return "셀룰러로 나가고 있습니다. PC와 같은 Wi-Fi에 연결하세요"
            }
        }
        return nil
    }

    func connect(host: String, port: UInt16) {
        let trimmed = host.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty, let nwPort = NWEndpoint.Port(rawValue: port) else {
            disconnect()
            onState?("PC 주소를 입력하세요")
            return
        }
        connect(to: .hostPort(host: NWEndpoint.Host(trimmed), port: nwPort))
    }

    /// Bonjour로 찾은 서비스 등 임의의 endpoint로 연결한다.
    func connect(to newEndpoint: NWEndpoint) {
        if newEndpoint == endpoint && connection != nil { return }
        endpoint = newEndpoint
        reconnect()
    }

    func disconnect() {
        connection?.cancel()
        connection = nil
        endpoint = nil
        pathProblem = nil
        onProblem?(nil)
    }

    func send(_ data: Data) {
        guard let conn = connection else { return }
        conn.send(content: data, completion: .contentProcessed { [weak self] error in
            guard let error else { return }
            DispatchQueue.main.async {
                guard let self else { return }
                let now = ProcessInfo.processInfo.systemUptime
                if now - self.lastSendErrorReport < 1 { return }
                self.lastSendErrorReport = now
                self.onProblem?("전송 실패: " + UDPSender.explain(error))
            }
        })
    }

    private func reconnect() {
        connection?.cancel()
        connection = nil
        guard let target = endpoint else { return }
        let conn = NWConnection(to: target, using: .udp)
        conn.stateUpdateHandler = { [weak self, weak conn] state in
            DispatchQueue.main.async {
                guard let self, let conn, conn === self.connection else { return }
                switch state {
                case .ready:
                    self.onState?("연결 준비됨")
                case .waiting(let error):
                    self.onState?("대기 중: " + UDPSender.explain(error))
                case .failed(let error):
                    self.onState?("실패: " + UDPSender.explain(error))
                    // 잠시 후 다시 시도한다 (PC 프로그램이 재시작된 경우 등).
                    DispatchQueue.main.asyncAfter(deadline: .now() + 1) { [weak self] in
                        guard let self, self.connection === conn else { return }
                        self.reconnect()
                    }
                default:
                    break
                }
            }
        }
        conn.pathUpdateHandler = { [weak self, weak conn] path in
            let problem = UDPSender.describe(path)
            DispatchQueue.main.async {
                guard let self, let conn, conn === self.connection else { return }
                self.pathProblem = problem
                self.onProblem?(problem)
            }
        }
        connection = conn
        conn.start(queue: queue)
        receive(on: conn)
    }

    private func receive(on conn: NWConnection) {
        conn.receiveMessage { [weak self, weak conn] data, _, _, error in
            guard let self, let conn else { return }
            if let data, !data.isEmpty {
                self.onAck?(data)
            }
            if error != nil {
                if case .cancelled = conn.state { return }
                self.queue.asyncAfter(deadline: .now() + 0.5) { [weak self, weak conn] in
                    guard let self, let conn else { return }
                    self.receive(on: conn)
                }
                return
            }
            self.receive(on: conn)
        }
    }
}

enum LocalAddress {
    /// iPhone의 Wi-Fi(en0) IPv4 주소
    static func wifi() -> String? {
        var ifaddr: UnsafeMutablePointer<ifaddrs>?
        guard getifaddrs(&ifaddr) == 0, let first = ifaddr else { return nil }
        defer { freeifaddrs(ifaddr) }
        for ptr in sequence(first: first, next: { $0.pointee.ifa_next }) {
            let iface = ptr.pointee
            guard let addr = iface.ifa_addr, addr.pointee.sa_family == UInt8(AF_INET) else { continue }
            guard String(cString: iface.ifa_name) == "en0" else { continue }
            var host = [CChar](repeating: 0, count: Int(NI_MAXHOST))
            if getnameinfo(addr, socklen_t(addr.pointee.sa_len), &host, socklen_t(host.count),
                           nil, 0, NI_NUMERICHOST) == 0 {
                return String(cString: host)
            }
        }
        return nil
    }
}

/// Bonjour로 같은 네트워크의 PC 중계 프로그램을 찾는다.
/// 서비스 타입은 relay/src/arkit_fbt_relay/discovery.py 와 ios/project.yml(NSBonjourServices)과 같아야 한다.
final class RelayBrowser {
    static let serviceType = "_arkitfbt._udp"

    struct Found: Equatable {
        let name: String
        let endpoint: NWEndpoint
    }

    var onChange: (([Found]) -> Void)?
    var onState: ((String) -> Void)?
    private var browser: NWBrowser?

    var isRunning: Bool { browser != nil }

    func start() {
        guard browser == nil else { return }
        let params = NWParameters()
        params.includePeerToPeer = false
        let b = NWBrowser(for: .bonjour(type: RelayBrowser.serviceType, domain: nil), using: params)
        b.browseResultsChangedHandler = { [weak self] results, _ in
            let found = results.compactMap { result -> Found? in
                guard case let .service(name, _, _, _) = result.endpoint else { return nil }
                return Found(name: name, endpoint: result.endpoint)
            }.sorted { $0.name < $1.name }
            self?.onChange?(found)
        }
        b.stateUpdateHandler = { [weak self, weak b] state in
            guard let self, let b, b === self.browser else { return }
            switch state {
            case .failed(let error):
                self.onState?("자동 찾기 실패: " + UDPSender.explain(error))
                self.stop()
                DispatchQueue.main.asyncAfter(deadline: .now() + 2) { [weak self] in self?.start() }
            case .waiting(let error):
                self.onState?("자동 찾기 대기: " + UDPSender.explain(error))
            default:
                break
            }
        }
        browser = b
        b.start(queue: .main)
    }

    func stop() {
        browser?.cancel()
        browser = nil
        onChange?([])
    }
}
