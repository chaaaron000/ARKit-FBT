import Foundation
import Network

/// PC 중계 프로그램으로 UDP 패킷을 보내고 응답(ack)을 받는다.
/// 모든 공개 메서드는 메인 스레드에서 호출한다.
final class UDPSender {
    var onAck: ((Data) -> Void)?
    var onState: ((String) -> Void)?

    private let queue = DispatchQueue(label: "arkitfbt.udp")
    private var connection: NWConnection?
    private var endpoint: NWEndpoint?

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
    }

    func send(_ data: Data) {
        connection?.send(content: data, completion: .idempotent)
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
                    self.onState?("대기 중: \(error.localizedDescription)")
                case .failed(let error):
                    self.onState?("실패: \(error.localizedDescription)")
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
                self.onState?("자동 찾기 실패: \(error.localizedDescription)")
                self.stop()
                DispatchQueue.main.asyncAfter(deadline: .now() + 2) { [weak self] in self?.start() }
            case .waiting(let error):
                self.onState?("자동 찾기 대기: \(error.localizedDescription)")
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
