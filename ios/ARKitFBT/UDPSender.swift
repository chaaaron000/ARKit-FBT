import Foundation
import Network

/// PC 중계 프로그램으로 UDP 패킷을 보내고 응답(ack)을 받는다.
/// 모든 공개 메서드는 메인 스레드에서 호출한다.
final class UDPSender {
    var onAck: ((Data) -> Void)?
    var onState: ((String) -> Void)?

    private let queue = DispatchQueue(label: "arkitfbt.udp")
    private var connection: NWConnection?
    private var host = ""
    private var port: UInt16 = 0

    func connect(host newHost: String, port newPort: UInt16) {
        let trimmed = newHost.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed == host && newPort == port && connection != nil { return }
        host = trimmed
        port = newPort
        reconnect()
    }

    func disconnect() {
        connection?.cancel()
        connection = nil
        host = ""
        onState?("꺼짐")
    }

    func send(_ data: Data) {
        connection?.send(content: data, completion: .idempotent)
    }

    private func reconnect() {
        connection?.cancel()
        connection = nil
        guard !host.isEmpty, let nwPort = NWEndpoint.Port(rawValue: port) else {
            onState?("PC 주소를 입력하세요")
            return
        }
        let conn = NWConnection(host: NWEndpoint.Host(host), port: nwPort, using: .udp)
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
