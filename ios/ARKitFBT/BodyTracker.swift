import ARKit
import Combine
import RealityKit
import UIKit

/// ARKit 바디 트래킹을 돌리고 매 프레임 관절 데이터를 PC로 보낸다.
final class BodyTracker: NSObject, ObservableObject, ARSessionDelegate {
    @Published private(set) var isSupported = ARBodyTrackingConfiguration.isSupported
    @Published private(set) var bodyTracked = false
    @Published private(set) var fps: Double = 0
    @Published private(set) var bodyFps: Double = 0
    @Published private(set) var pcConnected = false
    @Published private(set) var rttMs: Double = 0
    @Published private(set) var relayStatus: UInt8 = 0
    @Published private(set) var connectionText = "PC 주소를 입력하세요"
    @Published private(set) var thermalText = "-"
    @Published private(set) var missingJoints: [String] = []

    private weak var arView: ARView?
    private let sender = UDPSender()
    private var sendingEnabled = true
    private var seq: UInt32 = 0
    private let jointIndices: [Int]
    private var jointEntities: [ModelEntity] = []

    private var frameCount = 0
    private var bodyFrameCount = 0
    private var fpsWindowStart: TimeInterval = 0
    private var lastAckTime: TimeInterval = 0
    private var statusTimer: Timer?

    override init() {
        let definition = ARSkeletonDefinition.defaultBody3D
        jointIndices = JointSet.names.map { definition.index(for: ARSkeleton.JointName(rawValue: $0)) }
        super.init()
        missingJoints = zip(JointSet.names, jointIndices).filter { $0.1 == NSNotFound }.map { $0.0 }

        sender.onAck = { [weak self] data in
            DispatchQueue.main.async { self?.handleAck(data) }
        }
        sender.onState = { [weak self] text in
            self?.connectionText = text
        }
        statusTimer = Timer.scheduledTimer(withTimeInterval: 0.5, repeats: true) { [weak self] _ in
            self?.tickStatus()
        }
        tickStatus()
    }

    deinit {
        statusTimer?.invalidate()
    }

    // MARK: - 설정

    func configure(host: String, port: Int, enabled: Bool) {
        sendingEnabled = enabled
        if enabled, let p = UInt16(exactly: port) {
            sender.connect(host: host, port: p)
        } else {
            sender.disconnect()
        }
    }

    func attach(_ view: ARView) {
        arView = view
        view.session.delegate = self

        let anchor = AnchorEntity(world: SIMD3<Float>(0, 0, 0))
        let mesh = MeshResource.generateSphere(radius: 0.035)
        let material = SimpleMaterial(color: .systemTeal, isMetallic: false)
        jointEntities = JointSet.names.map { _ in
            let entity = ModelEntity(mesh: mesh, materials: [material])
            entity.isEnabled = false
            anchor.addChild(entity)
            return entity
        }
        view.scene.addAnchor(anchor)
        restartSession()
    }

    func restartSession() {
        guard ARBodyTrackingConfiguration.isSupported, let view = arView else { return }
        let config = ARBodyTrackingConfiguration()
        config.automaticSkeletonScaleEstimationEnabled = true
        view.session.run(config, options: [.resetTracking, .removeExistingAnchors])
    }

    // MARK: - ARSessionDelegate

    func session(_ session: ARSession, didUpdate frame: ARFrame) {
        var body: ARBodyAnchor?
        for anchor in frame.anchors {
            if let b = anchor as? ARBodyAnchor {
                body = b
                break
            }
        }
        let tracked = body?.isTracked ?? false

        if sendingEnabled {
            seq &+= 1
            let packet = PacketBuilder.frame(seq: seq,
                                             timestamp: frame.timestamp,
                                             camera: frame.camera.transform,
                                             body: body,
                                             jointIndices: jointIndices)
            sender.send(packet)
        }

        updateVisuals(tracked ? body : nil)
        if tracked != bodyTracked { bodyTracked = tracked }

        frameCount += 1
        if tracked { bodyFrameCount += 1 }
        let now = frame.timestamp
        if fpsWindowStart == 0 { fpsWindowStart = now }
        let elapsed = now - fpsWindowStart
        if elapsed >= 1 {
            fps = Double(frameCount) / elapsed
            bodyFps = Double(bodyFrameCount) / elapsed
            frameCount = 0
            bodyFrameCount = 0
            fpsWindowStart = now
        }
    }

    func session(_ session: ARSession, didFailWithError error: Error) {
        connectionText = "AR 오류: \(error.localizedDescription)"
    }

    // MARK: - 내부

    private func updateVisuals(_ body: ARBodyAnchor?) {
        guard let body else {
            for entity in jointEntities where entity.isEnabled { entity.isEnabled = false }
            return
        }
        let models = body.skeleton.jointModelTransforms
        for (i, idx) in jointIndices.enumerated() where i < jointEntities.count {
            guard idx != NSNotFound, idx < models.count else { continue }
            let m = body.transform * models[idx]
            jointEntities[i].position = SIMD3<Float>(m.columns.3.x, m.columns.3.y, m.columns.3.z)
            if !jointEntities[i].isEnabled { jointEntities[i].isEnabled = true }
        }
    }

    private func handleAck(_ data: Data) {
        guard let ack = RelayAck(data) else { return }
        let now = ProcessInfo.processInfo.systemUptime
        lastAckTime = now
        let rtt = (now - ack.timestamp) * 1000
        if rtt >= 0 && rtt < 5000 {
            rttMs = rttMs == 0 ? rtt : rttMs * 0.8 + rtt * 0.2
        }
        if relayStatus != ack.status { relayStatus = ack.status }
        if !pcConnected { pcConnected = true }
    }

    private func tickStatus() {
        let now = ProcessInfo.processInfo.systemUptime
        let connected = lastAckTime > 0 && now - lastAckTime < 2.0
        if connected != pcConnected { pcConnected = connected }
        if !connected { rttMs = 0 }

        let text: String
        switch ProcessInfo.processInfo.thermalState {
        case .nominal: text = "정상"
        case .fair: text = "약간 따뜻함"
        case .serious: text = "뜨거움"
        case .critical: text = "과열!"
        @unknown default: text = "알 수 없음"
        }
        if text != thermalText { thermalText = text }
    }
}
