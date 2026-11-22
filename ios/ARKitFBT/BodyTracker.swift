import ARKit
import Combine
import SceneKit
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
    /// 자동 찾기로 찾은 PC 이름 목록
    @Published private(set) var discoveredNames: [String] = []
    /// 지금 자동 찾기로 연결된 PC 이름 (수동 주소를 쓰면 nil)
    @Published private(set) var autoTargetName: String?
    /// 연결이 안 될 때 원인 (권한, VPN, 전송 실패 등)
    @Published private(set) var problemText: String?
    @Published private(set) var discoveryText: String?
    @Published private(set) var phoneAddress: String?
    @Published private(set) var videoFormatText = "-"
    @Published private(set) var supportedFormatsText = "-"
    private var lowResCamera = true

    /// 화면과 독립된 AR 세션. 미리보기(ARSCNView)는 있을 때만 이 세션을 빌려 쓴다.
    let session = ARSession()
    /// 미리보기가 보일 때만 관절 노드를 갱신한다
    var showVisuals = true
    private var started = false
    private let sender = UDPSender()
    private let browser = RelayBrowser()
    private let preview = PreviewSender()
    private var discovered: [RelayBrowser.Found] = []
    private var sendingEnabled = true
    private var autoDiscover = true
    private var manualHost = ""
    private var manualPort = WireProtocol.defaultPort
    private var seq: UInt32 = 0
    private let jointIndices: [Int]

    private var frameCount = 0
    private var bodyFrameCount = 0
    private var fpsWindowStart: TimeInterval = 0
    private var lastAckTime: TimeInterval = 0
    private var statusTimer: Timer?

    override init() {
        let definition = ARSkeletonDefinition.defaultBody3D
        jointIndices = JointSet.names.map { definition.index(for: ARSkeleton.JointName(rawValue: $0)) }
        super.init()
        session.delegate = self
        missingJoints = zip(JointSet.names, jointIndices).filter { $0.1 == NSNotFound }.map { $0.0 }

        sender.onAck = { [weak self] data in
            DispatchQueue.main.async { self?.handleAck(data) }
        }
        sender.onState = { [weak self] text in
            self?.connectionText = text
        }
        browser.onChange = { [weak self] found in
            guard let self else { return }
            self.discovered = found
            self.discoveredNames = found.map { $0.name }
            self.updateConnection()
        }
        browser.onState = { [weak self] text in
            self?.discoveryText = text
        }
        sender.onProblem = { [weak self] text in
            guard let self, self.problemText != text else { return }
            self.problemText = text
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

    func configure(host: String, port: Int, enabled: Bool, autoDiscover auto: Bool) {
        sendingEnabled = enabled
        autoDiscover = auto
        manualHost = host.trimmingCharacters(in: .whitespacesAndNewlines)
        manualPort = port
        if enabled && auto {
            browser.start()
        } else if browser.isRunning {
            browser.stop()
        }
        updateConnection()
    }

    /// 우선순위: 자동으로 찾은 PC → 수동 입력 주소
    private func updateConnection() {
        guard sendingEnabled else {
            sender.disconnect()
            autoTargetName = nil
            connectionText = "전송 꺼짐"
            return
        }
        if autoDiscover, let first = discovered.first {
            if autoTargetName != first.name { autoTargetName = first.name }
            sender.connect(to: first.endpoint)
            return
        }
        if autoTargetName != nil { autoTargetName = nil }
        if !manualHost.isEmpty, let p = UInt16(exactly: manualPort) {
            sender.connect(host: manualHost, port: p)
        } else {
            sender.disconnect()
            connectionText = autoDiscover ? "PC를 찾는 중… (PC 프로그램이 켜져 있나요?)" : "PC 주소를 입력하세요"
        }
    }

    /// 앱 시작 시 한 번 호출한다. 화면(미리보기) 유무와 관계없이 세션은 계속 돈다.
    func start() {
        guard !started else { return }
        started = true
        restartSession()
    }

    /// 미리보기 화면이 사라진 뒤 세션이 멈추지 않도록 다시 이어서 실행한다 (리셋 없음).
    func ensureRunning() {
        guard started, let config = session.configuration else { return }
        session.run(config)
    }

    /// 미리보기(ARSCNView)에 붙일 관절 표시 노드
    let visualRoot: SCNNode = {
        let root = SCNNode()
        let sphere = SCNSphere(radius: 0.035)
        sphere.firstMaterial?.diffuse.contents = UIColor.systemTeal
        sphere.firstMaterial?.lightingModel = .constant
        for _ in JointSet.names {
            let node = SCNNode(geometry: sphere)
            node.isHidden = true
            root.addChildNode(node)
        }
        return root
    }()

    /// true면 가장 높은 fps 중 가장 낮은 해상도를 고른다 (발열↓). false면 같은 fps 중 가장 높은 해상도.
    func setLowResCamera(_ value: Bool) {
        guard value != lowResCamera else { return }
        lowResCamera = value
        restartSession()
    }

    func restartSession() {
        guard ARBodyTrackingConfiguration.isSupported, started else { return }
        let config = ARBodyTrackingConfiguration()
        config.automaticSkeletonScaleEstimationEnabled = true

        let formats = ARBodyTrackingConfiguration.supportedVideoFormats
        supportedFormatsText = formats.map(BodyTracker.describe).joined(separator: ", ")
        let maxFps = formats.map { $0.framesPerSecond }.max() ?? 30
        let candidates = formats.filter { $0.framesPerSecond == maxFps }
        let area: (ARConfiguration.VideoFormat) -> CGFloat = { $0.imageResolution.width * $0.imageResolution.height }
        let chosen = lowResCamera ? candidates.min { area($0) < area($1) } : candidates.max { area($0) < area($1) }
        if let chosen {
            config.videoFormat = chosen
        }
        videoFormatText = BodyTracker.describe(config.videoFormat)
        session.run(config, options: [.resetTracking, .removeExistingAnchors])
    }

    private static func describe(_ f: ARConfiguration.VideoFormat) -> String {
        "\(Int(f.imageResolution.width))x\(Int(f.imageResolution.height))@\(f.framesPerSecond)"
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

            if pcConnected && (relayStatus & RelayAck.statusWantPreview) != 0 {
                preview.maybeSend(frame: frame, body: body, jointIndices: jointIndices) { [weak self] chunk in
                    self?.sender.send(chunk)
                }
            }
        }

        if showVisuals { updateVisuals(tracked ? body : nil) }
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
        let nodes = visualRoot.childNodes
        guard let body else {
            for node in nodes where !node.isHidden { node.isHidden = true }
            return
        }
        let models = body.skeleton.jointModelTransforms
        for (i, idx) in jointIndices.enumerated() where i < nodes.count {
            guard idx != NSNotFound, idx < models.count else { continue }
            let m = body.transform * models[idx]
            nodes[i].simdPosition = SIMD3<Float>(m.columns.3.x, m.columns.3.y, m.columns.3.z)
            if nodes[i].isHidden { nodes[i].isHidden = false }
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
        if problemText != nil { problemText = nil }
        if discoveryText != nil { discoveryText = nil }
    }

    private func tickStatus() {
        let now = ProcessInfo.processInfo.systemUptime
        let connected = lastAckTime > 0 && now - lastAckTime < 2.0
        if connected != pcConnected { pcConnected = connected }
        if !connected { rttMs = 0 }
        let addr = LocalAddress.wifi()
        if addr != phoneAddress { phoneAddress = addr }

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
