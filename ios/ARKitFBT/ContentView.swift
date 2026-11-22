import RealityKit
import SwiftUI

struct ARViewContainer: UIViewRepresentable {
    let tracker: BodyTracker

    func makeUIView(context: Context) -> ARView {
        let view = ARView(frame: .zero, cameraMode: .ar, automaticallyConfigureSession: false)
        tracker.attach(view)
        return view
    }

    func updateUIView(_ uiView: ARView, context: Context) {}
}

struct ContentView: View {
    @StateObject private var tracker = BodyTracker()
    @AppStorage("relayHost") private var host = ""
    @AppStorage("relayPort") private var port = WireProtocol.defaultPort
    @AppStorage("sendEnabled") private var sendEnabled = true
    @State private var showSettings = false

    var body: some View {
        if tracker.isSupported {
            ZStack(alignment: .top) {
                ARViewContainer(tracker: tracker)
                    .ignoresSafeArea()

                VStack(spacing: 8) {
                    statusBar
                    if showSettings || host.isEmpty {
                        settingsPanel
                    }
                    Spacer()
                    if !tracker.bodyTracked {
                        hint
                    }
                }
                .padding(12)
            }
            .onAppear {
                UIApplication.shared.isIdleTimerDisabled = true
                apply()
            }
            .onChange(of: host) { _ in apply() }
            .onChange(of: port) { _ in apply() }
            .onChange(of: sendEnabled) { _ in apply() }
        } else {
            VStack(spacing: 12) {
                Image(systemName: "exclamationmark.triangle").font(.largeTitle)
                Text("이 기기는 ARKit 바디 트래킹을 지원하지 않습니다.")
                Text("A12 칩 이상(iPhone XS / XR 이후)이 필요합니다.").font(.footnote)
            }
            .padding()
        }
    }

    private func apply() {
        tracker.configure(host: host, port: port, enabled: sendEnabled)
    }

    private var statusBar: some View {
        HStack(spacing: 10) {
            indicator(tracker.pcConnected, on: "PC 연결", off: "PC 없음")
            indicator(tracker.bodyTracked, on: "몸 인식", off: "몸 없음")
            if tracker.pcConnected && (tracker.relayStatus & RelayAck.statusOscSending) != 0 {
                Text("VRChat ▶").font(.caption.bold()).foregroundColor(.green)
            }
            Spacer(minLength: 4)
            VStack(alignment: .trailing, spacing: 1) {
                Text(String(format: "%.0f fps / 몸 %.0f", tracker.fps, tracker.bodyFps))
                if tracker.pcConnected {
                    Text(String(format: "왕복 %.0f ms", tracker.rttMs))
                }
                Text("발열 \(tracker.thermalText)")
            }
            .font(.caption2.monospacedDigit())
            Button {
                showSettings.toggle()
            } label: {
                Image(systemName: "gearshape.fill").font(.title3)
            }
        }
        .padding(10)
        .foregroundColor(.white)
        .background(.black.opacity(0.55), in: RoundedRectangle(cornerRadius: 12))
    }

    private func indicator(_ ok: Bool, on: String, off: String) -> some View {
        HStack(spacing: 4) {
            Circle().fill(ok ? Color.green : Color.red).frame(width: 9, height: 9)
            Text(ok ? on : off).font(.caption.bold())
        }
    }

    private var settingsPanel: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("PC 중계 프로그램").font(.headline)
            HStack {
                TextField("PC IP (예: 192.168.0.2)", text: $host)
                    .keyboardType(.numbersAndPunctuation)
                    .autocorrectionDisabled()
                    .textInputAutocapitalization(.never)
                    .textFieldStyle(.roundedBorder)
                TextField("포트", value: $port, format: .number.grouping(.never))
                    .keyboardType(.numberPad)
                    .textFieldStyle(.roundedBorder)
                    .frame(width: 80)
            }
            Toggle("데이터 전송", isOn: $sendEnabled)
            Text("상태: \(tracker.connectionText)").font(.caption)
            if !tracker.missingJoints.isEmpty {
                Text("찾지 못한 관절: \(tracker.missingJoints.joined(separator: ", "))")
                    .font(.caption2)
                    .foregroundColor(.orange)
            }
            HStack {
                Button("AR 다시 시작") { tracker.restartSession() }
                Spacer()
                Button("닫기") {
                    showSettings = false
                    UIApplication.shared.sendAction(#selector(UIResponder.resignFirstResponder), to: nil, from: nil, for: nil)
                }
            }
        }
        .padding(12)
        .foregroundColor(.white)
        .background(.black.opacity(0.7), in: RoundedRectangle(cornerRadius: 12))
    }

    private var hint: some View {
        Text("2.5~3m 떨어져서 머리부터 발끝까지 화면에 들어오게 서 주세요")
            .font(.callout.bold())
            .multilineTextAlignment(.center)
            .padding(10)
            .foregroundColor(.white)
            .background(.black.opacity(0.55), in: RoundedRectangle(cornerRadius: 10))
    }
}
