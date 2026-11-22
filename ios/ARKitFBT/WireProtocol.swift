import ARKit
import Foundation
import simd

/// iPhone -> PC 패킷 형식. relay/src/arkit_fbt_relay/protocol.py 와 반드시 같아야 한다.
///
/// 헤더 (84 bytes, little-endian)
///   0  magic "AFBT"            4s
///   4  version                 u8
///   5  flags (bit0 = 몸 인식)   u8
///   6  joint count             u16
///   8  seq                     u32
///  12  frame timestamp (s)     f64   ARFrame.timestamp (기기 부팅 후 경과 시간)
///  20  estimatedScaleFactor    f32
///  24  joint tracked mask      u32   bit i = JointSet.names[i] 가 실제로 추적됨
///  28  camera pose             7*f32 (px, py, pz, qx, qy, qz, qw) ARKit 월드
///  56  body anchor pose        7*f32 (몸 미인식 시 NaN)
///  84  joints                  count * 7*f32 (ARKit 월드, JointSet.names 순서)
enum WireProtocol {
    static let frameMagic: [UInt8] = Array("AFBT".utf8)
    static let ackMagic: [UInt8] = Array("AFBA".utf8)
    static let version: UInt8 = 1
    static let defaultPort = 39570
}

enum JointSet {
    static let names: [String] = [
        "hips_joint", "spine_4_joint", "spine_7_joint", "neck_1_joint", "head_joint",
        "left_shoulder_1_joint", "left_arm_joint", "left_forearm_joint", "left_hand_joint",
        "right_shoulder_1_joint", "right_arm_joint", "right_forearm_joint", "right_hand_joint",
        "left_upLeg_joint", "left_leg_joint", "left_foot_joint", "left_toes_joint",
        "right_upLeg_joint", "right_leg_joint", "right_foot_joint", "right_toes_joint",
    ]
}

struct PacketWriter {
    private(set) var data = Data()

    init(capacity: Int) {
        data.reserveCapacity(capacity)
    }

    mutating func bytes(_ b: [UInt8]) { data.append(contentsOf: b) }
    mutating func u8(_ v: UInt8) { data.append(v) }
    mutating func u16(_ v: UInt16) { appendLE(v) }
    mutating func u32(_ v: UInt32) { appendLE(v) }
    mutating func f32(_ v: Float) { appendLE(v.bitPattern) }
    mutating func f64(_ v: Double) { appendLE(v.bitPattern) }

    mutating func pose(_ m: simd_float4x4) {
        let t = m.columns.3
        f32(t.x); f32(t.y); f32(t.z)
        let q = PacketWriter.rotation(of: m)
        f32(q.imag.x); f32(q.imag.y); f32(q.imag.z); f32(q.real)
    }

    mutating func invalidPose() {
        for _ in 0..<7 { f32(.nan) }
    }

    private mutating func appendLE<T: FixedWidthInteger>(_ v: T) {
        var le = v.littleEndian
        withUnsafeBytes(of: &le) { data.append(contentsOf: $0) }
    }

    /// 스케일이 섞여 있어도 회전만 뽑아낸다.
    static func rotation(of m: simd_float4x4) -> simd_quatf {
        let c0 = simd_normalize(SIMD3<Float>(m.columns.0.x, m.columns.0.y, m.columns.0.z))
        let c1 = simd_normalize(SIMD3<Float>(m.columns.1.x, m.columns.1.y, m.columns.1.z))
        let c2 = simd_normalize(SIMD3<Float>(m.columns.2.x, m.columns.2.y, m.columns.2.z))
        return simd_normalize(simd_quatf(simd_float3x3(columns: (c0, c1, c2))))
    }
}

enum PacketBuilder {
    static func frame(seq: UInt32,
                      timestamp: TimeInterval,
                      camera: simd_float4x4,
                      body: ARBodyAnchor?,
                      jointIndices: [Int]) -> Data {
        let trackedBody: ARBodyAnchor? = (body?.isTracked ?? false) ? body : nil
        let count = trackedBody == nil ? 0 : jointIndices.count
        var w = PacketWriter(capacity: 84 + count * 28)

        w.bytes(WireProtocol.frameMagic)
        w.u8(WireProtocol.version)
        w.u8(trackedBody == nil ? 0 : 1)
        w.u16(UInt16(count))
        w.u32(seq)
        w.f64(timestamp)
        w.f32(Float(trackedBody?.estimatedScaleFactor ?? 1))

        var mask: UInt32 = 0
        if let b = trackedBody {
            for (i, idx) in jointIndices.enumerated() where i < 32 && idx != NSNotFound {
                if b.skeleton.isJointTracked(idx) {
                    mask |= UInt32(1) << UInt32(i)
                }
            }
        }
        w.u32(mask)

        w.pose(camera)
        if let b = trackedBody {
            w.pose(b.transform)
            let models = b.skeleton.jointModelTransforms
            for idx in jointIndices {
                if idx != NSNotFound && idx < models.count {
                    w.pose(b.transform * models[idx])
                } else {
                    w.invalidPose()
                }
            }
        } else {
            w.invalidPose()
        }
        return w.data
    }
}

/// PC -> iPhone 응답 (20 bytes): "AFBA", version u8, status u8, reserved u16, seq u32, echo timestamp f64
struct RelayAck {
    let status: UInt8
    let seq: UInt32
    let timestamp: Double

    static let statusOscSending: UInt8 = 0x01
    static let statusYawAligned: UInt8 = 0x02
    static let statusDryRun: UInt8 = 0x04
    static let statusWantPreview: UInt8 = 0x08

    init?(_ raw: Data) {
        let d = Data(raw)
        guard d.count >= 20, Array(d.prefix(4)) == WireProtocol.ackMagic else { return nil }
        status = d[5]
        seq = d.withUnsafeBytes { UInt32(littleEndian: $0.loadUnaligned(fromByteOffset: 8, as: UInt32.self)) }
        let bits = d.withUnsafeBytes { UInt64(littleEndian: $0.loadUnaligned(fromByteOffset: 12, as: UInt64.self)) }
        timestamp = Double(bitPattern: bits)
    }
}
