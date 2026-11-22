import ARKit
import CoreImage
import Foundation
import simd

/// PC 창에 보여 줄 카메라 미리보기 (저해상도 JPEG + 관절 2D 좌표).
/// PC가 ack 상태에 statusWantPreview 를 켰을 때만 보낸다.
///
/// 한 장의 미리보기(blob)를 조각(chunk)으로 나눠 보낸다. relay/src/arkit_fbt_relay/preview.py 와 같아야 한다.
///
/// chunk (16 bytes 헤더 + payload ≤ 1100)
///   0  magic "AFBC"   4s
///   4  version        u8
///   5  reserved       u8
///   6  chunk index    u16
///   8  chunk count    u16
///  10  reserved       u16
///  12  frame id       u32
///
/// blob
///   0  jpeg width     u16
///   2  jpeg height    u16
///   4  joint count    u16
///   6  reserved       u16
///   8  timestamp      f64
///  16  joints         count * (u f32, v f32)  원본 이미지(센서 방향) 기준 0..1, 없으면 NaN
///   …  jpeg bytes
final class PreviewSender {
    static let chunkMagic: [UInt8] = Array("AFBC".utf8)
    static let chunkPayload = 1100

    var maxWidth: CGFloat = 384
    var interval: TimeInterval = 1.0 / 8

    private let queue = DispatchQueue(label: "arkitfbt.preview", qos: .utility)
    private let context = CIContext(options: [.cacheIntermediates: false])
    private let colorSpace = CGColorSpace(name: CGColorSpace.sRGB)!
    private var busy = false  // 메인 스레드에서만 읽고 쓴다
    private var lastTime: TimeInterval = 0
    private var frameId: UInt32 = 0

    /// 메인 스레드에서 매 ARFrame마다 호출한다. 간격이 안 됐거나 이전 인코딩 중이면 건너뛴다.
    func maybeSend(frame: ARFrame, body: ARBodyAnchor?, jointIndices: [Int], send: @escaping (Data) -> Void) {
        guard !busy, frame.timestamp - lastTime >= interval else { return }
        busy = true
        lastTime = frame.timestamp
        frameId &+= 1
        let id = frameId

        let joints = PreviewSender.project(frame: frame, body: body, jointIndices: jointIndices)
        let pixelBuffer = frame.capturedImage
        let timestamp = frame.timestamp

        queue.async { [weak self] in
            guard let self else { return }
            let chunks = self.encode(pixelBuffer: pixelBuffer, joints: joints, timestamp: timestamp, id: id)
            DispatchQueue.main.async {
                self.busy = false
                chunks.forEach(send)
            }
        }
    }

    private func encode(pixelBuffer: CVPixelBuffer, joints: [SIMD2<Float>], timestamp: TimeInterval, id: UInt32) -> [Data] {
        let image = CIImage(cvPixelBuffer: pixelBuffer)
        let scale = min(1, maxWidth / image.extent.width)
        let scaled = image.transformed(by: CGAffineTransform(scaleX: scale, y: scale))
        var jpeg: Data?
        for quality in [0.5, 0.3] as [CGFloat] {
            let options = [kCGImageDestinationLossyCompressionQuality as CIImageRepresentationOption: quality]
            jpeg = context.jpegRepresentation(of: scaled, colorSpace: colorSpace, options: options)
            if let j = jpeg, j.count <= 60_000 { break }
        }
        guard let jpeg else { return [] }

        var w = PacketWriter(capacity: 16 + joints.count * 8 + jpeg.count)
        w.u16(UInt16(clamping: Int(scaled.extent.width.rounded())))
        w.u16(UInt16(clamping: Int(scaled.extent.height.rounded())))
        w.u16(UInt16(joints.count))
        w.u16(0)
        w.f64(timestamp)
        for p in joints {
            w.f32(p.x)
            w.f32(p.y)
        }
        var blob = w.data
        blob.append(jpeg)

        let size = PreviewSender.chunkPayload
        let count = (blob.count + size - 1) / size
        guard count <= Int(UInt16.max) else { return [] }
        var chunks: [Data] = []
        chunks.reserveCapacity(count)
        for i in 0..<count {
            var c = PacketWriter(capacity: 16 + size)
            c.bytes(PreviewSender.chunkMagic)
            c.u8(WireProtocol.version)
            c.u8(0)
            c.u16(UInt16(i))
            c.u16(UInt16(count))
            c.u16(0)
            c.u32(id)
            var data = c.data
            let start = blob.startIndex + i * size
            data.append(blob[start..<min(start + size, blob.endIndex)])
            chunks.append(data)
        }
        return chunks
    }

    /// 관절을 원본 카메라 이미지(센서 방향, landscapeRight) 좌표 0..1 로 투영한다.
    static func project(frame: ARFrame, body: ARBodyAnchor?, jointIndices: [Int]) -> [SIMD2<Float>] {
        let missing = SIMD2<Float>(.nan, .nan)
        guard let body, body.isTracked else { return [] }
        let camera = frame.camera
        let size = camera.imageResolution
        let worldToCamera = simd_inverse(camera.transform)
        let models = body.skeleton.jointModelTransforms
        return jointIndices.map { idx in
            guard idx != NSNotFound, idx < models.count else { return missing }
            let m = body.transform * models[idx]
            let world = SIMD3<Float>(m.columns.3.x, m.columns.3.y, m.columns.3.z)
            let local = worldToCamera * SIMD4<Float>(world.x, world.y, world.z, 1)
            guard local.z < 0 else { return missing }  // 카메라 뒤
            let p = camera.projectPoint(world, orientation: .landscapeRight, viewportSize: size)
            return SIMD2<Float>(Float(p.x / size.width), Float(p.y / size.height))
        }
    }
}
