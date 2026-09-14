import Accelerate
import CoreML
import Foundation
import simd

/// A (vertexCount × trCount) matrix of predicted brain activity — copied out
/// of the FmriEncoder's `predictions` MLMultiArray, assembled by the analysis
/// pipeline, or loaded from the on-disk cache — with derived views used by the
/// timeline tracks and the 3D brain surface.
struct BrainActivity {
    let vertexCount: Int
    let trCount: Int
    /// Row-major [vertex][tr].
    private(set) var values: [Float]
    let lo: Float
    let hi: Float
    /// Fixed over the entire clip; robust to isolated prediction outliers.
    let positiveDisplayScale: Float

    init?(predictions: MLMultiArray) {
        guard predictions.shape.count == 3 else { return nil }
        vertexCount = predictions.shape[1].intValue
        trCount = predictions.shape[2].intValue
        let total = vertexCount * trCount
        // Index via strides — CoreML may pad output rows, so the data is not
        // necessarily contiguous [vertex][tr].
        let vertexStride = predictions.strides[1].intValue
        let trStride = predictions.strides[2].intValue
        // Locals so the closures below don't capture the not-yet-fully-
        // initialized self.
        let vc = vertexCount, tc = trCount, vs = vertexStride, ts = trStride

        switch predictions.dataType {
        case .float32:
            let ptr = predictions.dataPointer.assumingMemoryBound(to: Float.self)
            let copied = [Float](unsafeUninitializedCapacity: total) { buffer, initializedCount in
                for v in 0..<vc {
                    let srcRow = ptr + v * vs
                    let dstRow = buffer.baseAddress! + v * tc
                    for t in 0..<tc { dstRow[t] = srcRow[t * ts] }
                }
                initializedCount = total
            }
            values = copied
        case .float16:
            let ptr = predictions.dataPointer.assumingMemoryBound(to: UInt16.self)
            let copied = [Float](unsafeUninitializedCapacity: total) { buffer, initializedCount in
                for v in 0..<vc {
                    let srcRow = ptr + v * vs
                    let dstRow = buffer.baseAddress! + v * tc
                    for t in 0..<tc {
                        dstRow[t] = Float(Float16(bitPattern: srcRow[t * ts]))
                    }
                }
                initializedCount = total
            }
            values = copied
        default:
            return nil
        }

        var minV: Float = 0, maxV: Float = 0
        vDSP_minv(values, 1, &minV, vDSP_Length(total))
        vDSP_maxv(values, 1, &maxV, vDSP_Length(total))
        lo = minV
        hi = maxV
        positiveDisplayScale = Self.displayScale(values)
    }

    /// Builds from a raw row-major [vertex][tr] Float32 buffer, e.g. loaded
    /// from the on-disk analysis cache (AnalysisCache).
    init(values: [Float], vertexCount: Int, trCount: Int) {
        precondition(values.count == vertexCount * trCount)
        self.vertexCount = vertexCount
        self.trCount = trCount
        self.values = values

        var minV: Float = 0, maxV: Float = 0
        vDSP_minv(values, 1, &minV, vDSP_Length(values.count))
        vDSP_maxv(values, 1, &maxV, vDSP_Length(values.count))
        lo = minV
        hi = maxV
        positiveDisplayScale = Self.displayScale(values)
    }

    /// Per-vertex values at one TR (strided column of the matrix).
    func trSlice(_ tr: Int) -> [Float] {
        let t = min(max(tr, 0), trCount - 1)
        var slice = [Float](repeating: 0, count: vertexCount)
        for v in 0..<vertexCount {
            slice[v] = values[v * trCount + t]
        }
        return slice
    }

    /// Interpolate predictions before applying the color map. A stable scale
    /// avoids frame-to-frame pumping; negative responses stay anatomical gray.
    func normalizedColors(predictionTime: Double, mesh: BrainMesh) -> [SIMD4<Float>] {
        guard mesh.vertexCount == vertexCount, trCount > 0 else { return mesh.anatomyColors }
        let time = predictionTime.isFinite ? min(max(predictionTime, 0), Double(trCount - 1)) : 0
        let first = Int(time), next = min(first + 1, trCount - 1)
        let fraction = Float(time - Double(first))
        return (0..<vertexCount).map { v in
            let anatomy = mesh.anatomyColors[v]
            guard mesh.roiLabels[v] != 0 else { return anatomy }
            let a = values[v * trCount + first], b = values[v * trCount + next]
            let value = a + (b - a) * fraction
            return ActivityPalette.activation(value / positiveDisplayScale, anatomy: anatomy)
        }
    }

    private static func displayScale(_ values: [Float]) -> Float {
        // Bound sorting cost for long videos with a deterministic sample.
        let step = max(1, values.count / 100_000)
        let positive = stride(from: 0, to: values.count, by: step)
            .map { values[$0] }.filter { $0.isFinite && $0 > 0 }.sorted()
        guard !positive.isEmpty else { return 1 }
        return max(positive[Int(Double(positive.count - 1) * 0.98)], 1e-9)
    }

    /// Overall stimulation: RMS across vertices per TR.
    func overallTrack() -> [Double] {
        (0..<trCount).map { t in
            var sumSquares = 0.0
            for v in 0..<vertexCount {
                let x = Double(values[v * trCount + t])
                sumSquares += x * x
            }
            return (sumSquares / Double(vertexCount)).squareRoot()
        }
    }

    /// One track per ROI: mean activation across the ROI's vertices per TR.
    func roiTracks(mesh: BrainMesh) -> [HeatTrack] {
        guard mesh.vertexCount == vertexCount else { return [] }
        var members: [[Int]] = Array(repeating: [], count: mesh.roiNames.count + 1)
        for v in 0..<vertexCount {
            let roi = Int(mesh.roiLabels[v])
            if roi >= 1 { members[roi].append(v) }
        }
        return mesh.roiNames.enumerated().compactMap { index, name in
            let vs = members[index + 1]
            guard !vs.isEmpty else { return nil }
            let valuesPerTR: [Double] = (0..<trCount).map { t in
                var sum = 0.0
                for v in vs { sum += Double(values[v * trCount + t]) }
                return sum / Double(vs.count)
            }
            return HeatTrack(name: name, values: valuesPerTR)
        }
    }
}
