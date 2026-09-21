import Foundation
import simd

/// fsaverage5 cortical mesh + ROI labels exported by `Conversion/export_brain_mesh.py`.
/// Vertex order matches TRIBE v2's prediction rows: left hemisphere 0..10241,
/// right hemisphere 10242..20483.
struct BrainMesh {
    let vertexCount: Int
    /// Pial and inflated positions, vertexCount × 3 floats (xyz).
    let pialPositions: [Float]
    let inflatedPositions: [Float]
    /// Triangle indices, 3 per face.
    let faces: [UInt32]
    /// Per-vertex ROI id: 0 = unassigned/medial wall, 1...roiNames.count.
    let roiLabels: [UInt8]
    let roiNames: [String]
    /// Per-vertex smooth normals over the pial surface (for lit rendering).
    let normals: [Float]
    let inflatedNormals: [Float]
    /// Grayscale anatomical underlay, independent of the prediction.
    let anatomyColors: [SIMD4<Float>]

    enum LoadError: LocalizedError {
        case notFound, badMagic, truncated
        var errorDescription: String? {
            switch self {
            case .notFound: "BrainMesh.bin not found in the app bundle. Run apple/Conversion/export_brain_mesh.py and rebuild."
            case .badMagic: "BrainMesh.bin has an invalid header."
            case .truncated: "BrainMesh.bin is truncated or corrupt."
            }
        }
    }

    static func loadFromBundle() throws -> BrainMesh {
        guard let url = Bundle.main.url(forResource: "BrainMesh", withExtension: "bin") else {
            throw LoadError.notFound
        }
        return try load(from: Data(contentsOf: url))
    }

    static func load(from data: Data) throws -> BrainMesh {
        var reader = ByteReader(data: data)
        guard reader.bytes(4) == [0x42, 0x4D, 0x53, 0x48] else { throw LoadError.badMagic } // "BMSH"
        guard let version = reader.uint32(), (version == 1 || version == 2),
              let nVertices = reader.uint32(),
              let nFaces = reader.uint32(),
              let nROIs = reader.uint32(),
              let pial = reader.floats(Int(nVertices) * 3),
              let inflated = reader.floats(Int(nVertices) * 3),
              let faces = reader.uint32s(Int(nFaces) * 3),
              let roiLabels = reader.bytes(Int(nVertices))
        else { throw LoadError.truncated }

        guard nVertices > 0, nVertices % 2 == 0,
              faces.allSatisfy({ $0 < nVertices }),
              roiLabels.allSatisfy({ $0 <= nROIs }),
              pial.allSatisfy(\.isFinite), inflated.allSatisfy(\.isFinite)
        else { throw LoadError.truncated }
        let sulcalDepth: [Float]
        if version == 2 {
            guard let depth = reader.floats(Int(nVertices)), depth.allSatisfy(\.isFinite)
            else { throw LoadError.truncated }
            sulcalDepth = depth
        } else {
            sulcalDepth = [Float](repeating: 0, count: Int(nVertices))
        }

        var roiNames: [String] = []
        for _ in 0..<Int(nROIs) {
            guard let length = reader.uint16(),
                  let nameData = reader.data(Int(length)),
                  let name = String(data: nameData, encoding: .utf8)
            else { throw LoadError.truncated }
            roiNames.append(name)
        }

        // FreeSurfer's inflated hemispheres are each centered on their own
        // origin; separate them laterally so both fit in one scene.
        var inflatedPositions = inflated
        let hemiCount = Int(nVertices) / 2
        for v in 0..<Int(nVertices) {
            inflatedPositions[v * 3] += v < hemiCount ? -48 : 48
        }

        return BrainMesh(
            vertexCount: Int(nVertices),
            pialPositions: pial,
            inflatedPositions: inflatedPositions,
            faces: faces,
            roiLabels: roiLabels,
            roiNames: roiNames,
            normals: Self.computeNormals(positions: pial, faces: faces, vertexCount: Int(nVertices)),
            inflatedNormals: Self.computeNormals(positions: inflatedPositions, faces: faces, vertexCount: Int(nVertices)),
            anatomyColors: sulcalDepth.map { depth in
                // FreeSurfer: positive sulcal depth is inside a fold. A soft
                // mapping keeps the gyri ivory and the sulci graphite.
                let gray: Float = 0.66 - 0.16 * tanh(depth * 0.8)
                return SIMD4(gray * 1.02, gray * 1.01, gray, 1)
            }
        )
    }

    private static func computeNormals(positions: [Float], faces: [UInt32], vertexCount: Int) -> [Float] {
        var normals = [Float](repeating: 0, count: vertexCount * 3)
        var i = 0
        while i + 2 < faces.count {
            let a = Int(faces[i]) * 3, b = Int(faces[i + 1]) * 3, c = Int(faces[i + 2]) * 3
            let ab = SIMD3(positions[b] - positions[a], positions[b + 1] - positions[a + 1], positions[b + 2] - positions[a + 2])
            let ac = SIMD3(positions[c] - positions[a], positions[c + 1] - positions[a + 1], positions[c + 2] - positions[a + 2])
            let n = cross(ab, ac)
            for v in [a, b, c] {
                normals[v] += n.x; normals[v + 1] += n.y; normals[v + 2] += n.z
            }
            i += 3
        }
        for v in 0..<vertexCount {
            let o = v * 3
            let accumulated = SIMD3(normals[o], normals[o + 1], normals[o + 2])
            let n = length_squared(accumulated) > 1e-12 ? normalize(accumulated) : SIMD3<Float>(0, 0, 1)
            normals[o] = n.x; normals[o + 1] = n.y; normals[o + 2] = n.z
        }
        return normals
    }
}

private struct ByteReader {
    let data: Data
    var offset = 0

    init(data: Data) { self.data = data }

    mutating func bytes(_ count: Int) -> [UInt8]? {
        guard offset + count <= data.count else { return nil }
        defer { offset += count }
        return Array(data[offset..<offset + count])
    }

    mutating func data(_ count: Int) -> Data? {
        guard offset + count <= data.count else { return nil }
        defer { offset += count }
        return data.subdata(in: offset..<offset + count)
    }

    mutating func uint16() -> UInt16? {
        guard let b = bytes(2) else { return nil }
        return UInt16(b[0]) | UInt16(b[1]) << 8
    }

    mutating func uint32() -> UInt32? {
        guard let b = bytes(4) else { return nil }
        return UInt32(b[0]) | UInt32(b[1]) << 8 | UInt32(b[2]) << 16 | UInt32(b[3]) << 24
    }

    mutating func floats(_ count: Int) -> [Float]? {
        guard let raw = data(count * 4) else { return nil }
        return raw.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
    }

    mutating func uint32s(_ count: Int) -> [UInt32]? {
        guard let raw = data(count * 4) else { return nil }
        return raw.withUnsafeBytes { Array($0.bindMemory(to: UInt32.self)) }
    }
}
