import AppKit
import CoreML
import Metal
import SceneKit
import SwiftUI

/// Run with `make verify-brain`. The snapshots use a deterministic synthetic
/// field solely to inspect spatial registration, shading, and color blending.
@main
struct BrainRenderingCheck {
    @MainActor static func main() throws {
        let meshData = try Data(contentsOf: URL(fileURLWithPath: "Models/BrainMesh.bin"))
        let mesh = try BrainMesh.load(from: meshData)
        precondition(mesh.vertexCount == 20484)
        precondition(mesh.normals != mesh.inflatedNormals, "Inflated surface must have its own normals")
        for normals in [mesh.normals, mesh.inflatedNormals] {
            for v in 0..<mesh.vertexCount {
                let n = SIMD3(normals[v * 3], normals[v * 3 + 1], normals[v * 3 + 2])
                precondition(abs(simd_length(n) - 1) < 0.001)
            }
        }
        // Malformed files fail cleanly; v1 anatomy still opens without sulcal data.
        do { _ = try BrainMesh.load(from: meshData.prefix(40)); fatalError("Accepted truncated mesh") }
        catch is BrainMesh.LoadError {}
        var legacy = meshData
        let labelEnd = 20 + mesh.vertexCount * 24 + mesh.faces.count * 4 + mesh.vertexCount
        legacy.removeSubrange(labelEnd..<(labelEnd + mesh.vertexCount * 4))
        legacy[4] = 1
        let legacyMesh = try BrainMesh.load(from: legacy)
        precondition(legacyMesh.faces == mesh.faces)

        func luminance(_ color: SIMD4<Float>) -> Float {
            func linear(_ x: Float) -> Float { x <= 0.04045 ? x / 12.92 : pow((x + 0.055) / 1.055, 2.4) }
            return 0.2126 * linear(color.x) + 0.7152 * linear(color.y) + 0.0722 * linear(color.z)
        }
        var previous = ActivityPalette.timeline(0)
        for i in 1...1000 {
            let color = ActivityPalette.timeline(Float(i) / 1000)
            precondition(luminance(color) > luminance(previous), "Intensity must increase brightness monotonically")
            precondition(simd_distance(color, previous) < 0.01, "Gradient has a visible discontinuity")
            previous = color
        }
        for value: Float in [-1, 0, 0.3, 0.5, 1, 2, .nan, .infinity] {
            let color = ActivityPalette.timeline(value)
            precondition((0...1).contains(color.x) && (0...1).contains(color.y) && (0...1).contains(color.z))
        }
        let neutral = mesh.anatomyColors[0]
        precondition(ActivityPalette.activation(-1, anatomy: neutral) == neutral)
        precondition(ActivityPalette.activation(0, anatomy: neutral) == neutral)

        var values = [Float]()
        for v in 0..<mesh.vertexCount {
            let p = SIMD3(mesh.pialPositions[v * 3], mesh.pialPositions[v * 3 + 1], mesh.pialPositions[v * 3 + 2])
            func hotspot(_ center: SIMD3<Float>, _ width: Float) -> Float {
                exp(-simd_length_squared(p - center) / (2 * width * width))
            }
            let value = max(hotspot(SIMD3(-30, -86, 8), 20), hotspot(SIMD3(40, -67, 25), 16) * 0.82)
            values.append(contentsOf: [value, value * 0.5])
        }
        let activity = BrainActivity(values: values, vertexCount: mesh.vertexCount, trCount: 2)
        let colors = activity.normalizedColors(predictionTime: 0.5, mesh: mesh)
        precondition(activity.values == values, "Display must not mutate model predictions")
        for v in 0..<mesh.vertexCount where mesh.roiLabels[v] == 0 {
            precondition(colors[v] == mesh.anatomyColors[v], "Medial wall received activation")
        }
        let v = (0..<mesh.vertexCount).first { mesh.roiLabels[$0] != 0 && values[$0 * 2] > 0.5 }!
        let expected = ActivityPalette.activation(values[v * 2] * 0.75 / activity.positiveDisplayScale,
                                                  anatomy: mesh.anatomyColors[v])
        precondition(simd_distance(colors[v], expected) < 0.00001, "Fractional TR interpolation failed")
        precondition(activity.normalizedColors(predictionTime: -10, mesh: mesh)
                     == activity.normalizedColors(predictionTime: 0, mesh: mesh))
        let outlier = BrainActivity(values: Array(repeating: Float(1), count: 1000) + [10000], vertexCount: 1, trCount: 1001)
        precondition(outlier.positiveDisplayScale == 1, "Outlier flattened the display scale")

        let output = URL(fileURLWithPath: "build/brain-review", isDirectory: true)
        try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
        func modalityField(_ center: SIMD3<Float>) -> BrainActivity {
            let values: [Float] = (0..<mesh.vertexCount).map { v in
                let p = SIMD3(mesh.pialPositions[v * 3], mesh.pialPositions[v * 3 + 1], mesh.pialPositions[v * 3 + 2])
                return exp(-simd_length_squared(p - center) / (2 * 45 * 45))
            }
            return BrainActivity(values: values, vertexCount: mesh.vertexCount, trCount: 1)
        }
        let modalities = MultimodalActivity(text: modalityField(SIMD3(-45, 30, 10)),
                                           audio: modalityField(SIMD3(-55, -15, 0)),
                                           video: modalityField(SIMD3(-30, -85, 15)))
        let multimodalColors = modalities.colors(predictionTime: 0, mesh: mesh)
        guard let device = MTLCreateSystemDefaultDevice() else { fatalError("Metal unavailable for visual QA") }
        for (name, inflated, field) in [("folded", false, colors), ("neutral", false, mesh.anatomyColors),
                                        ("inflated", true, colors), ("multimodal-folded", false, multimodalColors),
                                        ("multimodal-inflated", true, multimodalColors)] {
            let setup = BrainView.makeScene(mesh: mesh, colors: field, inflated: inflated)
            let renderer = SCNRenderer(device: device, options: nil)
            renderer.scene = setup.scene
            renderer.pointOfView = setup.camera
            let image = renderer.snapshot(atTime: 0, with: CGSize(width: 1000, height: 1000), antialiasingMode: .multisampling4X)
            guard let tiff = image.tiffRepresentation,
                  let png = NSBitmapImageRep(data: tiff)?.representation(using: .png, properties: [:]) else {
                fatalError("Snapshot failed")
            }
            try png.write(to: output.appendingPathComponent("\(name).png"))
        }
        let legend = ImageRenderer(content: MultimodalLegend().padding(20)
            .background(Color(white: 0.035)).environment(\.colorScheme, .dark))
        legend.scale = 3
        if let image = legend.nsImage, let tiff = image.tiffRepresentation,
           let png = NSBitmapImageRep(data: tiff)?.representation(using: .png, properties: [:]) {
            try png.write(to: output.appendingPathComponent("multimodal-legend.png"))
        }
        let track = HeatTrack(name: "Gradient", values: [0, 0.15, 0.45, 0.95, 0.6, 0.3, 0.4, 0.75, 1])
        let timeline = VStack(alignment: .leading, spacing: 16) {
            HStack {
                Text("ACTIVITY OVER TIME").font(.caption.weight(.semibold))
                Text("Relative within each row").font(.caption2).foregroundStyle(.secondary)
                Spacer()
                ActivityLegend()
            }
            TrackBand(track: track).frame(height: 26).clipShape(RoundedRectangle(cornerRadius: 4))
            TrackBand(track: HeatTrack(name: "Full ramp", values: [0, 1]))
                .frame(height: 26).clipShape(RoundedRectangle(cornerRadius: 4))
        }
        .padding(20).frame(width: 1000).background(Color(white: 0.07)).environment(\.colorScheme, .dark)
        let renderer = ImageRenderer(content: timeline)
        renderer.scale = 2
        if let image = renderer.nsImage, let tiff = image.tiffRepresentation,
           let png = NSBitmapImageRep(data: tiff)?.representation(using: .png, properties: [:]) {
            try png.write(to: output.appendingPathComponent("timeline.png"))
        }
        print("PASS: mesh compatibility, normals, color continuity, monotonic luminance, masking, interpolation, stable scaling")
        print("Rendered activity, multimodal, legend, and timeline previews to build/brain-review/")
    }
}
