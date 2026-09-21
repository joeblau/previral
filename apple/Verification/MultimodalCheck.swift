import CoreML
import Foundation
import simd

@main
struct MultimodalCheck {
    static func main() throws {
        let mesh = try BrainMesh.load(from: Data(contentsOf: URL(fileURLWithPath: "Models/BrainMesh.bin")))
        func field(_ values: [Float]) -> BrainActivity {
            BrainActivity(values: (0..<mesh.vertexCount).flatMap { _ in values }, vertexCount: mesh.vertexCount, trCount: values.count)
        }
        let zero = field([0, 0]), signal = field([1, 0])
        let neutral = SIMD4<Float>(0.6, 0.6, 0.6, 1)
        for (inputs, expected) in [(SIMD3<Float>(1, 0, 0), SIMD4<Float>(1, 0, 0, 1)),
                                   (SIMD3(0, 1, 0), SIMD4(0, 1, 0, 1)),
                                   (SIMD3(0, 0, 1), SIMD4(0, 0, 1, 1)),
                                   (SIMD3(1, 1, 1), SIMD4(1, 1, 1, 1))] {
            precondition(ActivityPalette.multimodal(text: inputs.x, audio: inputs.y, video: inputs.z, anatomy: neutral) == expected)
        }
        precondition(ActivityPalette.multimodal(text: .nan, audio: -1, video: .infinity, anatomy: neutral) == neutral)
        precondition(MultimodalActivity.response(signal, relativeTo: signal).hi == 0, "Absent inputs must have no response")
        precondition(MultimodalActivity.response(zero, relativeTo: signal).hi == 0, "Negative differences must stay gray")
        let channels = MultimodalActivity(text: signal, audio: zero, video: field([0, 1]))
        let colors = channels.colors(predictionTime: 0.5, mesh: mesh)
        for v in 0..<mesh.vertexCount {
            let expected = mesh.roiLabels[v] == 0 ? mesh.anatomyColors[v] : SIMD4<Float>(0.5, 0, 0.5, 1)
            precondition(simd_distance(colors[v], expected) < 0.00001, "Interpolation or medial-wall masking failed")
        }
        precondition(channels.colors(predictionTime: -10, mesh: mesh) == channels.colors(predictionTime: 0, mesh: mesh))
        precondition(channels.colors(predictionTime: 1000, mesh: mesh) == channels.colors(predictionTime: 1, mesh: mesh))
        let weak = MultimodalActivity(text: field([0.01, 0.01]), audio: zero, video: zero)
        precondition(abs(weak.displayScale - 0.01) < 0.00001, "Absent modalities must not flatten the scale")

        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let videoURL = directory.appendingPathComponent("fixture.mov")
        try Data([0]).write(to: videoURL)
        let result = AnalysisResult(activity: signal, notes: ["No speech detected"], multimodal: channels)
        try AnalysisCache.save(result: result, videoURL: videoURL, duration: 2)
        let cached = AnalysisCache.loadFresh(for: videoURL)!
        precondition(cached.activity.values == signal.values && cached.notes == result.notes)
        precondition(cached.multimodal?.text.values == signal.values && cached.multimodal?.audio.values == zero.values)
        precondition(cached.multimodal?.video.values == channels.video.values)
        let binURL = AnalysisCache.binURL(for: videoURL)
        let data = try Data(contentsOf: binURL)
        try data.dropLast().write(to: binURL)
        precondition(AnalysisCache.loadFresh(for: videoURL) == nil, "Truncated modality cache accepted")
        // Version 1 remains readable, with no invented modality data.
        try signal.values.withUnsafeBufferPointer { try Data(buffer: $0).write(to: binURL) }
        let metadata = AnalysisCache.Metadata(formatVersion: 1, videoPath: videoURL.path, duration: 2,
                                             trCount: 2, vertexCount: mesh.vertexCount, created: Date())
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        try encoder.encode(metadata).write(to: AnalysisCache.jsonURL(for: videoURL))
        let legacy = AnalysisCache.loadFresh(for: videoURL)!
        precondition(legacy.activity.values == signal.values && legacy.multimodal == nil)
        try FileManager.default.setAttributes([.modificationDate: Date().addingTimeInterval(10)], ofItemAtPath: videoURL.path)
        precondition(AnalysisCache.loadFresh(for: videoURL) == nil, "Stale cache accepted")

        // Optional real Core ML check: pass the compiled FmriEncoder.mlmodelc.
        if CommandLine.arguments.count > 1 {
            let head = try FmriEncoderModel(contentsOfCompiledModel: URL(fileURLWithPath: CommandLine.arguments[1]))
            func zeros(_ shape: [NSNumber]) throws -> MLMultiArray {
                let array = try MLMultiArray(shape: shape, dataType: .float32)
                memset(array.dataPointer, 0, array.count * MemoryLayout<Float>.size)
                return array
            }
            let textZero = try zeros(FmriEncoderModel.textShape)
            let audioZero = try zeros(FmriEncoderModel.audioShape)
            let videoZero = try zeros(FmriEncoderModel.videoShape)
            let baseline = try BrainActivity(predictions: head.predict(text: textZero, audio: audioZero, video: videoZero))!
            var responses: [BrainActivity] = []
            for modality in 0..<3 {
                let text = try modality == 0 ? FmriEncoderModel.randomFeatures(shape: FmriEncoderModel.textShape) : textZero
                let audio = try modality == 1 ? FmriEncoderModel.randomFeatures(shape: FmriEncoderModel.audioShape) : audioZero
                let video = try modality == 2 ? FmriEncoderModel.randomFeatures(shape: FmriEncoderModel.videoShape) : videoZero
                let prediction = try BrainActivity(predictions: head.predict(text: text, audio: audio, video: video))!
                let response = MultimodalActivity.response(prediction, relativeTo: baseline)
                precondition(response.values.allSatisfy(\.isFinite) && response.hi > 0)
                responses.append(response)
            }
            precondition(responses[0].values != responses[1].values && responses[1].values != responses[2].values)
            print("PASS: real Core ML single-input predictions are finite, nonzero, and distinct")
        }
        print("PASS: RGB channels, baseline subtraction, missing inputs, interpolation, masking, scaling, cache round trip, legacy and stale caches")
    }
}
