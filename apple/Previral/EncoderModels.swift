import CoreML
import Foundation

enum FeatureEncoderError: LocalizedError {
    case missingOutput(String)

    var errorDescription: String? {
        switch self {
        case .missingOutput(let name):
            "The CoreML prediction did not contain the expected '\(name)' output."
        }
    }
}

/// Thin CoreML wrapper around the converted w2v-bert-2.0 audio encoder.
///
/// Contract (produced by `Conversion/convert_audio_encoder.py`, see
/// Conversion/NOTES_audio.md):
///   waveform       (1, 960000) float32 — raw 16 kHz mono 60 s, NO
///                  normalization (CMVN lives inside the model)
///   audio_features (1, 2, 1024, 120) float16 — the chunk's 2 Hz features
final class AudioEncoderModel: @unchecked Sendable {
    static let sampleRate = 16_000
    static let chunkSeconds = 60
    static let chunkSamples = sampleRate * chunkSeconds
    static let columnsPerChunk = chunkSeconds * 2
    static let featureDims = 1024

    private let model: MLModel

    init(contentsOfCompiledModel url: URL, configuration: MLModelConfiguration = .init()) throws {
        configuration.computeUnits = .all
        model = try MLModel(contentsOf: url, configuration: configuration)
    }

    /// `chunk` must be exactly `chunkSamples` long; callers zero-pad the tail.
    func encode(chunk: [Float]) throws -> MLMultiArray {
        precondition(chunk.count == Self.chunkSamples)
        let input = try MLMultiArray(shape: [1, NSNumber(value: Self.chunkSamples)], dataType: .float32)
        _ = chunk.withUnsafeBufferPointer {
            memcpy(input.dataPointer, $0.baseAddress, Self.chunkSamples * MemoryLayout<Float>.size)
        }
        let provider = try MLDictionaryFeatureProvider(dictionary: [
            "waveform": MLFeatureValue(multiArray: input),
        ])
        let result = try model.prediction(from: provider)
        guard let features = result.featureValue(for: "audio_features")?.multiArrayValue else {
            throw FeatureEncoderError.missingOutput("audio_features")
        }
        return features
    }
}

/// Thin CoreML wrapper around the converted V-JEPA2 video encoder.
///
/// Contract (produced by `Conversion/convert_video_encoder.py`, see
/// Conversion/NOTES_video.md):
///   pixel_values_videos (1, 64, 3, 256, 256) float32 — 64 preprocessed frames
///   features            (1, 2, 1408) float32
/// One call produces the features for ONE 2 Hz grid timestep.
final class VideoEncoderModel: @unchecked Sendable {
    static let framesPerTimestep = 64
    static let frameSize = 256
    static let featureDims = 1408

    private let model: MLModel

    init(contentsOfCompiledModel url: URL, configuration: MLModelConfiguration = .init()) throws {
        configuration.computeUnits = .all
        model = try MLModel(contentsOf: url, configuration: configuration)
    }

    func makeInputTensor() throws -> MLMultiArray {
        try MLMultiArray(
            shape: [1, NSNumber(value: Self.framesPerTimestep), 3,
                    NSNumber(value: Self.frameSize), NSNumber(value: Self.frameSize)],
            dataType: .float32
        )
    }

    func predict(pixelValues: MLMultiArray) throws -> MLMultiArray {
        let provider = try MLDictionaryFeatureProvider(dictionary: [
            "pixel_values_videos": MLFeatureValue(multiArray: pixelValues),
        ])
        let result = try model.prediction(from: provider)
        guard let features = result.featureValue(for: "features")?.multiArrayValue else {
            throw FeatureEncoderError.missingOutput("features")
        }
        return features
    }
}
