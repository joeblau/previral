import Accelerate
import CoreML
import Foundation

enum FmriEncoderError: LocalizedError {
    case modelNotFoundInBundle
    case missingOutput

    var errorDescription: String? {
        switch self {
        case .modelNotFoundInBundle:
            "FmriEncoder.mlmodelc not found in the app bundle. Run the conversion scripts in apple/Conversion/ and rebuild so apple/Models/FmriEncoder.mlpackage is included."
        case .missingOutput:
            "The CoreML prediction did not contain the expected 'predictions' output."
        }
    }
}

/// Thin CoreML wrapper around the converted TRIBE v2 FmriEncoder head.
///
/// Contract (produced by `Conversion/convert_fmri_encoder.py`, validated by
/// `Conversion/validate.py --stage head`):
///   text_features  (1, 2, 3072, 200) float32
///   audio_features (1, 2, 1024, 200) float32
///   video_features (1, 2, 1408, 200) float32
///   -> predictions (1, 20484, 100)
///
/// Features live on a 2 Hz grid, so T=200 covers a 100 s window that pools to
/// 100 per-TR prediction rows.
final class FmriEncoderModel: @unchecked Sendable {
    static let featureTimesteps = 200
    static let outputTimesteps = 100
    static let vertexCount = 20484

    static let textShape: [NSNumber] = [1, 2, 3072, NSNumber(value: featureTimesteps)]
    static let audioShape: [NSNumber] = [1, 2, 1024, NSNumber(value: featureTimesteps)]
    static let videoShape: [NSNumber] = [1, 2, 1408, NSNumber(value: featureTimesteps)]

    private let model: MLModel

    init(contentsOfCompiledModel url: URL, configuration: MLModelConfiguration = .init()) throws {
        configuration.computeUnits = .all
        model = try MLModel(contentsOf: url, configuration: configuration)
    }

    convenience init(configuration: MLModelConfiguration = .init()) throws {
        guard let url = Bundle.main.url(forResource: "FmriEncoder", withExtension: "mlmodelc") else {
            throw FmriEncoderError.modelNotFoundInBundle
        }
        try self.init(contentsOfCompiledModel: url, configuration: configuration)
    }

    func predict(text: MLMultiArray, audio: MLMultiArray, video: MLMultiArray) throws -> MLMultiArray {
        let provider = try MLDictionaryFeatureProvider(dictionary: [
            "text_features": MLFeatureValue(multiArray: text),
            "audio_features": MLFeatureValue(multiArray: audio),
            "video_features": MLFeatureValue(multiArray: video),
        ])
        let result = try model.prediction(from: provider)
        guard let predictions = result.featureValue(for: "predictions")?.multiArrayValue else {
            throw FmriEncoderError.missingOutput
        }
        return predictions
    }

    /// Standard-normal random features with a fixed seed path, for smoke-testing
    /// the in-app inference path before the feature encoders (M3) land.
    static func randomFeatures(shape: [NSNumber]) throws -> MLMultiArray {
        let array = try MLMultiArray(shape: shape, dataType: .float32)
        let count = shape.reduce(1) { $0 * $1.intValue }
        let ptr = array.dataPointer.assumingMemoryBound(to: Float.self)
        var state: UInt64 = 0x9E3779B97F4A7C15
        for i in 0..<count {
            // xorshift64 -> two uniforms -> Box-Muller
            state ^= state << 13; state ^= state >> 7; state ^= state << 17
            let u1 = max(Float(state >> 11) / Float(1 << 53), 1e-7)
            state ^= state << 13; state ^= state >> 7; state ^= state << 17
            let u2 = Float(state >> 11) / Float(1 << 53)
            ptr[i] = sqrt(-2 * log(u1)) * cos(2 * .pi * u2)
        }
        return array
    }

    /// Per-TR RMS across all 20,484 vertices — the "Overall stimulation" track.
    /// Predictions are laid out [1, vertex, TR]; indexed via strides because
    /// CoreML may pad output rows.
    static func overallTrack(from predictions: MLMultiArray) -> [Double] {
        let trCount = predictions.shape[2].intValue
        let vertices = predictions.shape[1].intValue
        let vertexStride = predictions.strides[1].intValue
        let trStride = predictions.strides[2].intValue
        var track = [Double](repeating: 0, count: trCount)

        if predictions.dataType == .float32 {
            let base = predictions.dataPointer.assumingMemoryBound(to: Float.self)
            for t in 0..<trCount {
                var rms: Float = 0
                vDSP_rmsqv(base + t * trStride, vertexStride, &rms, vDSP_Length(vertices))
                track[t] = Double(rms)
            }
        } else {
            // Fallback for fp16 outputs: scalar gather via subscripting.
            for t in 0..<trCount {
                var sumSquares = 0.0
                for v in 0..<vertices {
                    let value = predictions[[0, NSNumber(value: v), NSNumber(value: t)]].doubleValue
                    sumSquares += value * value
                }
                track[t] = (sumSquares / Double(vertices)).squareRoot()
            }
        }
        return track
    }
}
