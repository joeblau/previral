import CoreML
import Foundation

/// Thin CoreML wrapper around the converted Llama-3.2-3B text encoder.
///
/// Contract (produced by `Conversion/convert_text_encoder.py`, see
/// Conversion/NOTES_text.md), one word-context per call:
///   token_ids       (1, L) int32  — context token ids, left-padded with
///                     128004 (<|finetune_right_pad_id|>) to the batch-max length
///   attention_mask  (1, L) int32  — 0 on pads, 1 on real tokens
///   target_weights  (1, L) float32 — 1/n_target on the last n_target positions
///   text_features   (1, 2, 3072) float16 out
/// L is flexible (1...2048).
final class TextEncoderModel: @unchecked Sendable {
    static let padTokenID: Int32 = 128_004
    static let maxTokens = 2048
    static let featureDims = 3072
    static let layerCount = 2
    /// Reference batches words in consecutive groups of 4; the batch-max token
    /// count determines padding, so batching matters even for per-word calls.
    static let batchSize = 4

    private let model: MLModel

    init(contentsOfCompiledModel url: URL, configuration: MLModelConfiguration = .init()) throws {
        // NOT .all: the MPS (GPU) backend of this flexible-length model
        // crashes with "shape for TensorData is not static" when one model
        // instance serves varying sequence lengths (MPSGraph shape-specialization
        // bug, macOS 27 beta). CPU+ANE handles varying lengths correctly.
        configuration.computeUnits = .cpuAndNeuralEngine
        model = try MLModel(contentsOf: url, configuration: configuration)
    }

    func encode(tokenIDs: [Int32], attentionMask: [Int32], targetWeights: [Float]) throws -> MLMultiArray {
        let length = tokenIDs.count
        precondition(attentionMask.count == length && targetWeights.count == length)

        let ids = try MLMultiArray(shape: [1, NSNumber(value: length)], dataType: .int32)
        _ = tokenIDs.withUnsafeBufferPointer {
            memcpy(ids.dataPointer, $0.baseAddress, length * MemoryLayout<Int32>.size)
        }
        let mask = try MLMultiArray(shape: [1, NSNumber(value: length)], dataType: .int32)
        _ = attentionMask.withUnsafeBufferPointer {
            memcpy(mask.dataPointer, $0.baseAddress, length * MemoryLayout<Int32>.size)
        }
        let weights = try MLMultiArray(shape: [1, NSNumber(value: length)], dataType: .float32)
        _ = targetWeights.withUnsafeBufferPointer {
            memcpy(weights.dataPointer, $0.baseAddress, length * MemoryLayout<Float>.size)
        }

        let provider = try MLDictionaryFeatureProvider(dictionary: [
            "token_ids": MLFeatureValue(multiArray: ids),
            "attention_mask": MLFeatureValue(multiArray: mask),
            "target_weights": MLFeatureValue(multiArray: weights),
        ])
        let result = try model.prediction(from: provider)
        guard let features = result.featureValue(for: "text_features")?.multiArrayValue else {
            throw FeatureEncoderError.missingOutput("text_features")
        }
        return features
    }
}
