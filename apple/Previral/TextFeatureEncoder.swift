import CoreML
import Foundation
import Tokenizers

/// Builds per-word LLaMA context windows from a speech transcript, runs the
/// TextEncoder, and places word features onto the 2 Hz grid. Replicates
/// Conversion/NOTES_text.md exactly — including the reference's pad/overshoot
/// span semantics (n_pads is always 0 because the reference miscounts
/// 128004 pads as 128001; n_target inflates into pads and previous words).
final class TextFeatureEncoder: @unchecked Sendable {
    let tokenizer: any Tokenizer

    init(tokenizerDirectory: URL) async throws {
        tokenizer = try await AutoTokenizer.from(modelFolder: tokenizerDirectory)
    }

    /// Context per word (AddContextToWords with sentence_only=false,
    /// max_context_len=1024, split_field=""): word fragments tile the
    /// transcript contiguously, so the context is the transcript substring
    /// from the end of the word 1024 words back through the end of this word.
    static func contexts(for transcript: SpeechTranscript) -> [String] {
        let characters = Array(transcript.text)
        return transcript.words.enumerated().map { index, word in
            let start = index >= 1024 ? transcript.words[index - 1024].charEnd : 0
            let end = min(word.charEnd, characters.count)
            guard start < end else { return word.text }
            return String(characters[start..<end])
        }
    }

    /// Tokenize with addSpecialTokens=false (no BOS); left-truncate beyond the
    /// model's 2048-token cap (the reference effectively never truncates —
    /// deviation only for >1024-word continuous speech, see NOTES_text.md).
    func tokenize(_ context: String) -> [Int] {
        let ids = tokenizer.encode(text: context, addSpecialTokens: false)
        return ids.count > TextEncoderModel.maxTokens ? Array(ids.suffix(TextEncoderModel.maxTokens)) : ids
    }

    /// Reference span math: prefix = context minus the word text, rstripped
    /// (Python code-point semantics), re-encoded separately (BPE boundary
    /// merges with the prefix are reference behavior).
    func prefixTokenCount(context: String, word: String) -> Int {
        var scalars = context.unicodeScalars.dropLast(word.unicodeScalars.count)
        while let last = scalars.last, CharacterSet.whitespaces.contains(last) {
            scalars = scalars.dropLast()
        }
        guard !scalars.isEmpty else { return 0 }
        return tokenizer.encode(text: String(String.UnicodeScalarView(scalars)), addSpecialTokens: false).count
    }

    /// Builds the (left-padded ids, mask, target weights) model inputs for one
    /// word within its batch of 4 consecutive words — the batch-max token
    /// count sets the padding length (reference DataLoader semantics, and
    /// n_pads is always 0 because the reference miscounts 128004 pads as
    /// 128001, so n_target inflates into pads/previous words).
    func rowInputs(
        index: Int, batchStart: Int, batchEnd: Int,
        tokenRows: [[Int]], contexts: [String], words: [WordTiming]
    ) -> (ids: [Int32], mask: [Int32], weights: [Float]) {
        let batchMaxLength = tokenRows[batchStart..<batchEnd].map(\.count).max() ?? 1
        let tokens = tokenRows[index]
        let padCount = batchMaxLength - tokens.count
        let ids = [Int32](repeating: TextEncoderModel.padTokenID, count: padCount)
            + tokens.map { Int32($0) }
        let mask = [Int32](repeating: 0, count: padCount)
            + [Int32](repeating: 1, count: tokens.count)
        let nPrefix = prefixTokenCount(context: contexts[index], word: words[index].text)
        let nTarget = max(1, batchMaxLength - nPrefix)
        var weights = [Float](repeating: 0, count: batchMaxLength)
        for position in (batchMaxLength - nTarget)..<batchMaxLength {
            weights[position] = 1 / Float(nTarget)
        }
        return (ids, mask, weights)
    }

    /// Encodes all words and returns the (2 × 3072, n) feature grid
    /// (row-major planar: row = layer*3072 + dim, column = 2 Hz bin).
    func encodeWords(
        transcript: SpeechTranscript, timesteps n: Int,
        encoder: TextEncoderModel, reporter: ProgressReporter
    ) throws -> [Float] {
        let words = transcript.words
        let contexts = Self.contexts(for: transcript)
        let tokenRows = contexts.map { tokenize($0) }
        var grid = [Float](repeating: 0, count: 2 * TextEncoderModel.featureDims * n)

        var batchStart = 0
        while batchStart < words.count {
            let batchEnd = min(batchStart + TextEncoderModel.batchSize, words.count)
            for index in batchStart..<batchEnd {
                try Task.checkCancellation()
                let row = rowInputs(
                    index: index, batchStart: batchStart, batchEnd: batchEnd,
                    tokenRows: tokenRows, contexts: contexts, words: words
                )
                let features = try encoder.encode(
                    tokenIDs: row.ids, attentionMask: row.mask, targetWeights: row.weights
                )
                Self.place(features, word: words[index], into: &grid, columns: n)
            }
            batchStart = batchEnd
            reporter.report(
                stage: "Encoding text features",
                fraction: AnalysisPipeline.textEncodeRange.mapped(Double(batchEnd) / Double(words.count))
            )
        }
        return grid
    }

    /// Adds a word's (2, 3072) feature to every 2 Hz bin overlapping
    /// [start, start+duration): start_bin = round(2·start),
    /// n_bins = max(1, round(2·duration)), with the reference's right-edge
    /// clamp (TimedArray._overlap_slice). round() is Python half-to-even.
    /// Overlapping words sum; bins with no words stay zero.
    static func place(_ features: MLMultiArray, word: WordTiming, into grid: inout [Float], columns n: Int) {
        var startBin = Int((word.start * 2).rounded(.toNearestOrEven))
        let binCount = max(1, Int((word.duration * 2).rounded(.toNearestOrEven)))
        if startBin > n - binCount { startBin = n - binCount }
        guard startBin >= 0 else { return }

        // Index via strides — CoreML pads output rows (M5 finding).
        let src = features.dataPointer.assumingMemoryBound(to: UInt16.self) // fp16
        let layerStride = features.strides[1].intValue
        let dimStride = features.strides[2].intValue
        grid.withUnsafeMutableBufferPointer { gridPtr in
            guard let base = gridPtr.baseAddress else { return }
            for row in 0..<(2 * TextEncoderModel.featureDims) {
                let value = Float(Float16(bitPattern: src[
                    (row / TextEncoderModel.featureDims) * layerStride
                        + (row % TextEncoderModel.featureDims) * dimStride
                ]))
                let rowBase = base + row * n
                for bin in startBin..<(startBin + binCount) {
                    rowBase[bin] += value
                }
            }
        }
    }
}
