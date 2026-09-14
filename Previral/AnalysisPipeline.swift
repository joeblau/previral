import Accelerate
import AVFoundation
import CoreML
import Foundation

/// Progress of a running analysis, reported from a background thread.
struct AnalysisProgress: Sendable, Equatable {
    var stage: String
    /// Overall fraction complete, 0...1.
    var fraction: Double
}

enum AnalysisError: LocalizedError {
    case modelNotFoundInBundle(String)
    case invalidDuration
    case noVideoTrack
    case readerFailed(String)
    case audioConversionFailed(String)
    case imageConversionFailed(String)
    case unexpectedPixelFormat(OSType)

    var errorDescription: String? {
        switch self {
        case .modelNotFoundInBundle(let name):
            "\(name).mlmodelc not found in the app bundle. Run the conversion scripts in Conversion/ and rebuild so Models/\(name).mlpackage is included."
        case .invalidDuration:
            "The video's duration is not finite or is zero."
        case .noVideoTrack:
            "The file has no video track."
        case .readerFailed(let detail):
            "AVAssetReader failed: \(detail)"
        case .audioConversionFailed(let detail):
            "Audio extraction/conversion failed: \(detail)"
        case .imageConversionFailed(let detail):
            "Frame preprocessing failed: \(detail)"
        case .unexpectedPixelFormat(let format):
            "Decoded video frame has an unexpected pixel format (\(format))."
        }
    }
}

/// Locations of the compiled CoreML models + tokenizer files. The app
/// resolves them from the bundle; headless tests pass explicit paths.
struct AnalysisModelURLs: Sendable {
    var audio: URL
    var video: URL
    var text: URL
    var fmri: URL
    /// Directory containing tokenizer.json + tokenizer_config.json.
    var tokenizerDirectory: URL

    static func bundled() throws -> AnalysisModelURLs {
        func url(_ name: String) throws -> URL {
            guard let url = Bundle.main.url(forResource: name, withExtension: "mlmodelc") else {
                throw AnalysisError.modelNotFoundInBundle(name)
            }
            return url
        }
        guard let tokenizer = Bundle.main.url(forResource: "tokenizer", withExtension: "json") else {
            throw AnalysisError.modelNotFoundInBundle("tokenizer.json")
        }
        return try AnalysisModelURLs(
            audio: url("AudioEncoder"), video: url("VideoEncoder"),
            text: url("TextEncoder"), fmri: url("FmriEncoder"),
            tokenizerDirectory: tokenizer.deletingLastPathComponent()
        )
    }

    /// Names of the CoreML models / resources missing from the app bundle
    /// (empty = all present), for surfacing a clear setup message before
    /// analysis starts.
    static func missingBundledModels() -> [String] {
        var missing: [String] = ["AudioEncoder", "VideoEncoder", "TextEncoder", "FmriEncoder"].filter {
            Bundle.main.url(forResource: $0, withExtension: "mlmodelc") == nil
        }
        if Bundle.main.url(forResource: "tokenizer", withExtension: "json") == nil {
            missing.append("tokenizer.json")
        }
        return missing
    }
}

/// Throttles stage/fraction updates so the UI is not flooded with hops to the
/// main actor. Confined to the single task that runs the pipeline.
final class ProgressReporter {
    private let handler: @Sendable (AnalysisProgress) -> Void
    private var lastReport = ContinuousClock.Instant.now - .seconds(1)
    private var lastStage = ""

    init(handler: @escaping @Sendable (AnalysisProgress) -> Void) {
        self.handler = handler
    }

    func report(stage: String, fraction: Double, force: Bool = false) {
        let now = ContinuousClock.Instant.now
        guard force || stage != lastStage || now - lastReport > .milliseconds(100) else { return }
        lastReport = now
        lastStage = stage
        handler(AnalysisProgress(stage: stage, fraction: min(max(fraction, 0), 1)))
    }
}

/// The in-app TRIBE v2 analysis pipeline. Runs on a background task; each
/// CoreML model is loaded lazily for its stage and released before the next
/// stage loads its model (the four packages total ~9.4 GB).
///
/// Preprocessing follows Conversion/NOTES_audio.md, NOTES_video.md and
/// NOTES_text.md exactly.
enum AnalysisPipeline {
    /// Feature grid rate shared by all encoders.
    static let featureRate = 2.0
    /// Feature timesteps per head window (100 s at 2 Hz).
    static let headFeatureTimesteps = 200
    /// TRs emitted per head window.
    static let headOutputTRs = 100

    // Stage-to-overall-fraction map. Video + text encoding dominate runtime.
    static let audioExtractRange = 0.00...0.03
    static let audioEncodeRange = 0.03...0.08
    static let videoEncodeRange = 0.08...0.55
    static let transcribeRange = 0.55...0.62
    static let textEncodeRange = 0.62...0.85
    static let headRange = 0.85...0.98

    /// `transcript` / `transcriptDumpURL` are test seams (headless harness):
    /// pass a transcript to bypass Speech transcription, or a dump URL to
    /// record the transcript the pipeline used.
    static func run(
        videoURL: URL,
        models explicitModels: AnalysisModelURLs? = nil,
        transcript injectedTranscript: SpeechTranscript? = nil,
        transcriptDumpURL: URL? = nil,
        progress: @escaping @Sendable (AnalysisProgress) -> Void
    ) async throws -> AnalysisResult {
        let models = try explicitModels ?? AnalysisModelURLs.bundled()
        let reporter = ProgressReporter(handler: progress)
        var notes: [String] = []

        let asset = AVURLAsset(url: videoURL)
        let duration = try await asset.load(.duration).seconds
        guard duration.isFinite, duration > 0 else { throw AnalysisError.invalidDuration }
        let timesteps = max(1, Int((duration * featureRate).rounded()))

        // Stage 1 — audio: extract, encode, release the encoder.
        var audioGrid = [Float](
            repeating: 0,
            count: 2 * AudioEncoderModel.featureDims * timesteps
        )
        let samples = try await extractMono16k(asset: asset, duration: duration, reporter: reporter)
        if !samples.isEmpty {
            try Task.checkCancellation()
            do {
                let encoder = try AudioEncoderModel(contentsOfCompiledModel: models.audio)
                audioGrid = try encodeAudio(
                    samples: samples, timesteps: timesteps, encoder: encoder, reporter: reporter
                )
            }
        }

        // Stage 2 — video frames + encoder.
        try Task.checkCancellation()
        let videoGrid: [Float]
        do {
            let encoder = try VideoEncoderModel(contentsOfCompiledModel: models.video)
            videoGrid = try await encodeVideo(
                asset: asset, duration: duration, timesteps: timesteps, encoder: encoder, reporter: reporter
            )
        }

        // Stage 3 — text: transcribe (Speech framework), encode words with
        // the LLaMA text encoder, release it. Any soft failure leaves the
        // text grid zeroed (the head tolerates a missing modality).
        var textGrid = [Float](
            repeating: 0,
            count: 2 * TextEncoderModel.featureDims * timesteps
        )
        if !samples.isEmpty {
            try Task.checkCancellation()
            let outcome: TranscriptionOutcome
            if let injectedTranscript {
                outcome = .success(injectedTranscript)
            } else {
                outcome = try await SpeechTranscription.transcribe(samples: samples, reporter: reporter)
            }
            switch outcome {
            case .success(let transcript) where !transcript.words.isEmpty:
                if let transcriptDumpURL,
                   let data = try? JSONEncoder().encode(transcript) {
                    try? data.write(to: transcriptDumpURL)
                }
                do {
                    let featureEncoder = try await TextFeatureEncoder(tokenizerDirectory: models.tokenizerDirectory)
                    let encoder = try TextEncoderModel(contentsOfCompiledModel: models.text)
                    textGrid = try featureEncoder.encodeWords(
                        transcript: transcript, timesteps: timesteps, encoder: encoder, reporter: reporter
                    )
                }
            case .success:
                notes.append("No speech detected — text features are zeroed.")
            case .unavailable(let reason):
                notes.append("Speech transcription unavailable (\(reason)) — text features are zeroed.")
            }
        } else {
            notes.append("No audio track — audio and text features are zeroed.")
        }

        // Stage 4 — fMRI head over 100 s windows.
        try Task.checkCancellation()
        let activity: BrainActivity
        let multimodal: MultimodalActivity
        do {
            let head = try FmriEncoderModel(contentsOfCompiledModel: models.fmri)
            activity = try runHead(
                textGrid: textGrid, audioGrid: audioGrid, videoGrid: videoGrid,
                timesteps: timesteps, duration: duration, head: head, reporter: reporter,
                progressRange: 0.85...0.876
            )
            // Empty grids explicitly zero-fill an input. Subtracting the same
            // zero-input baseline removes the head's response to its biases.
            let baseline = try runHead(
                textGrid: [], audioGrid: [], videoGrid: [], timesteps: timesteps,
                duration: duration, head: head, reporter: reporter,
                stage: "Computing modality baseline", progressRange: 0.876...0.902)
            let text = try runHead(
                textGrid: textGrid, audioGrid: [], videoGrid: [], timesteps: timesteps,
                duration: duration, head: head, reporter: reporter,
                stage: "Predicting text response", progressRange: 0.902...0.928)
            let audio = try runHead(
                textGrid: [], audioGrid: audioGrid, videoGrid: [], timesteps: timesteps,
                duration: duration, head: head, reporter: reporter,
                stage: "Predicting audio response", progressRange: 0.928...0.954)
            let video = try runHead(
                textGrid: [], audioGrid: [], videoGrid: videoGrid, timesteps: timesteps,
                duration: duration, head: head, reporter: reporter,
                stage: "Predicting video response", progressRange: 0.954...0.98)
            multimodal = MultimodalActivity(
                text: MultimodalActivity.response(text, relativeTo: baseline),
                audio: MultimodalActivity.response(audio, relativeTo: baseline),
                video: MultimodalActivity.response(video, relativeTo: baseline))
        }

        // Stage 5 — cache next to the video. A cache failure must not sink a
        // finished analysis.
        reporter.report(stage: "Saving cache…", fraction: 0.99, force: true)
        try Task.checkCancellation()
        let result = AnalysisResult(activity: activity, notes: notes, multimodal: multimodal)
        try? AnalysisCache.save(result: result, videoURL: videoURL, duration: duration)
        reporter.report(stage: "Done", fraction: 1, force: true)
        return result
    }

    // MARK: - Stage 1: audio

    /// Decodes the first audio track to 16 kHz mono Float32 PCM. Returns an
    /// empty array when the asset has no audio track (the head was trained
    /// with zero-filled missing modalities; grid bins stay zero).
    static func extractMono16k(asset: AVAsset, duration: Double, reporter: ProgressReporter) async throws -> [Float] {
        guard let track = try await asset.loadTracks(withMediaType: .audio).first else { return [] }

        var sampleRate = 48_000.0
        var channels = 2
        if let description = try await track.load(.formatDescriptions).first,
           let asbd = CMAudioFormatDescriptionGetStreamBasicDescription(description)?.pointee {
            if asbd.mSampleRate > 0 { sampleRate = asbd.mSampleRate }
            if asbd.mChannelsPerFrame > 0 { channels = Int(asbd.mChannelsPerFrame) }
        }
        guard let inputFormat = AVAudioFormat(
            commonFormat: .pcmFormatFloat32, sampleRate: sampleRate,
            channels: AVAudioChannelCount(channels), interleaved: true
        ), let outputFormat = AVAudioFormat(
            commonFormat: .pcmFormatFloat32, sampleRate: Double(AudioEncoderModel.sampleRate),
            channels: 1, interleaved: false
        ), let converter = AVAudioConverter(from: inputFormat, to: outputFormat) else {
            throw AnalysisError.audioConversionFailed("unsupported track format")
        }

        let reader = try AVAssetReader(asset: asset)
        let output = AVAssetReaderTrackOutput(track: track, outputSettings: [
            AVFormatIDKey: kAudioFormatLinearPCM,
            AVLinearPCMBitDepthKey: 32,
            AVLinearPCMIsFloatKey: true,
            AVLinearPCMIsBigEndianKey: false,
        ])
        guard reader.canAdd(output) else {
            throw AnalysisError.readerFailed("cannot attach audio output")
        }
        reader.add(output)
        guard reader.startReading() else {
            throw AnalysisError.readerFailed(reader.error?.localizedDescription ?? "audio reader failed to start")
        }

        let expected = max(1, Int(duration * Double(AudioEncoderModel.sampleRate)))
        var samples: [Float] = []
        samples.reserveCapacity(expected + AudioEncoderModel.sampleRate)

        while true {
            try Task.checkCancellation()
            guard let buffer = AVAudioPCMBuffer(pcmFormat: outputFormat, frameCapacity: 16_384) else {
                throw AnalysisError.audioConversionFailed("buffer allocation failed")
            }
            var conversionError: NSError?
            let status = converter.convert(to: buffer, error: &conversionError) { _, outStatus in
                guard let sampleBuffer = output.copyNextSampleBuffer(),
                      let pcm = makePCMBuffer(from: sampleBuffer, format: inputFormat) else {
                    outStatus.pointee = .endOfStream
                    return nil
                }
                outStatus.pointee = .haveData
                return pcm
            }
            if let conversionError {
                throw AnalysisError.audioConversionFailed(conversionError.localizedDescription)
            }
            if buffer.frameLength > 0, let channelData = buffer.floatChannelData {
                samples.append(contentsOf: UnsafeBufferPointer(start: channelData[0], count: Int(buffer.frameLength)))
            }
            reporter.report(
                stage: "Extracting audio…",
                fraction: audioExtractRange.mapped(Double(samples.count) / Double(expected))
            )
            switch status {
            case .endOfStream: return samples
            case .error:
                throw AnalysisError.audioConversionFailed(reader.error?.localizedDescription ?? "converter error")
            default: continue
            }
        }
    }

    private static func makePCMBuffer(from sampleBuffer: CMSampleBuffer, format: AVAudioFormat) -> AVAudioPCMBuffer? {
        let frames = CMSampleBufferGetNumSamples(sampleBuffer)
        guard frames > 0,
              let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(frames))
        else { return nil }
        buffer.frameLength = AVAudioFrameCount(frames)
        guard CMSampleBufferCopyPCMDataIntoAudioBufferList(
            sampleBuffer, at: 0, frameCount: Int32(frames), into: buffer.mutableAudioBufferList
        ) == noErr else { return nil }
        return buffer
    }

    /// Consecutive 60 s chunks covering the audio. The final chunk may be
    /// partial (see `encodeAudio` for how the tail is fed to the fixed 60 s
    /// model input).
    static func audioChunkRanges(sampleCount: Int) -> [Range<Int>] {
        var ranges: [Range<Int>] = []
        var start = 0
        while start < sampleCount {
            let end = min(start + AudioEncoderModel.chunkSamples, sampleCount)
            ranges.append(start..<end)
            start = end
        }
        return ranges
    }

    /// Runs the audio encoder and places the 2 Hz output columns at
    /// `chunkStart × 2` on the feature grid.
    ///
    /// Full 60 s chunks match the reference exactly. A partial tail chunk
    /// (or a whole track shorter than 60 s) must still fill the model's
    /// fixed 60 s input: the real audio is TILED (repeated) rather than
    /// zero-padded. Padding with zeros lets the log-mel floor frames dominate
    /// the model's internal per-chunk CMVN and corrupts the features
    /// (measured per-dim r ~0.4–0.5 for a zero-padded 40 s tail); tiling
    /// preserves the real audio's log-mel mean/var exactly and measured best
    /// (r ~0.66) of the padding strategies tried. The reference processes the
    /// tail as a standalone sub-60 s event, which a fixed-input model cannot
    /// reproduce exactly — residual deviation remains (Conversion/NOTES_audio.md).
    /// Only the `round(seconds × 2)` columns covering real audio are written;
    /// grid bins with no audio stay zero (`allow_missing` in the reference
    /// config).
    static func encodeAudio(
        samples: [Float], timesteps n: Int, encoder: AudioEncoderModel, reporter: ProgressReporter
    ) throws -> [Float] {
        var grid = [Float](repeating: 0, count: 2 * AudioEncoderModel.featureDims * n)
        let rows = 2 * AudioEncoderModel.featureDims
        let chunkSamples = AudioEncoderModel.chunkSamples
        let chunkColumns = AudioEncoderModel.columnsPerChunk

        let fullChunks = samples.count / chunkSamples
        let tailSamples = samples.count % chunkSamples
        let totalCalls = fullChunks + (tailSamples > 0 ? 1 : 0)
        var call = 0

        for index in 0..<fullChunks {
            try Task.checkCancellation()
            let range = (index * chunkSamples)..<((index + 1) * chunkSamples)
            let output = try encoder.encode(chunk: Array(samples[range]))
            copyAudioColumns(output, into: &grid, rows: rows, columns: n,
                             gridColumnStart: index * chunkColumns,
                             outputColumnStart: 0,
                             count: min(chunkColumns, n - index * chunkColumns))
            call += 1
            reporter.report(
                stage: "Encoding audio features",
                fraction: audioEncodeRange.mapped(Double(call) / Double(totalCalls))
            )
        }

        if tailSamples > 0 {
            try Task.checkCancellation()
            let tailSeconds = Double(tailSamples) / Double(AudioEncoderModel.sampleRate)
            let validColumns = max(0, min(
                Int((tailSeconds * featureRate).rounded()),
                chunkColumns,
                n - fullChunks * chunkColumns
            ))
            if validColumns > 0 {
                let tail = samples[(samples.count - tailSamples)...]
                var chunk = [Float](repeating: 0, count: chunkSamples)
                chunk.withUnsafeMutableBufferPointer { dst in
                    guard let dstBase = dst.baseAddress else { return }
                    tail.withUnsafeBufferPointer { src in
                        guard let srcBase = src.baseAddress else { return }
                        var offset = 0
                        while offset < chunkSamples {
                            let n = min(tailSamples, chunkSamples - offset)
                            memcpy(dstBase + offset, srcBase, n * MemoryLayout<Float>.size)
                            offset += n
                        }
                    }
                }
                let output = try encoder.encode(chunk: chunk)
                copyAudioColumns(output, into: &grid, rows: rows, columns: n,
                                 gridColumnStart: fullChunks * chunkColumns,
                                 outputColumnStart: 0,
                                 count: validColumns)
            }
            call += 1
            reporter.report(
                stage: "Encoding audio features",
                fraction: audioEncodeRange.mapped(Double(call) / Double(totalCalls))
            )
        }
        return grid
    }

    /// Copies `count` columns of one chunk's output into the grid. Indexes
    /// the model output via strides — CoreML pads output rows (observed
    /// strides [., 131072, 128, 1] for logical [1, 2, 1024, 120]).
    private static func copyAudioColumns(
        _ output: MLMultiArray, into grid: inout [Float], rows: Int, columns n: Int,
        gridColumnStart: Int, outputColumnStart: Int, count: Int
    ) {
        guard count > 0 else { return }
        let src = output.dataPointer.assumingMemoryBound(to: UInt16.self) // fp16
        let layerStride = output.strides[1].intValue
        let dimStride = output.strides[2].intValue
        let timeStride = output.strides[3].intValue
        grid.withUnsafeMutableBufferPointer { gridPtr in
            guard let base = gridPtr.baseAddress else { return }
            for row in 0..<rows {
                let srcRow = src + (row / AudioEncoderModel.featureDims) * layerStride
                    + (row % AudioEncoderModel.featureDims) * dimStride
                let dstRow = base + row * n + gridColumnStart
                for column in 0..<count {
                    dstRow[column] = Float(Float16(bitPattern: srcRow[(outputColumnStart + column) * timeStride]))
                }
            }
        }
    }

    // MARK: - Stage 2: video

    struct FrameRequest {
        var time: Double
        var timestep: Int
        var slot: Int
    }

    /// 64 frame times per 2 Hz timestep: the window [t − 3.9375, t] sampled
    /// every 1/16 s, clamped at 0 (Conversion/NOTES_video.md). Timestep
    /// end-times are t_i = D·i/N for i = 1…N. Returned sorted by time for
    /// sequential decoding.
    static func videoFrameRequests(duration: Double, timesteps n: Int) -> [FrameRequest] {
        var requests: [FrameRequest] = []
        requests.reserveCapacity(n * VideoEncoderModel.framesPerTimestep)
        for i in 1...n {
            let t = duration * Double(i) / Double(n)
            for j in 0..<VideoEncoderModel.framesPerTimestep {
                requests.append(FrameRequest(
                    time: max(0, t - Double(VideoEncoderModel.framesPerTimestep - 1 - j) / 16),
                    timestep: i - 1,
                    slot: j
                ))
            }
        }
        requests.sort { ($0.time, $0.timestep, $0.slot) < ($1.time, $1.timestep, $1.slot) }
        return requests
    }

    /// Pull-decodes a video track in presentation order and serves frames for
    /// nondecreasing query times with previous-frame semantics: the frame
    /// whose display interval contains the requested time (the last frame for
    /// times at/past the end of the track).
    ///
    /// Known deviation: AVFoundation's YUV→RGB conversion of untagged
    /// content differs slightly from ffmpeg/swscale (the reference decode
    /// path) — measured ~1.66/255 mean pixel difference on the untagged SD
    /// test clip, which propagates to ~0.98 median per-dim feature
    /// correlation vs the PyTorch reference. Flagged for the M5 golden test.
    final class SequentialFrameDecoder {
        private let output: AVAssetReaderTrackOutput
        private let reader: AVAssetReader
        private var current: CMSampleBuffer?
        private var upcoming: CMSampleBuffer?

        init(asset: AVAsset, track: AVAssetTrack) throws {
            reader = try AVAssetReader(asset: asset)
            output = AVAssetReaderTrackOutput(track: track, outputSettings: [
                kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
            ])
            output.alwaysCopiesSampleData = false
            guard reader.canAdd(output) else {
                throw AnalysisError.readerFailed("cannot attach video output")
            }
            reader.add(output)
            guard reader.startReading() else {
                throw AnalysisError.readerFailed(reader.error?.localizedDescription ?? "video reader failed to start")
            }
            upcoming = output.copyNextSampleBuffer()
        }

        /// Query times must be nondecreasing. Returns the frame and its
        /// presentation time; nil when the track yields no frames at all.
        func frame(at time: Double) -> (buffer: CVPixelBuffer, pts: Double)? {
            while let next = upcoming, presentationSeconds(next) <= time {
                current = next
                upcoming = output.copyNextSampleBuffer()
            }
            guard let picked = current ?? upcoming else { return nil }
            guard let buffer = CMSampleBufferGetImageBuffer(picked) else { return nil }
            return (buffer, presentationSeconds(picked))
        }

        private func presentationSeconds(_ buffer: CMSampleBuffer) -> Double {
            let time = CMSampleBufferGetPresentationTimeStamp(buffer)
            return time.isNumeric ? time.seconds : 0
        }
    }

    /// V-JEPA2 frame preprocessing (Conversion/NOTES_video.md): quarter-turn
    /// to display orientation, shortest-edge → 292 bilinear resize, 256×256
    /// center crop, ×1/255, ImageNet mean/std, NaN → 0. Writes planar RGB
    /// float32 (3 × 256 × 256) into `dest`.
    final class FramePreprocessor {
        enum QuarterTurn {
            case none, clockwise, counterClockwise, half
        }

        private static let cropSize = 256
        private static let means: [Float] = [0.485, 0.456, 0.406]
        private static let stds: [Float] = [0.229, 0.224, 0.225]

        private let turn: QuarterTurn
        private var rotatedBuffer: UnsafeMutableRawPointer?
        private var rotatedBytes = 0
        private var scaledBuffer: UnsafeMutableRawPointer?
        private var scaledBytes = 0
        private let cropBuffer: UnsafeMutableRawPointer
        private let floatScratch: UnsafeMutablePointer<Float>

        init(preferredTransform: CGAffineTransform) {
            let angle = atan2(preferredTransform.b, preferredTransform.a)
            switch angle {
            case 0.4..<2.8: turn = .counterClockwise   // +90°
            case 2.8...3.6, -3.6 ... -2.8: turn = .half // 180°
            case -2.8 ... -0.4: turn = .clockwise        // −90°
            default: turn = .none
            }
            cropBuffer = UnsafeMutableRawPointer.allocate(
                byteCount: Self.cropSize * Self.cropSize * 4, alignment: 64
            )
            floatScratch = UnsafeMutablePointer<Float>.allocate(capacity: Self.cropSize * Self.cropSize)
        }

        deinit {
            rotatedBuffer?.deallocate()
            scaledBuffer?.deallocate()
            cropBuffer.deallocate()
            floatScratch.deallocate()
        }

        func preprocess(_ pixelBuffer: CVPixelBuffer, into dest: UnsafeMutablePointer<Float>) throws {
            let pixelFormat = CVPixelBufferGetPixelFormatType(pixelBuffer)
            guard pixelFormat == kCVPixelFormatType_32BGRA else {
                throw AnalysisError.unexpectedPixelFormat(pixelFormat)
            }
            CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
            defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }

            var width = CVPixelBufferGetWidth(pixelBuffer)
            var height = CVPixelBufferGetHeight(pixelBuffer)
            var source = vImage_Buffer(
                data: CVPixelBufferGetBaseAddress(pixelBuffer),
                height: vImagePixelCount(height),
                width: vImagePixelCount(width),
                rowBytes: CVPixelBufferGetBytesPerRow(pixelBuffer)
            )

            if turn != .none {
                let (outWidth, outHeight) = turn == .half ? (width, height) : (height, width)
                let rotated = cachedBuffer(&rotatedBuffer, size: &rotatedBytes, bytes: outHeight * outWidth * 4)
                var destination = vImage_Buffer(
                    data: rotated,
                    height: vImagePixelCount(outHeight),
                    width: vImagePixelCount(outWidth),
                    rowBytes: outWidth * 4
                )
                let rotationConstant = switch turn {
                case .clockwise: UInt8(kRotate90DegreesClockwise)
                case .counterClockwise: UInt8(kRotate90DegreesCounterClockwise)
                default: UInt8(kRotate180DegreesClockwise)
                }
                var background: UInt8 = 0
                let error = vImageRotate90_ARGB8888(&source, &destination, rotationConstant, &background, vImage_Flags(kvImageNoFlags))
                guard error == kvImageNoError else {
                    throw AnalysisError.imageConversionFailed("rotate failed (\(error))")
                }
                source = destination
                width = outWidth
                height = outHeight
            }

            // Shortest edge → 292, aspect preserved (bilinear).
            let scale = 292.0 / Double(min(width, height))
            let newWidth = max(1, Int((Double(width) * scale).rounded()))
            let newHeight = max(1, Int((Double(height) * scale).rounded()))
            let scaled = cachedBuffer(&scaledBuffer, size: &scaledBytes, bytes: newHeight * newWidth * 4)
            var scaledImage = vImage_Buffer(
                data: scaled,
                height: vImagePixelCount(newHeight),
                width: vImagePixelCount(newWidth),
                rowBytes: newWidth * 4
            )
            let scaleError = vImageScale_ARGB8888(&source, &scaledImage, nil, vImage_Flags(kvImageHighQualityResampling))
            guard scaleError == kvImageNoError else {
                throw AnalysisError.imageConversionFailed("scale failed (\(scaleError))")
            }

            // Center crop 256×256 into a contiguous buffer (shortest edge is
            // 292 ≥ 256, so the crop always fits).
            let cropX = (newWidth - Self.cropSize) / 2
            let cropY = (newHeight - Self.cropSize) / 2
            let sourceRowBytes = newWidth * 4
            for row in 0..<Self.cropSize {
                memcpy(
                    cropBuffer + row * Self.cropSize * 4,
                    scaled + (cropY + row) * sourceRowBytes + cropX * 4,
                    Self.cropSize * 4
                )
            }

            // BGRA bytes → planar RGB float32: (p/255 − mean)/std per channel.
            let pixelCount = Self.cropSize * Self.cropSize
            let bytes = cropBuffer.assumingMemoryBound(to: UInt8.self)
            for channel in 0..<3 {
                let byteOffset = 2 - channel // BGRA memory order: B=0, G=1, R=2
                vDSP_vfltu8(bytes + byteOffset, 4, floatScratch, 1, vDSP_Length(pixelCount))
                var multiplier = 1 / (255 * Self.stds[channel])
                var addend = -Self.means[channel] / Self.stds[channel]
                vDSP_vsmsa(
                    floatScratch, 1, &multiplier, &addend,
                    dest + channel * pixelCount, 1, vDSP_Length(pixelCount)
                )
            }

            // Reference safeguard (normally a no-op with constant mean/std).
            for i in 0..<(3 * pixelCount) where dest[i].isNaN {
                dest[i] = 0
            }
        }

        private func cachedBuffer(
            _ buffer: inout UnsafeMutableRawPointer?, size: inout Int, bytes: Int
        ) -> UnsafeMutableRawPointer {
            if bytes > size {
                buffer?.deallocate()
                buffer = UnsafeMutableRawPointer.allocate(byteCount: bytes, alignment: 64)
                size = bytes
            }
            return buffer!
        }
    }

    /// Decodes frames, preprocesses them per NOTES_video.md, and runs the
    /// video encoder once per 2 Hz timestep. Requests are consumed in
    /// ascending time order so the decoder never seeks; because consecutive
    /// timestep windows overlap by 56 frames, up to ~8 per-timestep input
    /// tensors (~50 MB each) are in flight at once.
    static func encodeVideo(
        asset: AVAsset, duration: Double, timesteps n: Int,
        encoder: VideoEncoderModel, reporter: ProgressReporter
    ) async throws -> [Float] {
        guard let track = try await asset.loadTracks(withMediaType: .video).first else {
            throw AnalysisError.noVideoTrack
        }
        let transform = try await track.load(.preferredTransform)
        let decoder = try SequentialFrameDecoder(asset: asset, track: track)
        let preprocessor = FramePreprocessor(preferredTransform: transform)
        let requests = videoFrameRequests(duration: duration, timesteps: n)

        var grid = [Float](repeating: 0, count: 2 * VideoEncoderModel.featureDims * n)
        var tensors: [Int: MLMultiArray] = [:]
        var filled: [Int: Int] = [:]
        let planes = UnsafeMutablePointer<Float>.allocate(
            capacity: 3 * VideoEncoderModel.frameSize * VideoEncoderModel.frameSize
        )
        defer { planes.deallocate() }
        var lastFramePTS = Double.nan
        var completed = 0
        let floatsPerFrame = 3 * VideoEncoderModel.frameSize * VideoEncoderModel.frameSize

        for request in requests {
            try Task.checkCancellation()
            guard let frame = decoder.frame(at: request.time) else {
                throw AnalysisError.readerFailed("video track produced no frames")
            }
            if frame.pts != lastFramePTS {
                try preprocessor.preprocess(frame.buffer, into: planes)
                lastFramePTS = frame.pts
            }

            let tensor: MLMultiArray
            if let existing = tensors[request.timestep] {
                tensor = existing
            } else {
                tensor = try encoder.makeInputTensor()
                tensors[request.timestep] = tensor
            }
            memcpy(
                tensor.dataPointer.assumingMemoryBound(to: Float.self) + request.slot * floatsPerFrame,
                planes,
                floatsPerFrame * MemoryLayout<Float>.size
            )
            filled[request.timestep, default: 0] += 1

            if filled[request.timestep] == VideoEncoderModel.framesPerTimestep {
                let features = try encoder.predict(pixelValues: tensor)
                let src = features.dataPointer.assumingMemoryBound(to: Float.self)
                // Index via strides — CoreML may pad output rows.
                let layerStride = features.strides[1].intValue
                let dimStride = features.strides[2].intValue
                grid.withUnsafeMutableBufferPointer { gridPtr in
                    guard let base = gridPtr.baseAddress else { return }
                    for row in 0..<(2 * VideoEncoderModel.featureDims) {
                        base[row * n + request.timestep] =
                            src[(row / VideoEncoderModel.featureDims) * layerStride
                                + (row % VideoEncoderModel.featureDims) * dimStride]
                    }
                }
                tensors[request.timestep] = nil
                filled[request.timestep] = nil
                completed += 1
                reporter.report(
                    stage: "Encoding video features",
                    fraction: videoEncodeRange.mapped(Double(completed) / Double(n))
                )
            }
        }
        return grid
    }

    // MARK: - Stage 4: fMRI head

    /// Assembles 100 s (T=200) feature windows and runs the FmriEncoder head.
    /// The tail window's features are zero-padded like the reference
    /// dataloader; TRs entirely past the video end are not emitted.
    static func runHead(
        textGrid: [Float], audioGrid: [Float], videoGrid: [Float], timesteps n: Int,
        duration: Double, head: FmriEncoderModel, reporter: ProgressReporter,
        stage: String = "Running brain model", progressRange: ClosedRange<Double> = headRange
    ) throws -> BrainActivity {
        let windowCount = Int(ceil(Double(n) / Double(headFeatureTimesteps)))
        let trTotal = min(windowCount * headOutputTRs, max(1, Int(ceil(duration))))
        var values = [Float](repeating: 0, count: FmriEncoderModel.vertexCount * trTotal)

        let textWindow = try MLMultiArray(shape: FmriEncoderModel.textShape, dataType: .float32)
        let audioWindow = try MLMultiArray(shape: FmriEncoderModel.audioShape, dataType: .float32)
        let videoWindow = try MLMultiArray(shape: FmriEncoderModel.videoShape, dataType: .float32)

        for window in 0..<windowCount {
            try Task.checkCancellation()
            let offset = window * headFeatureTimesteps
            let valid = max(0, min(headFeatureTimesteps, n - offset))
            fillWindow(textWindow, from: textGrid, rows: 2 * TextEncoderModel.featureDims, columns: n, offset: offset, valid: valid)
            fillWindow(audioWindow, from: audioGrid, rows: 2 * AudioEncoderModel.featureDims, columns: n, offset: offset, valid: valid)
            fillWindow(videoWindow, from: videoGrid, rows: 2 * VideoEncoderModel.featureDims, columns: n, offset: offset, valid: valid)

            let predictions = try head.predict(text: textWindow, audio: audioWindow, video: videoWindow)
            let keep = max(0, min(headOutputTRs, trTotal - window * headOutputTRs))
            copyPredictions(predictions, into: &values, trCount: trTotal, trOffset: window * headOutputTRs, keep: keep)
            reporter.report(
                stage: stage,
                fraction: progressRange.mapped(Double(window + 1) / Double(windowCount))
            )
        }
        return BrainActivity(values: values, vertexCount: FmriEncoderModel.vertexCount, trCount: trTotal)
    }

    /// Zero-fills the window, then copies the valid columns of each
    /// (layer, dim) row from the grid.
    private static func fillWindow(
        _ window: MLMultiArray, from grid: [Float], rows: Int, columns n: Int, offset: Int, valid: Int
    ) {
        let dst = window.dataPointer.assumingMemoryBound(to: Float.self)
        memset(dst, 0, rows * headFeatureTimesteps * MemoryLayout<Float>.size)
        guard valid > 0, !grid.isEmpty else { return }
        grid.withUnsafeBufferPointer { gridPtr in
            guard let src = gridPtr.baseAddress else { return }
            for row in 0..<rows {
                memcpy(
                    dst + row * headFeatureTimesteps,
                    src + row * n + offset,
                    valid * MemoryLayout<Float>.size
                )
            }
        }
    }

    /// Copies the first `keep` TR columns of a window's predictions
    /// ([1, vertex, TR]) into the global (vertexCount × trCount) matrix.
    /// Indexes via strides — CoreML pads output rows.
    private static func copyPredictions(
        _ predictions: MLMultiArray, into values: inout [Float], trCount: Int, trOffset: Int, keep: Int
    ) {
        guard keep > 0 else { return }
        let vertexCount = predictions.shape[1].intValue
        let vertexStride = predictions.strides[1].intValue
        let trStride = predictions.strides[2].intValue
        values.withUnsafeMutableBufferPointer { valuesPtr in
            guard let base = valuesPtr.baseAddress else { return }
            switch predictions.dataType {
            case .float32:
                let src = predictions.dataPointer.assumingMemoryBound(to: Float.self)
                for vertex in 0..<vertexCount {
                    let dstRow = base + vertex * trCount + trOffset
                    let srcRow = src + vertex * vertexStride
                    for t in 0..<keep {
                        dstRow[t] = srcRow[t * trStride]
                    }
                }
            case .float16:
                let src = predictions.dataPointer.assumingMemoryBound(to: UInt16.self)
                for vertex in 0..<vertexCount {
                    let dstRow = base + vertex * trCount + trOffset
                    let srcRow = src + vertex * vertexStride
                    for t in 0..<keep {
                        dstRow[t] = Float(Float16(bitPattern: srcRow[t * trStride]))
                    }
                }
            default:
                break
            }
        }
    }
}

extension ClosedRange<Double> {
    func mapped(_ fraction: Double) -> Double {
        lowerBound + (upperBound - lowerBound) * Swift.min(Swift.max(fraction, 0), 1)
    }
}
