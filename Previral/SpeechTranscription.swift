import Accelerate
import AVFAudio
import CoreMedia
import Foundation
import Speech

/// One transcribed word with its audio timing and its character range within
/// the full transcript (the range drives reference-faithful context assembly).
struct WordTiming: Sendable, Codable, Equatable {
    var text: String
    var start: Double
    var duration: Double
    /// Character offsets into `SpeechTranscript.text` (leading inter-word
    /// whitespace belongs to the following word's fragment, matching the
    /// reference's AddContextToWords fragment tiling).
    var charStart: Int
    var charEnd: Int
}

/// Speech transcription of a video's audio track: full transcript text plus
/// word-level timings.
struct SpeechTranscript: Sendable, Codable, Equatable {
    var text: String
    var words: [WordTiming]
}

enum TranscriptionOutcome: Sendable {
    case success(SpeechTranscript)
    /// Speech unavailable / nothing transcribed — the pipeline falls back to
    /// zero text features (the head was trained with modality dropout 0.3).
    case unavailable(String)
}

/// Word-level speech transcription via the macOS 26 Speech framework
/// (SpeechAnalyzer + SpeechTranscriber) — the Apple-native replacement for
/// the reference pipeline's Whisper step.
enum SpeechTranscription {
    /// Transcribes 16 kHz mono samples. Throws CancellationError on cancel;
    /// all other failures come back as `.unavailable(reason)`.
    static func transcribe(
        samples: [Float], reporter: ProgressReporter
    ) async throws -> TranscriptionOutcome {
        guard !samples.isEmpty else { return .unavailable("no audio track") }
        let totalSeconds = Double(samples.count) / Double(AudioEncoderModel.sampleRate)

        let currentLocale = await SpeechTranscriber.supportedLocale(equivalentTo: .current)
        let locale: Locale?
        if let currentLocale {
            locale = currentLocale
        } else {
            locale = await SpeechTranscriber.supportedLocale(equivalentTo: Locale(identifier: "en_US"))
        }
        guard let locale else { return .unavailable("no supported speech locale") }

        let transcriber = SpeechTranscriber(
            locale: locale,
            preset: SpeechTranscriber.Preset(
                transcriptionOptions: [],
                reportingOptions: [],
                attributeOptions: [.audioTimeRange]
            )
        )
        // AssetInventory.status is misleading for this module configuration
        // (reports .supported while the asset is present and usable; the
        // install request may be nil). Attempt a download only when one is
        // actually offered, then transcribe regardless — real failures
        // surface as errors from the results sequence.
        if await AssetInventory.status(forModules: [transcriber]) == .supported,
           let request = try? await AssetInventory.assetInstallationRequest(supporting: [transcriber]) {
            reporter.report(
                stage: "Downloading speech model…",
                fraction: AnalysisPipeline.transcribeRange.lowerBound, force: true
            )
            try? await request.downloadAndInstall()
        }

        // SpeechAnalyzer reads files; reuse the stage-1 16 kHz mono samples.
        let tempURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("previral-speech-\(UUID().uuidString).wav")
        defer { try? FileManager.default.removeItem(at: tempURL) }
        do {
            try writeWAV(samples: samples, to: tempURL)
        } catch {
            return .unavailable("temporary wav: \(error.localizedDescription)")
        }

        do {
            let audioFile = try AVAudioFile(forReading: tempURL)
            // The file-based convenience init handles end-of-file
            // finalization (finishAfterFile); the bare modules: init +
            // analyzeSequence path never terminated in headless testing.
            let analyzer = try await SpeechAnalyzer(
                inputAudioFile: audioFile, modules: [transcriber], finishAfterFile: true
            )

            var text = ""
            var words: [WordTiming] = []
            var charBase = 0
            do {
                for try await result in transcriber.results {
                    try Task.checkCancellation()
                    append(result: result, text: &text, words: &words, charBase: &charBase)
                    let end = CMTimeGetSeconds(result.range.start) + CMTimeGetSeconds(result.range.duration)
                    reporter.report(
                        stage: "Transcribing speech…",
                        fraction: AnalysisPipeline.transcribeRange.mapped(end / max(totalSeconds, 1e-9))
                    )
                }
            } catch let error as CancellationError {
                await analyzer.cancelAndFinishNow()
                throw error
            }
            guard !words.isEmpty else { return .unavailable("no speech detected") }
            return .success(SpeechTranscript(text: text, words: words))
        } catch let error as CancellationError {
            throw error
        } catch {
            return .unavailable(error.localizedDescription)
        }
    }

    /// Extracts words from a result. Speech runs are phrase fragments
    /// ("Hello,", " world."), not words — a run spanning several words is
    /// split on whitespace and its time range is distributed proportionally
    /// to character counts (approximation; word-boundary deviations vs the
    /// reference's Whisper timings are an accepted deviation of the
    /// Apple-native substitution).
    private static func append(
        result: SpeechTranscriber.Result, text: inout String,
        words: inout [WordTiming], charBase: inout Int
    ) {
        let resultText = String(result.text.characters)
        let resultCharacters = Array(resultText)

        for run in result.text.runs {
            guard let timeRange: CMTimeRange = run.audioTimeRange else { continue }
            let runStart = CMTimeGetSeconds(timeRange.start)
            let runDuration = CMTimeGetSeconds(timeRange.duration)
            guard runStart.isFinite, runDuration.isFinite, runDuration >= 0 else { continue }

            let runCharacterOffset = result.text.characters.distance(
                from: result.text.characters.startIndex, to: run.range.lowerBound
            )
            let runCharacters = Array(String(result.text.characters[run.range]))

            // Word tokens: maximal non-whitespace spans; inter-word whitespace
            // belongs to the following word's fragment.
            var wordSpans: [(start: Int, end: Int)] = []
            var wordStart: Int?
            for (i, character) in runCharacters.enumerated() {
                if character.isWhitespace {
                    if let start = wordStart {
                        wordSpans.append((start, i))
                        wordStart = nil
                    }
                } else if wordStart == nil {
                    wordStart = i
                }
            }
            if let start = wordStart { wordSpans.append((start, runCharacters.count)) }
            guard !wordSpans.isEmpty else { continue }

            let secondsPerChar = runCharacters.isEmpty ? 0 : runDuration / Double(runCharacters.count)
            for (start, end) in wordSpans {
                words.append(WordTiming(
                    text: String(runCharacters[start..<end]),
                    start: runStart + Double(start) * secondsPerChar,
                    duration: Double(end - start) * secondsPerChar,
                    charStart: charBase + runCharacterOffset + start,
                    charEnd: charBase + runCharacterOffset + end
                ))
            }
        }
        text += resultText
        charBase += resultCharacters.count
    }

    /// PCM16 mono 16 kHz WAV, hand-rolled — AVAudioFile's writer path aborts
    /// inside ExtAudioFile on this platform version (caulk CAAssertRtn in
    /// WriteInputProc), so build the 44-byte header + sample bytes directly.
    private static func writeWAV(samples: [Float], to url: URL) throws {
        let count = vDSP_Length(samples.count)
        var pcm = [Int16](unsafeUninitializedCapacity: samples.count) { buffer, initializedCount in
            var scaled = [Float](unsafeUninitializedCapacity: samples.count) { raw, done in
                var scale: Float = 32_767
                vDSP_vsmul(samples, 1, &scale, raw.baseAddress!, 1, count)
                var lo: Float = -32_768, hi: Float = 32_767
                vDSP_vclip(raw.baseAddress!, 1, &lo, &hi, raw.baseAddress!, 1, count)
                done = samples.count
            }
            scaled.withUnsafeMutableBufferPointer { ptr in
                vDSP_vfix16(ptr.baseAddress!, 1, buffer.baseAddress!, 1, count)
            }
            initializedCount = samples.count
        }

        let byteCount = UInt32(samples.count * 2)
        var data = Data()
        data.append(contentsOf: [0x52, 0x49, 0x46, 0x46])             // "RIFF"
        data.appendLE(UInt32(36) + byteCount)
        data.append(contentsOf: [0x57, 0x41, 0x56, 0x45])             // "WAVE"
        data.append(contentsOf: [0x66, 0x6D, 0x74, 0x20])             // "fmt "
        data.appendLE(UInt32(16))                                     // fmt chunk size
        data.appendLE(UInt16(1))                                      // PCM
        data.appendLE(UInt16(1))                                      // mono
        data.appendLE(UInt32(16_000))                                 // sample rate
        data.appendLE(UInt32(32_000))                                 // byte rate
        data.appendLE(UInt16(2))                                      // block align
        data.appendLE(UInt16(16))                                     // bits per sample
        data.append(contentsOf: [0x64, 0x61, 0x74, 0x61])             // "data"
        data.appendLE(byteCount)
        pcm.withUnsafeMutableBytes { data.append(contentsOf: $0) }
        try data.write(to: url)
    }
}

private extension Data {
    mutating func appendLE(_ value: UInt32) {
        var value = value.littleEndian
        append(Data(bytes: &value, count: 4))
    }

    mutating func appendLE(_ value: UInt16) {
        var value = value.littleEndian
        append(Data(bytes: &value, count: 2))
    }
}
