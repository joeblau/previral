import Foundation

/// On-disk cache for a finished analysis, written next to the video file:
/// `<video-name>.previral.json` (metadata) and `<video-name>.previral.bin`
/// (Float32, vertexCount × trCount row-major — the same layout as
/// `BrainActivity.values`). Version 2 appends text, audio, and video matrices
/// when available, and keeps input-availability notes in the metadata.
enum AnalysisCache {
    struct Metadata: Codable {
        var formatVersion: Int
        var videoPath: String
        var duration: Double
        var trCount: Int
        var vertexCount: Int
        var created: Date
        var hasMultimodal: Bool? = nil
        var notes: [String]? = nil
    }

    static let formatVersion = 2

    static func jsonURL(for videoURL: URL) -> URL {
        videoURL.deletingPathExtension().appendingPathExtension("previral.json")
    }

    static func binURL(for videoURL: URL) -> URL {
        videoURL.deletingPathExtension().appendingPathExtension("previral.bin")
    }

    static func save(result: AnalysisResult, videoURL: URL, duration: Double) throws {
        let activity = result.activity
        var binary = activity.values.withUnsafeBufferPointer { Data(buffer: $0) }
        if let channels = result.multimodal {
            precondition(channels.text.vertexCount == activity.vertexCount && channels.text.trCount == activity.trCount)
            for channel in [channels.text, channels.audio, channels.video] {
                channel.values.withUnsafeBufferPointer { binary.append(Data(buffer: $0)) }
            }
        }
        try binary.write(to: binURL(for: videoURL), options: .atomic)

        let metadata = Metadata(
            formatVersion: formatVersion,
            videoPath: videoURL.path,
            duration: duration,
            trCount: activity.trCount,
            vertexCount: activity.vertexCount,
            created: Date(),
            hasMultimodal: result.multimodal != nil,
            notes: result.notes
        )
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(metadata).write(to: jsonURL(for: videoURL), options: .atomic)
    }

    /// Returns the cached analysis when both cache files exist, are mutually
    /// consistent, and are at least as new as the video file; nil otherwise.
    static func loadFresh(for videoURL: URL) -> AnalysisResult? {
        let jsonURL = jsonURL(for: videoURL)
        let binURL = binURL(for: videoURL)
        guard let jsonData = try? Data(contentsOf: jsonURL) else { return nil }
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        guard let metadata = try? decoder.decode(Metadata.self, from: jsonData),
              (metadata.formatVersion == 1 || metadata.formatVersion == formatVersion),
              metadata.videoPath == videoURL.path,
              metadata.trCount > 0,
              metadata.vertexCount > 0,
              let binData = try? Data(contentsOf: binURL),
              let videoDate = modificationDate(of: videoURL),
              let jsonDate = modificationDate(of: jsonURL),
              let binDate = modificationDate(of: binURL),
              jsonDate >= videoDate, binDate >= videoDate
        else { return nil }

        let hasMultimodal = metadata.formatVersion >= 2 && metadata.hasMultimodal == true
        let (count, countOverflow) = metadata.vertexCount.multipliedReportingOverflow(by: metadata.trCount)
        let (expectedBytes, byteOverflow) = count.multipliedReportingOverflow(by: (hasMultimodal ? 4 : 1) * MemoryLayout<Float>.size)
        guard !countOverflow, !byteOverflow, binData.count == expectedBytes else { return nil }
        let values = binData.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
        guard values.allSatisfy(\.isFinite) else { return nil }
        func activity(_ channel: Int) -> BrainActivity {
            BrainActivity(values: Array(values[(channel * count)..<((channel + 1) * count)]),
                          vertexCount: metadata.vertexCount, trCount: metadata.trCount)
        }
        return AnalysisResult(activity: activity(0), notes: metadata.notes ?? [],
                              multimodal: hasMultimodal ? MultimodalActivity(text: activity(1), audio: activity(2), video: activity(3)) : nil)
    }

    /// Reads the modification date straight from the file system —
    /// `URL.resourceValues` caches per URL instance and would serve stale
    /// dates on repeat freshness checks.
    private static func modificationDate(of url: URL) -> Date? {
        (try? FileManager.default.attributesOfItem(atPath: url.path))?[.modificationDate] as? Date
    }
}
