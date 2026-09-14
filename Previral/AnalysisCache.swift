import Foundation

/// On-disk cache for a finished analysis, written next to the video file:
/// `<video-name>.previral.json` (metadata) and `<video-name>.previral.bin`
/// (Float32, vertexCount × trCount row-major — the same layout as
/// `BrainActivity.values`).
enum AnalysisCache {
    struct Metadata: Codable {
        var formatVersion: Int
        var videoPath: String
        var duration: Double
        var trCount: Int
        var vertexCount: Int
        var created: Date
    }

    static let formatVersion = 1

    static func jsonURL(for videoURL: URL) -> URL {
        videoURL.deletingPathExtension().appendingPathExtension("previral.json")
    }

    static func binURL(for videoURL: URL) -> URL {
        videoURL.deletingPathExtension().appendingPathExtension("previral.bin")
    }

    static func save(activity: BrainActivity, videoURL: URL, duration: Double) throws {
        let binary = activity.values.withUnsafeBufferPointer { Data(buffer: $0) }
        try binary.write(to: binURL(for: videoURL), options: .atomic)

        let metadata = Metadata(
            formatVersion: formatVersion,
            videoPath: videoURL.path,
            duration: duration,
            trCount: activity.trCount,
            vertexCount: activity.vertexCount,
            created: Date()
        )
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(metadata).write(to: jsonURL(for: videoURL), options: .atomic)
    }

    /// Returns the cached analysis when both cache files exist, are mutually
    /// consistent, and are at least as new as the video file; nil otherwise.
    static func loadFresh(for videoURL: URL) -> BrainActivity? {
        let jsonURL = jsonURL(for: videoURL)
        let binURL = binURL(for: videoURL)
        guard let jsonData = try? Data(contentsOf: jsonURL) else { return nil }
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        guard let metadata = try? decoder.decode(Metadata.self, from: jsonData),
              metadata.formatVersion == formatVersion,
              metadata.trCount > 0,
              metadata.vertexCount > 0,
              let binData = try? Data(contentsOf: binURL),
              binData.count == metadata.vertexCount * metadata.trCount * MemoryLayout<Float>.size,
              let videoDate = modificationDate(of: videoURL),
              let jsonDate = modificationDate(of: jsonURL),
              let binDate = modificationDate(of: binURL),
              jsonDate >= videoDate, binDate >= videoDate
        else { return nil }

        let values = binData.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
        return BrainActivity(values: values, vertexCount: metadata.vertexCount, trCount: metadata.trCount)
    }

    /// Reads the modification date straight from the file system —
    /// `URL.resourceValues` caches per URL instance and would serve stale
    /// dates on repeat freshness checks.
    private static func modificationDate(of url: URL) -> Date? {
        (try? FileManager.default.attributesOfItem(atPath: url.path))?[.modificationDate] as? Date
    }
}
