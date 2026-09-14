import Foundation

/// Hemodynamic lag compensation: the prediction at TR t describes the
/// stimulus at t−5 s, so the prediction for stimulus time s sits at
/// prediction time s+5 (clamped at the ends).
nonisolated let hemodynamicOffsetSeconds = 5.0

/// Owns analysis state for the currently open video: runs the pipeline off
/// the main actor, loads the on-disk cache, and maps between playhead time
/// and prediction TRs. All mutable state is MainActor-confined; Sendable so
/// background tasks can hop back safely.
@MainActor
@Observable
final class AnalysisViewModel: @unchecked Sendable {
    enum State: Equatable {
        case idle
        case running(stage: String, fraction: Double)
        case ready(trCount: Int)
        case failed(String)
    }

    private(set) var state: State = .idle
    private(set) var activity: BrainActivity?
    private(set) var multimodal: MultimodalActivity?
    private(set) var modalityTracks: [HeatTrack]?
    /// Soft-failure notes from the last analysis (e.g. transcription fallback).
    private(set) var notes: [String] = []
    /// Timeline tracks derived from `activity`, expressed in stimulus time.
    private(set) var tracks: [HeatTrack]?
    private(set) var videoURL: URL?
    /// Bumped every time `activity` is replaced, so views can refresh derived
    /// rendering state even when the State case is unchanged.
    private(set) var generation = 0

    private var mesh: BrainMesh?
    private var task: Task<Void, Never>?
    /// Bumped whenever a new operation supersedes the previous one, so stale
    /// async completions (cache check, cancelled analysis) never clobber the
    /// current state.
    private var sessionToken = 0

    var isRunning: Bool {
        if case .running = state { true } else { false }
    }

    func setMesh(_ mesh: BrainMesh) {
        self.mesh = mesh
        rebuildTracks()
    }

    /// A new video was opened: drop any running analysis and surface a fresh
    /// on-disk cache instead of re-analyzing.
    func videoOpened(_ url: URL) {
        guard url != videoURL else { return }
        task?.cancel()
        sessionToken += 1
        let token = sessionToken
        videoURL = url
        activity = nil
        multimodal = nil
        modalityTracks = nil
        generation += 1
        tracks = nil
        notes = []
        state = .running(stage: "Checking cache…", fraction: 0)
        task = Task.detached {
            let cached = AnalysisCache.loadFresh(for: url)
            await MainActor.run {
                guard self.sessionToken == token else { return }
                if let cached {
                    self.adopt(cached)
                } else {
                    self.state = .idle
                }
            }
        }
    }

    func analyze(videoURL url: URL) {
        guard !isRunning else { return }
        if videoURL != url { videoOpened(url) }
        task?.cancel()
        sessionToken += 1
        let token = sessionToken
        videoURL = url
        state = .running(stage: "Preparing…", fraction: 0)
        task = Task.detached {
            do {
                let result = try await AnalysisPipeline.run(videoURL: url) { progress in
                    Task { @MainActor in
                        guard self.sessionToken == token else { return }
                        self.progressUpdated(progress)
                    }
                }
                await MainActor.run {
                    guard self.sessionToken == token else { return }
                    self.adopt(result)
                }
            } catch {
                await MainActor.run {
                    guard self.sessionToken == token else { return }
                    if error is CancellationError {
                        self.state = .idle
                    } else {
                        self.state = .failed(error.localizedDescription)
                    }
                }
            }
        }
    }

    /// Cancels a running analysis. The pipeline checks for cancellation
    /// between audio chunks, frame requests, and head windows, so this takes
    /// effect within one model call.
    func cancel() {
        task?.cancel()
        sessionToken += 1
        state = .idle
    }

    /// The smoke test (M1 verification path) renders its synthetic result
    /// through the same display path; it is tied to no video and no cache.
    func adoptSmokeTestResult(_ result: BrainActivity) {
        task?.cancel()
        sessionToken += 1
        videoURL = nil
        notes = []
        adopt(AnalysisResult(activity: result, notes: []))
    }

    /// Maps playhead (stimulus) time to the prediction TR that describes it —
    /// see `hemodynamicOffsetSeconds`.
    nonisolated static func trIndex(forPlayheadSeconds seconds: Double, trCount: Int) -> Int {
        guard trCount > 0 else { return 0 }
        return min(max(Int((seconds + hemodynamicOffsetSeconds).rounded(.down)), 0), trCount - 1)
    }

    func trIndex(forPlayheadSeconds seconds: Double) -> Int {
        Self.trIndex(forPlayheadSeconds: seconds, trCount: activity?.trCount ?? 0)
    }

    /// Re-expresses a per-TR track in stimulus time: raw TR k describes the
    /// stimulus at k−5 s, so stimulus-time column j reads raw TR j+5, clamped
    /// at the end by repeating the last value.
    static func stimulusTimeTrack(_ values: [Double]) -> [Double] {
        let offset = Int(hemodynamicOffsetSeconds)
        guard values.count > offset else { return values }
        return Array(values.dropFirst(offset)) + Array(repeating: values.last ?? 0, count: offset)
    }

    private func progressUpdated(_ progress: AnalysisProgress) {
        guard case .running = state else { return }
        state = .running(stage: progress.stage, fraction: progress.fraction)
    }

    private func adopt(_ result: AnalysisResult) {
        activity = result.activity
        multimodal = result.multimodal
        notes = result.notes
        modalityTracks = result.multimodal.map { channels in
            [("Audio", channels.audio), ("Video", channels.video), ("Text", channels.text)].map { name, activity in
                HeatTrack(name: name, values: Self.stimulusTimeTrack(activity.overallTrack()))
            }
        }
        rebuildTracks()
        generation += 1
        state = .ready(trCount: result.activity.trCount)
    }

    private func rebuildTracks() {
        guard let activity else {
            tracks = nil
            return
        }
        var built = [HeatTrack(name: "Overall", values: Self.stimulusTimeTrack(activity.overallTrack()))]
        if let mesh {
            built.append(contentsOf: activity.roiTracks(mesh: mesh).map {
                HeatTrack(name: $0.name, values: $0.values.map(Self.stimulusTimeTrack))
            })
        }
        tracks = built
    }
}
