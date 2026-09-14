import AVKit
import SwiftUI

struct ContentView: View {
    @State private var viewModel = PlayerViewModel()
    @State private var analysis = AnalysisViewModel()
    @State private var mesh: BrainMesh?
    @State private var inflatedBrain = false
    @State private var multimodalMode = false
    @State private var brainResetVersion = 0
    @State private var meshError: String?
    @State private var currentColors: [SIMD4<Float>]?
    @State private var smokeStatus: String?
    @State private var smokeRunning = false
    @State private var missingModels: [String] = []

    var body: some View {
        VStack(spacing: 12) {
            HSplitView {
                Group {
                    if let player = viewModel.player {
                        PlayerView(player: player)
                    } else {
                        ContentUnavailableView {
                            Label("No Video", systemImage: "film")
                        } description: {
                            Text("Open a video to analyze its predicted brain activity.")
                        } actions: {
                            Button("Open Video…") { viewModel.openVideo() }
                                .buttonStyle(.borderedProminent)
                        }
                    }
                }
                .frame(minWidth: 480, maxWidth: .infinity, minHeight: 360, maxHeight: .infinity)

                Group {
                    if mesh != nil {
                        BrainView(mesh: mesh, vertexColors: currentColors,
                                  inflated: inflatedBrain, resetVersion: brainResetVersion)
                            .overlay(alignment: .top) {
                                BrainControls(multimodal: $multimodalMode, inflated: $inflatedBrain) {
                                    brainResetVersion += 1
                                }
                            }
                            .overlay(alignment: .bottomLeading) {
                                VStack(alignment: .leading, spacing: 6) {
                                    Text(brainLegendTitle)
                                        .font(.caption.weight(.medium))
                                    if multimodalMode {
                                        Text(analysis.multimodal == nil
                                             ? (analysis.isRunning ? "Computing audio, video, and text responses…" : "Analyze the video to generate modality responses.")
                                             : "Audio · Video · Text — blended where responses overlap")
                                            .font(.caption2).foregroundStyle(.secondary)
                                        if analysis.activity != nil && analysis.multimodal == nil && !analysis.isRunning {
                                            Button("Analyze for Multimodality") {
                                                if let url = viewModel.videoURL { analysis.analyze(videoURL: url) }
                                            }
                                            .controlSize(.small)
                                            .disabled(viewModel.videoURL == nil || !missingModels.isEmpty)
                                        }
                                    } else if currentColors != nil {
                                        HStack(spacing: 7) {
                                            Text("Low")
                                            LinearGradient(colors: (0...24).map {
                                                ActivityPalette.color(ActivityPalette.activation(Float($0) / 24,
                                                    anatomy: SIMD4(0.66, 0.66, 0.66, 1)))
                                            }, startPoint: .leading, endPoint: .trailing)
                                            .frame(width: 100, height: 5).clipShape(Capsule())
                                            Text("High")
                                        }
                                        .font(.system(size: 10)).foregroundStyle(.secondary)
                                    }
                                }
                                .help(multimodalMode
                                      ? "Single-input predictions above the zero-input baseline, with a shared scale across the video. Text is red, audio green, and video blue. These comparisons are not additive contributions to the combined prediction."
                                      : "Positive predictions use a fixed scale across this video. Weak and negative responses remain gray; the medial wall is masked.")
                                .padding(16)
                            }
                    } else {
                        ContentUnavailableView(
                            "No Brain Mesh",
                            systemImage: "brain",
                            description: Text(meshError ?? "Loading…")
                        )
                    }
                }
                .frame(minWidth: 440, maxWidth: .infinity, minHeight: 360, maxHeight: .infinity)
            }
            .frame(minHeight: 400, maxHeight: .infinity)

            TimelineView(
                progress: viewModel.progress,
                onSeek: { viewModel.seek(toFraction: $0) },
                tracks: multimodalMode
                    ? (analysis.modalityTracks ?? [HeatTrack(name: "Audio", values: nil), HeatTrack(name: "Video", values: nil), HeatTrack(name: "Text", values: nil)])
                    : (analysis.tracks ?? TimelineView.defaultTracks)
            )
            .padding(.horizontal)

            if !missingModels.isEmpty {
                HStack(spacing: 8) {
                    Image(systemName: "exclamationmark.triangle.fill")
                        .foregroundStyle(.yellow)
                    Text("Missing CoreML models: \(missingModels.joined(separator: ", ")). Run the conversion scripts in Conversion/ (see Conversion/NOTES_audio.md / NOTES_video.md), then rebuild with `make build`.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                .padding(8)
                .frame(maxWidth: .infinity, alignment: .leading)
                .background(.yellow.opacity(0.1), in: RoundedRectangle(cornerRadius: 8))
                .padding(.horizontal)
            }

            HStack {
                Button("Open Video…") { viewModel.openVideo() }
                    .keyboardShortcut("o", modifiers: .command)
                Button(viewModel.player?.timeControlStatus == .paused ? "Play" : "Pause") {
                    viewModel.togglePlayPause()
                }
                .disabled(viewModel.player == nil)
                Button("Analyze Video") {
                    if let url = viewModel.videoURL { analysis.analyze(videoURL: url) }
                }
                .disabled(viewModel.videoURL == nil || analysis.isRunning || !missingModels.isEmpty)
                analysisStatus
                Spacer()
                Button("Smoke Test") { runSmokeTest() }
                    .controlSize(.small)
                    .disabled(smokeRunning)
                if let smokeStatus {
                    Text(smokeStatus)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .lineLimit(1)
                }
                Text("\(format(viewModel.currentTime)) / \(format(viewModel.duration))")
                    .font(.caption.monospacedDigit())
                    .foregroundStyle(.secondary)
            }
            .padding(.horizontal)
            .padding(.bottom, 8)
        }
        .frame(minWidth: 1100, minHeight: 720)
        .task {
            missingModels = AnalysisModelURLs.missingBundledModels()
            do {
                let loaded = try BrainMesh.loadFromBundle()
                mesh = loaded
                analysis.setMesh(loaded)
            } catch {
                meshError = error.localizedDescription
            }
        }
        .onChange(of: viewModel.progress) { _, _ in updateBrainColors() }
        .onChange(of: viewModel.videoURL) { _, url in
            if let url { analysis.videoOpened(url) }
        }
        .onChange(of: analysis.generation) { _, _ in updateBrainColors() }
        .onChange(of: multimodalMode) { _, _ in updateBrainColors() }
    }

    private var brainLegendTitle: String {
        if multimodalMode {
            return analysis.multimodal == nil ? "Multimodality · Awaiting analysis" : "Single-input responses"
        }
        return currentColors == nil ? "Anatomy · Awaiting analysis" : "Positive response"
    }

    @ViewBuilder
    private var analysisStatus: some View {
        switch analysis.state {
        case .idle:
            if viewModel.videoURL != nil {
                Text("Not analyzed — run Analyze Video.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        case .running(let stage, let fraction):
            ProgressView(value: fraction)
                .frame(width: 140)
            Text("\(stage) \(Int((fraction * 100).rounded()))%")
                .font(.caption)
                .foregroundStyle(.secondary)
                .lineLimit(1)
            Button("Cancel") { analysis.cancel() }
                .controlSize(.small)
        case .ready(let trCount):
            Text("Analysis ready — \(trCount) TRs" + (analysis.notes.first.map { " · \($0)" } ?? ""))
                .font(.caption)
                .foregroundStyle(.secondary)
                .lineLimit(1)
        case .failed(let message):
            Text("Analysis failed: \(message)")
                .font(.caption)
                .foregroundStyle(.red)
                .lineLimit(1)
        }
    }

    private func updateBrainColors() {
        if multimodalMode {
            currentColors = mesh.flatMap { mesh in
                analysis.multimodal?.colors(predictionTime: viewModel.currentTime + hemodynamicOffsetSeconds, mesh: mesh)
            }
            return
        }
        guard let activity = analysis.activity, let mesh else {
            currentColors = nil
            return
        }
        currentColors = activity.normalizedColors(
            predictionTime: viewModel.currentTime + hemodynamicOffsetSeconds, mesh: mesh
        )
    }

    /// Runs the converted FmriEncoder head on random features to prove the
    /// in-app CoreML path end-to-end; renders the resulting per-TR activity as
    /// real heat tracks and drives the 3D brain surface.
    private func runSmokeTest() {
        smokeRunning = true
        smokeStatus = "Loading model…"
        Task.detached {
            do {
                let model = try FmriEncoderModel()
                let text = try FmriEncoderModel.randomFeatures(shape: FmriEncoderModel.textShape)
                let audio = try FmriEncoderModel.randomFeatures(shape: FmriEncoderModel.audioShape)
                let video = try FmriEncoderModel.randomFeatures(shape: FmriEncoderModel.videoShape)
                await MainActor.run { smokeStatus = "Running inference…" }
                let predictions = try model.predict(text: text, audio: audio, video: video)
                guard let result = BrainActivity(predictions: predictions) else {
                    throw FmriEncoderError.missingOutput
                }
                await MainActor.run {
                    analysis.adoptSmokeTestResult(result)
                    smokeStatus = "OK — \(result.trCount) TRs × \(result.vertexCount) vertices"
                    smokeRunning = false
                }
            } catch {
                await MainActor.run {
                    smokeStatus = "Failed: \(error.localizedDescription)"
                    smokeRunning = false
                }
            }
        }
    }

    private func format(_ seconds: Double) -> String {
        guard seconds.isFinite, seconds >= 0 else { return "0:00" }
        let total = Int(seconds.rounded(.down))
        return String(format: "%d:%02d", total / 60, total % 60)
    }
}
