import Foundation
import simd

/// A finished analysis, including any unavailable-input notes.
struct AnalysisResult: Sendable {
    var activity: BrainActivity
    var notes: [String]
    var multimodal: MultimodalActivity? = nil
}

/// Separate single-input responses, relative to the all-zero feature baseline.
/// These are input-isolation comparisons, not an additive attribution of the
/// full model prediction: the head is nonlinear and retains projector biases.
struct MultimodalActivity: Sendable {
    let text: BrainActivity
    let audio: BrainActivity
    let video: BrainActivity
    let displayScale: Float

    init(text: BrainActivity, audio: BrainActivity, video: BrainActivity) {
        precondition(text.vertexCount == audio.vertexCount && text.vertexCount == video.vertexCount)
        precondition(text.trCount == audio.trCount && text.trCount == video.trCount)
        self.text = text
        self.audio = audio
        self.video = video
        // One fixed scale preserves relative channel strength across the clip.
        displayScale = [text, audio, video].filter { $0.hi > 0 }.map(\.positiveDisplayScale).max() ?? 1
    }

    static func response(_ activity: BrainActivity, relativeTo baseline: BrainActivity) -> BrainActivity {
        precondition(activity.vertexCount == baseline.vertexCount && activity.trCount == baseline.trCount)
        return BrainActivity(values: zip(activity.values, baseline.values).map { value, base in
            let difference = value - base
            return difference.isFinite ? max(0, difference) : 0
        }, vertexCount: activity.vertexCount, trCount: activity.trCount)
    }

    func colors(predictionTime: Double, mesh: BrainMesh) -> [SIMD4<Float>] {
        guard mesh.vertexCount == text.vertexCount, text.trCount > 0 else { return mesh.anatomyColors }
        let time = predictionTime.isFinite ? min(max(predictionTime, 0), Double(text.trCount - 1)) : 0
        let first = Int(time), next = min(first + 1, text.trCount - 1)
        let fraction = Float(time - Double(first))
        return (0..<mesh.vertexCount).map { vertex in
            let anatomy = mesh.anatomyColors[vertex]
            guard mesh.roiLabels[vertex] != 0 else { return anatomy }
            func sample(_ activity: BrainActivity) -> Float {
                let a = activity.values[vertex * activity.trCount + first]
                let b = activity.values[vertex * activity.trCount + next]
                return (a + (b - a) * fraction) / displayScale
            }
            return ActivityPalette.multimodal(text: sample(text), audio: sample(audio),
                                              video: sample(video), anatomy: anatomy)
        }
    }
}
