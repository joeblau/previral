import SwiftUI

struct HeatTrack: Identifiable {
    var id: String { name }
    let name: String
    let values: [Double]?
}

struct TimelineView: View {
    static let defaultTracks: [HeatTrack] = [
        HeatTrack(name: "Overall", values: nil),
        HeatTrack(name: "Visual", values: nil),
        HeatTrack(name: "Auditory", values: nil),
        HeatTrack(name: "Language", values: nil),
    ]

    var progress: Double
    var onSeek: (Double) -> Void
    var tracks: [HeatTrack] = defaultTracks

    var body: some View {
        VStack(spacing: 10) {
            HStack {
                Text("ACTIVITY OVER TIME")
                    .font(.system(size: 10, weight: .semibold)).tracking(1.1)
                Text("Relative within each row")
                    .font(.caption2).foregroundStyle(.tertiary)
                Spacer()
                ActivityLegend()
            }
            .foregroundStyle(.secondary)
            ScrollView {
                VStack(alignment: .leading, spacing: 7) {
                    ForEach(tracks) { track in
                        HStack(spacing: 12) {
                            TimelineTrackLabel(name: track.name)
                            TrackBand(track: track)
                                .clipShape(RoundedRectangle(cornerRadius: 4))
                                .overlay(alignment: .leading) {
                                    GeometryReader { geo in
                                        Rectangle().fill(.white.opacity(0.95))
                                            .frame(width: 1.5)
                                            .shadow(color: .black.opacity(0.6), radius: 2)
                                            .offset(x: geo.size.width * min(max(progress, 0), 1) - 0.75)
                                    }
                                    .allowsHitTesting(false)
                                }
                                .overlay {
                                    GeometryReader { geo in
                                        Color.clear.contentShape(Rectangle())
                                            .gesture(DragGesture(minimumDistance: 0).onChanged { value in
                                                onSeek(min(max(value.location.x / max(geo.size.width, 1), 0), 1))
                                            })
                                    }
                                }
                        }
                        .frame(height: 20)
                    }
                }
            }
        }
        .padding(14)
        .background(Color.white.opacity(0.035), in: RoundedRectangle(cornerRadius: 12))
        .frame(height: min(max(CGFloat(tracks.count) * 27 + 53, 155), 285))
    }

}

struct TimelineTrackInfo {
    let name: String

    var title: String {
        switch name {
        case "DorsalAttention": "Dorsal attention"
        case "VentralAttention": "Ventral attention"
        case "Default": "Default mode"
        default: name
        }
    }

    // Network labels follow the Yeo 2011 seven-network cortical atlas:
    // https://surfer.nmr.mgh.harvard.edu/fswiki/CorticalParcellation_Yeo2011
    var summary: String {
        switch name {
        case "Overall":
            "A summary of predicted response strength across the entire cortical surface, combining all brain networks."
        case "Visual":
            "Processes visual information, including shapes, colors, and motion."
        case "Somatomotor":
            "Supports movement and body sensations, such as touch and awareness of body position."
        case "DorsalAttention":
            "Helps direct attention toward a chosen location or goal, such as following an object on screen."
        case "VentralAttention":
            "Helps shift attention toward unexpected or relevant events, such as a sudden sound or change in a scene."
        case "Limbic":
            "Includes cortical regions associated with emotion, motivation, and memory. This atlas label covers part of the broader limbic system."
        case "Frontoparietal":
            "Supports flexible control of thinking and behavior, including working memory, planning, and adjusting to a task."
        case "Default":
            "Often involved in internally directed thought, such as remembering, imagining, and thinking about yourself or other people."
        case "Auditory":
            "Processes sounds, including speech, music, and environmental sounds."
        case "Language":
            "Supports understanding words and combining them into meaningful language."
        case "Audio":
            "Predicted cortical responses when the model receives only the video's audio features."
        case "Video":
            "Predicted cortical responses when the model receives only visual features from the video frames."
        case "Text":
            "Predicted cortical responses when the model receives only text features from the video's speech transcript."
        default:
            "Predicted activity averaged across the cortical regions assigned to this network."
        }
    }

    var isAtlasNetwork: Bool {
        ["Visual", "Somatomotor", "DorsalAttention", "VentralAttention", "Limbic", "Frontoparietal", "Default"].contains(name)
    }

    var measurement: String {
        switch name {
        case "Overall":
            "This row summarizes response magnitude across the cortex, including both positive and negative predictions."
        case "Audio", "Video", "Text":
            "This row summarizes positive responses above the zero-input baseline across the cortex. Unavailable inputs contribute no response."
        default:
            "This row shows the average predicted response within the network."
        }
    }
}

struct TimelineTrackLabel: View {
    let name: String
    @State private var isHovered = false
    @State private var showsInfo = false
    @FocusState private var infoIsFocused: Bool

    private var info: TimelineTrackInfo { TimelineTrackInfo(name: name) }

    var body: some View {
        HStack(spacing: 6) {
            Text(info.title)
                .font(.system(size: 11, weight: .medium))
                .foregroundStyle(.secondary)
                .frame(width: 112, alignment: .trailing)
                .lineLimit(1)
            Button { showsInfo.toggle() } label: {
                Image(systemName: "info.circle")
                    .font(.system(size: 12))
                    .frame(width: 18, height: 20)
                    .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            .foregroundStyle(isHovered || showsInfo || infoIsFocused ? Color.secondary : Color.clear)
            .focused($infoIsFocused)
            .accessibilityLabel("About \(info.title)")
            .help("About \(info.title)")
            .popover(isPresented: $showsInfo, arrowEdge: .trailing) {
                TimelineTrackExplanation(info: info)
            }
        }
        .contentShape(Rectangle())
        .onHover { isHovered = $0 }
    }
}

struct TimelineTrackExplanation: View {
    let info: TimelineTrackInfo

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text(info.title)
                .font(.headline)
            Text(info.summary)
                .font(.callout)
            Text(info.measurement)
                .font(.caption).foregroundStyle(.secondary)
            Text("Brighter colors indicate higher values within this row over the video. Colors are scaled separately for each row.")
                .font(.caption).foregroundStyle(.secondary)
            if info.isAtlasNetwork {
                Link("About the Yeo 7-network atlas", destination: URL(string: "https://surfer.nmr.mgh.harvard.edu/fswiki/CorticalParcellation_Yeo2011")!)
                    .font(.caption)
            }
        }
        .fixedSize(horizontal: false, vertical: true)
        .padding(16)
        .frame(width: 300, alignment: .leading)
    }
}

struct TrackBand: View {
    let track: HeatTrack

    var body: some View {
        Canvas { context, size in
            let rect = CGRect(origin: .zero, size: size)
            guard let values = track.values, !values.isEmpty else {
                context.fill(Path(rect), with: .color(.white.opacity(0.035)))
                return
            }
            let finite = values.filter(\.isFinite)
            let lo = finite.min() ?? 0, hi = finite.max() ?? 0
            let span = hi - lo
            // Interpolate the scalar at pixel centers, then apply the shared
            // perceptual ramp. Constant tracks remain low, never a false peak.
            let columns = max(Int(ceil(size.width)), 1)
            for column in 0..<columns {
                let time = Double(column) / Double(max(columns - 1, 1)) * Double(values.count - 1)
                let first = Int(time), next = min(first + 1, values.count - 1)
                let fraction = time - Double(first)
                let value = values[first] * (1 - fraction) + values[next] * fraction
                let intensity = span > 1e-9 ? Float((value - lo) / span) : 0
                context.fill(Path(CGRect(x: CGFloat(column), y: 0, width: 1.5, height: size.height)),
                             with: .color(ActivityPalette.color(ActivityPalette.timeline(intensity))))
            }
        }
        .accessibilityLabel("\(track.name) relative activity")
    }
}
