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
                            Text(displayName(track.name))
                                .font(.system(size: 11, weight: .medium))
                                .foregroundStyle(.secondary)
                                .frame(width: 112, alignment: .trailing)
                                .lineLimit(1)
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

    private func displayName(_ name: String) -> String {
        switch name {
        case "DorsalAttention": "Dorsal attention"
        case "VentralAttention": "Ventral attention"
        case "Default": "Default mode"
        default: name
        }
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
