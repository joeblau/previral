import SwiftUI
import simd

/// Shared perceptual ramps. Interpolation in OKLab avoids muddy transitions
/// and the false boundaries produced by a full HSV rainbow.
enum ActivityPalette {
    static let timelineStops: [SIMD3<Float>] = [
        SIMD3(0.075, 0.085, 0.13), SIMD3(0.27, 0.16, 0.27),
        SIMD3(0.55, 0.20, 0.31), SIMD3(0.82, 0.32, 0.26),
        SIMD3(0.97, 0.57, 0.29), SIMD3(0.99, 0.79, 0.46),
        SIMD3(1.0, 0.95, 0.76),
    ]
    static let activationStops: [SIMD3<Float>] = [
        SIMD3(0.48, 0.12, 0.10), SIMD3(0.78, 0.20, 0.12),
        SIMD3(0.96, 0.42, 0.14), SIMD3(1.0, 0.70, 0.23),
        SIMD3(1.0, 0.93, 0.65),
    ]

    static func timeline(_ value: Float) -> SIMD4<Float> { sample(value, stops: timelineStops) }

    /// Only positive predicted responses receive heat. Negative / weak values
    /// retain anatomy; the medial wall is masked separately by BrainActivity.
    static func activation(_ value: Float, anatomy: SIMD4<Float>) -> SIMD4<Float> {
        let t = clamp(value)
        let strength = clamp((t - 0.25) / 0.75)
        let blend = clamp((t - 0.25) / 0.25)
        let opacity = blend * blend * (3 - 2 * blend)
        return anatomy * (1 - opacity) + sample(strength, stops: activationStops) * opacity
    }

    static func color(_ rgba: SIMD4<Float>) -> Color {
        Color(.sRGB, red: Double(rgba.x), green: Double(rgba.y), blue: Double(rgba.z), opacity: Double(rgba.w))
    }

    /// Additive RGB: text = red, audio = green, video = blue. Shared responses
    /// become yellow, cyan, magenta, or white; weak responses retain anatomy.
    static func multimodal(text: Float, audio: Float, video: Float, anatomy: SIMD4<Float>) -> SIMD4<Float> {
        let rgb = SIMD3(clamp(text), clamp(audio), clamp(video))
        let peak = max(rgb.x, rgb.y, rgb.z)
        let blend = clamp(peak / 0.35)
        let opacity = blend * blend * (3 - 2 * blend)
        let color = SIMD4(rgb.x, rgb.y, rgb.z, Float(1))
        return anatomy * (1 - opacity) + color * opacity
    }

    static func sample(_ value: Float, stops: [SIMD3<Float>]) -> SIMD4<Float> {
        let position = clamp(value) * Float(stops.count - 1)
        let index = min(Int(position), stops.count - 2)
        let fraction = position - Float(index)
        let lab = toOKLab(stops[index]) * (1 - fraction) + toOKLab(stops[index + 1]) * fraction
        let rgb = fromOKLab(lab)
        return SIMD4(clamp(rgb.x), clamp(rgb.y), clamp(rgb.z), 1)
    }

    private static func clamp(_ x: Float) -> Float { x.isFinite ? max(0, min(1, x)) : 0 }

    private static func toOKLab(_ c: SIMD3<Float>) -> SIMD3<Float> {
        func linear(_ x: Float) -> Float { x <= 0.04045 ? x / 12.92 : pow((x + 0.055) / 1.055, 2.4) }
        let r = linear(c.x), g = linear(c.y), b = linear(c.z)
        let l = cbrt(0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b)
        let m = cbrt(0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b)
        let s = cbrt(0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b)
        return SIMD3(0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s,
                     1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s,
                     0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s)
    }

    private static func fromOKLab(_ c: SIMD3<Float>) -> SIMD3<Float> {
        let l = pow(c.x + 0.3963377774 * c.y + 0.2158037573 * c.z, 3)
        let m = pow(c.x - 0.1055613458 * c.y - 0.0638541728 * c.z, 3)
        let s = pow(c.x - 0.0894841775 * c.y - 1.2914855480 * c.z, 3)
        func gamma(_ x: Float) -> Float { x <= 0.0031308 ? 12.92 * x : 1.055 * pow(x, 1 / 2.4) - 0.055 }
        return SIMD3(gamma(4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s),
                     gamma(-1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s),
                     gamma(-0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s))
    }
}

/// Three faces of an additive RGB cube, matching the surface color channels.
struct MultimodalLegend: View {
    var body: some View {
        VStack(spacing: 0) {
            Text("Video").foregroundStyle(Color(red: 0.48, green: 0.56, blue: 1))
            HStack(alignment: .bottom, spacing: 4) {
                Text("Audio").foregroundStyle(Color(red: 0.4, green: 0.88, blue: 0.46))
                Canvas { context, size in
                    let top = CGPoint(x: size.width / 2, y: 0)
                    let left = CGPoint(x: 0, y: size.height * 0.75)
                    let right = CGPoint(x: size.width, y: size.height * 0.75)
                    let center = CGPoint(x: size.width / 2, y: size.height / 2)
                    let axes = [CGPoint(x: left.x - center.x, y: left.y - center.y),
                                CGPoint(x: right.x - center.x, y: right.y - center.y),
                                CGPoint(x: top.x - center.x, y: top.y - center.y)]
                    let channels: [SIMD3<Float>] = [SIMD3(0, 1, 0), SIMD3(1, 0, 0), SIMD3(0, 0, 1)]
                    for face in 0..<3 {
                        let a = (face + 1) % 3, b = (face + 2) % 3
                        for i in 0..<24 {
                            for j in 0..<24 {
                                func point(_ u: CGFloat, _ v: CGFloat) -> CGPoint {
                                    CGPoint(x: center.x + axes[a].x * u + axes[b].x * v,
                                            y: center.y + axes[a].y * u + axes[b].y * v)
                                }
                                let u = CGFloat(i) / 24, v = CGFloat(j) / 24
                                let x = Float(u), y = Float(v)
                                let rgb = SIMD3<Float>(repeating: (1 - x) * (1 - y))
                                    + channels[a] * x + channels[b] * y
                                var path = Path()
                                path.move(to: point(u, v))
                                path.addLine(to: point(u + 1 / 24, v))
                                path.addLine(to: point(u + 1 / 24, v + 1 / 24))
                                path.addLine(to: point(u, v + 1 / 24))
                                path.closeSubpath()
                                context.fill(path, with: .color(ActivityPalette.color(SIMD4(rgb, 1))),
                                             style: FillStyle(antialiased: false))
                            }
                        }
                    }
                }
                .frame(width: 64, height: 64)
                Text("Text").foregroundStyle(Color(red: 1, green: 0.48, blue: 0.46))
            }
        }
        .font(.system(size: 10, weight: .medium))
        .accessibilityElement(children: .ignore)
        .accessibilityLabel("Multimodality: video blue, audio green, text red. Blended colors indicate shared responses.")
    }
}

struct ActivityLegend: View {
    var body: some View {
        HStack(spacing: 7) {
            Text("Low")
            LinearGradient(colors: (0...24).map { ActivityPalette.color(ActivityPalette.timeline(Float($0) / 24)) },
                           startPoint: .leading, endPoint: .trailing)
                .frame(width: 100, height: 5)
                .clipShape(Capsule())
            Text("High")
        }
        .font(.system(size: 10))
        .foregroundStyle(.secondary)
    }
}
