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
