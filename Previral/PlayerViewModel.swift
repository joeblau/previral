import AVFoundation
import AppKit
import SwiftUI

@MainActor
@Observable
final class PlayerViewModel {
    private(set) var player: AVPlayer?
    private(set) var videoURL: URL?
    private(set) var duration: Double = 0
    private(set) var currentTime: Double = 0

    private var timeObserver: Any?

    var progress: Double {
        guard duration > 0 else { return 0 }
        return min(max(currentTime / duration, 0), 1)
    }

    func openVideo() {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [.movie, .mpeg4Movie, .quickTimeMovie]
        panel.allowsMultipleSelection = false
        guard panel.runModal() == .OK, let url = panel.url else { return }
        load(url: url)
    }

    func load(url: URL) {
        teardown()
        videoURL = url
        let player = AVPlayer(url: url)
        self.player = player
        Task {
            if let duration = try? await player.currentItem?.asset.load(.duration) {
                self.duration = CMTimeGetSeconds(duration)
            }
        }
        timeObserver = player.addPeriodicTimeObserver(
            forInterval: CMTime(seconds: 1.0 / 30.0, preferredTimescale: 600),
            queue: .main
        ) { [weak self] time in
            guard let self else { return }
            Task { @MainActor in
                self.currentTime = CMTimeGetSeconds(time)
            }
        }
        player.play()
    }

    func seek(toFraction fraction: Double) {
        guard let player, duration > 0 else { return }
        let target = CMTime(seconds: duration * fraction, preferredTimescale: 600)
        player.seek(to: target, toleranceBefore: .zero, toleranceAfter: .zero)
    }

    func togglePlayPause() {
        guard let player else { return }
        if player.timeControlStatus == .paused {
            player.play()
        } else {
            player.pause()
        }
    }

    private func teardown() {
        if let timeObserver, let player {
            player.removeTimeObserver(timeObserver)
        }
        timeObserver = nil
        player?.pause()
        player = nil
        duration = 0
        currentTime = 0
    }
}
