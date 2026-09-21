import AVKit
import SwiftUI

/// AVKit's AVPlayerView wrapped for SwiftUI. Used instead of SwiftUI's
/// `VideoPlayer`, whose class metadata initialization crashes at runtime
/// with the current beta toolchain (EXC_CRASH in _AVKit_SwiftUI).
struct PlayerView: NSViewRepresentable {
    let player: AVPlayer?

    func makeNSView(context: Context) -> AVPlayerView {
        let view = AVPlayerView()
        view.controlsStyle = .inline
        view.showsFullScreenToggleButton = true
        return view
    }

    func updateNSView(_ view: AVPlayerView, context: Context) {
        if view.player !== player {
            view.player = player
        }
    }
}
