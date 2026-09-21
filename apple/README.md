# Previral for macOS

Native macOS video analysis using Meta's TRIBE v2 model, with synchronized
predicted cortical activity and network timelines. Requires macOS 26+, Xcode,
and XcodeGen. Run the commands in this document from `apple/`; the root
Makefile also forwards the app build and verification targets here.

Clone with `git clone --recurse-submodules`, or run
`git submodule update --init --recursive` in an existing checkout to retrieve
the pinned TRIBE v2 source. Then run `cd apple`. Converted Core ML models and tokenizer files are
generated locally under `Models/` and are not included in the repository. See
the conversion scripts and `Conversion/NOTES_{audio,video,text}.md` for setup
and validation details.

Run `make` (or `make run`) to stop all running previral instances, build the
current source, and launch the newly built app. Old instances are stopped before
Xcode updates the app bundle, preventing old code from reading new resources.

`make build` stops old instances and builds without launching. `make stop` only
stops the app. All copies with bundle ID `com.joeblau.previral` are covered;
unresponsive instances are force-stopped after a five-second grace period.

`make verify-brain` checks the brain rendering and writes previews to
`build/brain-review/`. See `Conversion/NOTES_brain.md` for the mesh export workflow.

Choose **Multimodality** above the brain to view synchronized audio (green),
video (blue), and text (red) responses. Mixed colors show overlapping responses.
Analysis runs the existing head with each input alone and subtracts its
all-zero feature baseline, displaying positive differences on one shared scale.
These input-isolation comparisons are not additive attributions of the combined
prediction.
Text comes from the video's speech transcript; unavailable inputs contribute no
color. The timeline switches to three corresponding response tracks.

Hover over a timeline label and click its info button for a description of the
network or input, what the row measures, and how to read its colors. The button
is also available through keyboard focus and VoiceOver.

New analyses cache all three channels and transcription notes. Existing caches
still open in Activity mode; choose **Analyze for Multimodality** to regenerate
them. This adds four head passes while reusing the encoded features.
`make verify-multimodal` checks colors, interpolation, missing inputs, and cache
compatibility. Pass a compiled `FmriEncoder.mlmodelc` path to
`build/verify-multimodal` for an additional real Core ML inference check.
