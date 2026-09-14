# previral

Native macOS video analysis using Meta's TRIBE v2 model, with synchronized
predicted cortical activity and network timelines. Requires macOS 26+, Xcode,
and XcodeGen.

Clone with `git clone --recurse-submodules`, or run
`git submodule update --init --recursive` in an existing checkout to retrieve
the pinned TRIBE v2 source. Converted Core ML models and tokenizer files are
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
