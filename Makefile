.DEFAULT_GOAL := run

.PHONY: xcodegen stop build run verify-brain verify-multimodal

xcodegen:
	xcodegen generate

stop:
	xcrun swift Scripts/StopPreviral.swift

# Both prerequisites must finish before Xcode can replace the app's resources,
# including when make is invoked with -j.
build: stop xcodegen
	xcodebuild -scheme previral -destination 'platform=macOS' build

run: build
	@app_path="$$(xcodebuild -scheme previral -destination 'platform=macOS' -showBuildSettings -json | python3 -c 'import json, os, sys; settings = next(item["buildSettings"] for item in json.load(sys.stdin) if item["target"] == "previral"); print(os.path.join(settings["BUILT_PRODUCTS_DIR"], settings["FULL_PRODUCT_NAME"]))')" && \
		test -d "$$app_path" && \
		open "$$app_path"

verify-brain:
	mkdir -p build
	xcrun swiftc -swift-version 6 -O -o build/verify-brain Previral/ActivityPalette.swift Previral/BrainMesh.swift Previral/BrainActivity.swift Previral/MultimodalActivity.swift Previral/BrainView.swift Previral/TimelineView.swift Verification/BrainRenderingCheck.swift
	./build/verify-brain

verify-multimodal:
	mkdir -p build
	xcrun swiftc -swift-version 6 -O -o build/verify-multimodal Previral/ActivityPalette.swift Previral/BrainMesh.swift Previral/BrainActivity.swift Previral/MultimodalActivity.swift Previral/AnalysisCache.swift Previral/TimelineView.swift Previral/FmriEncoderModel.swift Verification/MultimodalCheck.swift
	./build/verify-multimodal
