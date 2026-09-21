.DEFAULT_GOAL := run

.PHONY: xcodegen stop build run verify-brain verify-multimodal web-dev web-build web-preview web-deploy

xcodegen stop build run verify-brain verify-multimodal:
	$(MAKE) -C apple $@

web-dev:
	npm --prefix workers/web run dev

web-build:
	npm --prefix workers/web run build:worker

web-preview:
	npm --prefix workers/web run preview

web-deploy:
	npm --prefix workers/web run deploy
