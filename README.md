# Previral

Explore how video, sound, and language relate to predicted brain activity.
Previral is a native macOS app built with SwiftUI, Apple Core ML, and Meta's
TRIBE v2, with a Next.js landing page hosted on Cloudflare Workers.

## Repository layout

| Directory | Purpose |
| --- | --- |
| [`apple/`](apple/) | macOS app, XcodeGen project, model conversion tools, local models, and verification scripts |
| [`workers/web/`](workers/web/) | Next.js landing page using OpenNext; Cloudflare Worker `blau-previral` |

Clone with `git clone --recurse-submodules https://github.com/joeblau/previral.git`,
or run `git submodule update --init --recursive` in an existing checkout.

## macOS app

Requires macOS 26+, Xcode, and XcodeGen. Converted Core ML models and tokenizer
files are generated locally in `apple/Models/` and are not committed.
See [`apple/README.md`](apple/README.md) for model setup and app details.

```sh
make                 # stop old instances, build, and launch
make build           # build without launching
make verify-brain    # render checks and previews in apple/build/brain-review/
make verify-multimodal
```

App targets forward to `apple/Makefile`; `make -C apple` works too.

## Landing page

Requires Node.js 22+ and npm.

```sh
npm --prefix workers/web ci
make web-dev         # Next.js development server
make web-build       # production OpenNext Worker build
make web-preview     # build and run locally in the Workers runtime
make web-deploy      # build and deploy blau-previral (Cloudflare login required)
```

See [`workers/web/README.md`](workers/web/README.md) for deployment and validation.
