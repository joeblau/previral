# Audio encoder (w2v-bert-2.0) — conversion notes for M4 / Swift

## What the reference pipeline does (neuralset `Wav2VecBert`, config-verified)

Config (`cache/tribev2/config.yaml` → `data.audio_feature`): model
`facebook/w2v-bert-2.0`, `layers: [0.5, 0.75, 1.0]`, `layer_aggregation: group_mean`,
`token_aggregation: mean` (unused for audio — only text/image/video call
`_aggregate_tokens`), `cache_n_layers: 20` (**no-op for audio**: layer subselection
only happens inside `_aggregate_tokens`, which audio never calls — aggregation runs
on all 25 hidden states directly), `frequency: 2.0`, `norm_audio: true`,
`normalized: true`, `aggregation: sum`, `event_types: Audio`, `allow_missing: true`.

Exact chain per audio event (≤ 60 s after chunking, see below):

1. Decode, **downmix to mono** (mean over channels), **resample to 16 kHz**
   (julius `ResampleFrac`, integer rates only).
2. `norm_audio`: `wav = (wav - mean) / (1e-8 + std)` over the whole event.
   **This is a no-op after the feature extractor's own per-mel-bin CMVN** (any
   waveform scale/offset becomes a per-bin constant in log-mel domain, which CMVN
   subtracts exactly; verified: HF(norm) vs HF(raw) max diff 8.6e-6). The exported
   CoreML model therefore takes the **raw** waveform — no normalization needed
   Swift-side.
3. `SeamlessM4TFeatureExtractor` fbank (16 kHz): waveform ×2^15 (also CMVN-invariant,
   dropped in our replica), frames of **400 samples, hop 160, no centering**
   (frame *t* starts at sample 160·t), per-frame **DC removal**, **preemphasis
   0.97** (`y[0]=0.03·x[0]`, `y[i]=x[i]−0.97·x[i−1]`), **povey window**
   (periodic Hann^0.85), **512-pt rFFT, power spectrum**, **80-bin Kaldi mel**
   (fmin 20 Hz, fmax 8 kHz), floor **1.192092955078125e-07**, **natural log**.
   → 100 fps × 80 features.
4. Per-mel-bin **CMVN** over all frames of the event: `(x − mean)/sqrt(var + 1e-7)`
   with **ddof=1**. (neuralset passes `do_normalize=True`, which this extractor
   silently ignores — `do_normalize_per_mel_bins=True` is the default and always on.)
5. **Stride-2 pairing**: frames reshaped (T, 80) → (T//2, 160) (pairs of consecutive
   frames concatenated). → ~50 fps. Padding (value 1.0, to multiple of 2) never
   triggers for 60 s chunks (5998 frames is even).
6. `Wav2Vec2BertModel` (24 conformer layers, hidden 1024) → 25 hidden states
   (embeddings + 24 layers), no attention mask (all-valid; passing the all-ones mask
   is numerically identical, verified).
7. **2 Hz grid**: `F.interpolate(latents, round(duration·2))` with the DEFAULT
   `mode="nearest"` — i.e. a **nearest-neighbor pick**, not linear interpolation.
   Source indices for a 60 s chunk: `floor(k · 2999/120)`, k = 0..119.
   Output bin *k* nominally covers time [k/2, (k+1)/2) of the chunk.
8. **group_mean** of relative layers [0.5, 0.75, 1.0] on 25 hidden states:
   indices `[int(0.5·24), int(0.75·24), 24] = [12, 18, 24]`, groups
   `mean(hs[12:18])` and `mean(hs[18:25])` → **2 layer-features × 1024 dims**.
   (Steps 7/8 commute — both are linear/gather over independent axes.)

Result per event: `(L=2, D=1024, T=2·duration)` float32 — exactly the
`(2, 1024, T)` audio slice the FmriEncoder head consumes.

## Exported model

`Models/AudioEncoder.mlpackage` (1.1 GB) — mlprogram, fp16 compute, compute units
ALL, min macOS 15. Everything above (fbank → CMVN → pairing → 24-layer conformer →
group_mean → 2 Hz nearest-pick) is inside one model; the DFT+window+DC+preemphasis
is fused into a single `conv1d` (coremltools has no `unfold`), exact same math.

Two numerics-preserving rewrites were needed for CoreML (both validated end-to-end
against the true neuralset extractor):

1. **No waveform scaling / no `norm_audio`, no ×2^15**: any scaling changes which
   bins hit the mel floor clamp, and unscaled raw input (|wav| ≤ 1) keeps DFT power
   values ≲ 4e4 — fp16-safe (fp16 max 65504) thanks to per-frame DC removal.
2. **relative_key position bias via skew trick** (`patch_relative_key_attention`):
   the original `_apply_relative_key_position_encoding` materializes a per-layer
   (T, T, 64) embedding lookup — constant-folded by coremltools into 24 × 575.6 M
   element consts (a 27 GB package!). The bias is Toeplitz in (r−l), so it is
   computed as `matmul(q, E_ext.T)` + pad/slice rel-shift with a (2T−1, 64)
   extended table gathered from the (73, 64) embedding. Bit-identical math
   (~1e-6 vs original in fp32); package shrank 27 GB → 1.1 GB.

| name | shape | dtype | notes |
|---|---|---|---|
| `waveform` (in) | (1, 960000) | float32 | raw 16 kHz mono samples, 60 s, no normalization |
| `audio_features` (out) | (1, 2, 1024, 120) | float16 | (L=2, D=1024, T=120 @ 2 Hz) |

CoreML `predict` returns the output upcast to float32 numpy.

## Swift-side preprocessing recipe (M4)

1. AVFoundation: extract audio track, downmix mono, resample to **16 000 Hz**
   (`AVAudioConverter`). Reference uses julius polyphase resampling; any decent
   resampler is fine (CMVN removes absolute scale, so no gain normalization needed).
2. Split into chunks: reference chunks events with `ChunkEvents(max_duration=60,
   min_duration=30)`: split points on a 60 s grid from event start; points that
   would create a chunk < 30 s are dropped, so chunks are mostly exactly 60 s, but
   the tail can be anywhere in [30, 90) s (e.g. 100 s → 60+40; 80 s → single 80 s;
   130 s → 60+70).
3. Feed each 60 s chunk to the model; place the 120 output columns onto the
   segment's 2 Hz grid at `chunk_start · 2`. Overlaps never occur for one audio
   track (`aggregation: sum` only matters for overlapping events). Grid bins with
   no audio stay **zero** (`allow_missing` → zeros; the head was trained with
   zero-filled missing modality).

## Chunking caveats (deviations from reference)

- The CoreML model has a **fixed 60 s input**. Reference chunks can be up to ~90 s
  (tail merging rule above). For tail chunks in (60, 90) s the app must either
  sub-split (60 + remainder) or zero-pad — both deviate from the reference for that
  chunk because (a) the conformer attends over the whole chunk and (b) CMVN
  statistics are per-chunk. Boundary effects decay within ~±1 s of the cut; the
  bulk of a long video (full 60 s chunks) is bit-comparable to reference.
- Zero-padding a short chunk to 60 s shifts CMVN statistics (padded log-mel frames
  sit at the floor, ≈ −15.9 pre-CMVN, dragging means down) and lets attention see
  the padding. Prefer splitting over padding when the remainder is ≥ 30 s; for
  < 30 s remainders pad-and-tolerate (only the tail bins are affected).
- Full 60 s chunks processed independently match the reference exactly (the
  reference also processes each 60 s event in one independent forward pass).

## Validation

`uv run python validate_audio.py` compares the CoreML model against the TRUE
neuralset extractor (not a reimplementation) on a synthetic 60 s sweep+noise wav.

Results (60 s chunk, 2×1024 features × 120 timesteps):

| path | per-dim Pearson r median | min | max abs err |
|---|---|---|---|
| torch wrapper fp32 vs neuralset | 1.000000 | 0.999999 | 0.011 |
| **CoreML fp16 vs neuralset** | **0.999984** | **0.999810** | 0.210 |

Reference feature std is 0.571; max abs err occurs at edge/floor-affected bins.
**PASS** (target median r > 0.99).

## M5 addendum — app-side findings (golden e2e validation)

Found while building `Conversion/validate_e2e.py` (app pipeline vs PyTorch
reference at the prediction level):

- **CoreML output stride padding (Swift gotcha).** The AudioEncoder's output
  MLMultiArray is NOT contiguous: observed strides `[262144, 131072, 128, 1]`
  for logical shape `(1, 2, 1024, 120)` — rows padded 120 → 128. The
  FmriEncoder's `predictions` output is padded similarly. Always index model
  outputs via `strides`; computing offsets from `shape` silently scrambles
  rows (this produced r ≈ 0 predictions until caught by the e2e test).
- **Tail chunks: tile, don't zero-pad.** The reference processes a tail as a
  standalone sub-60 s event (own CMVN window), which the fixed 60 s CoreML
  input cannot reproduce. Measured on a 40 s tail (per-dim r across time vs
  the true chunked reference): zero-pad 0.49, sliding all-real window 0.50,
  **tiling (repeat the real audio to fill 60 s) 0.66** — tiling preserves the
  tail's log-mel mean/var exactly, keeping CMVN correct. The app tiles; only
  the `round(2·tailSeconds)` columns covering real audio are written.
  Residual deviation is inherent to the fixed-input export.
- **Decode perturbation sensitivity.** AAC container decode + resample
  (AVFoundation vs ffmpeg, waveform r ≈ 0.997) moves extractor features by
  ~0.98–0.9996 per-dim r; full 60 s chunks land at 0.9996 vs reference.
- **neuralset reference gotcha:** `Audio` events auto-detect duration AND
  frequency from the file when either is omitted — pass both `duration` and
  `frequency` (plus `offset` for nonzero starts) when constructing chunk
  events manually, or the event silently spans the whole file.

Golden e2e result (8 s synthetic clip, audio+video features, zero text):
per-TR r across vertices median **0.9945** (min 0.9916); per-vertex r across
TRs median 0.9677. **PASS** (target median r > 0.95).
