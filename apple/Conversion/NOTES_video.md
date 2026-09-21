# M3b — Video encoder (VJEPA2 ViT-g) → CoreML

Model: `facebook/vjepa2-vitg-fpc64-256` (~1.2B params, 40 transformer layers, hidden 1408,
tubelet 2×16×16). CoreML export: `Models/VideoEncoder.mlpackage` (mlprogram, fp16 compute,
fp32 in/out, min target macOS 15).

Reference path replicated from `neuralset.extractors.video.HuggingFaceVideo` +
`HuggingFaceImage`, using the deployed `tribev2/config.yaml` (`data.video_feature`):

| Config key | Value |
|---|---|
| frequency | 2.0 Hz output grid |
| clip_duration | 4.0 s per model call |
| num_frames | 64 (vjepa2 default; frames spaced 1/16 s inside the window) |
| image.layers | [0.5, 0.75, 1.0] (relative) |
| image.cache_n_layers | **20** — subselect 20 equidistant hidden states *before* group_mean |
| image.layer_aggregation | group_mean → 2 feature rows |
| image.token_aggregation | mean over all 8192 tokens |
| aggregation | sum (event-grid overlap; irrelevant per timestep) |

## Exact feature math (verified against neuralset on instrumented tensors)

`VJEPA2Model(output_hidden_states=True)` yields **41** states: index 0 = patch-embedding
output, indices 1–40 = outputs of transformer layers 1–40 (last one **pre** final LayerNorm,
`tie_last_hidden_states=False`). Each state is (1, 8192, 1408).

1. Token mean over all 8192 tokens → 41 × 1408.
2. `cache_n_layers=20` subselection: `int(round(x)) for x in linspace(0, 40, 20)` →
   original state indices **[0, 2, 4, 6, 8, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 32, 34, 36, 38, 40]**.
3. group_mean with relative layers [0.5, 0.75, 1.0] on the 20-row tensor: bounds
   [9, 14, 20] →
   - row 0 = mean of original states **[19, 21, 23, 25, 27]**
   - row 1 = mean of original states **[29, 32, 34, 36, 38, 40]**

Output per timestep: **(2, 1408) float32**.

## CoreML I/O contract

- Input `pixel_values_videos`: **(1, 64, 3, 256, 256) float32** — 64 frames, frame-major,
  RGB channels-first per frame, already resize/crop/normalized (recipe below).
- Output `features`: **(1, 2, 1408) float32** — the two group-mean layer features for ONE
  2 Hz grid timestep (the end time of the 4 s window).
- One CoreML call = one timestep. Assemble head input `video_features` as
  `(1, 2, 1408, T)` by writing each call's output into time column i.

## Swift-side preprocessing recipe (must match exactly)

For a video of duration D seconds, the 2 Hz grid has N = round(D × 2) timesteps at
end-times `t_i = D × i / N` for i = 1…N (i.e. 0.5 s, 1.0 s, … for an 8 s clip).
For each timestep:

1. **Frame times**: 64 frames at `max(0, t_i − (63−j)/16)` for j = 0…63 — i.e. the window
   `[t_i − 3.9375, t_i]` sampled every 1/16 s (62.5 ms). Clamped-to-0 for the first ~4 s.
   Decode the frame whose interval contains that timestamp (AVFoundation
   `copyNextSampleBuffer`-style nearest/prev-frame semantics; at t == D use the last frame).
2. **Per frame** (VJEPA2VideoProcessor, `do_rescale=True`):
   - Resize so the **shortest edge = 292 px**, aspect preserved, **bilinear**;
     e.g. 640×480 → 389×292.
   - **Center crop** to 256×256.
   - Scale pixels by 1/255, then normalize per RGB channel:
     mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225].
   - Replace any NaN with 0 (uniform-color safeguard; normally a no-op).
3. Stack frames → tensor (1, 64, 3, 256, 256) float32, call the model, write output
   column i.

## Reproduce / validate

```bash
cd Conversion
uv run python convert_video_encoder.py make-clip    # synthetic 8 s clip -> cache/test_clip.mp4
uv run python convert_video_encoder.py save-inputs  # processor inputs -> cache/clip_inputs.npy
uv run python convert_video_encoder.py reproduce    # PyTorch reference -> cache/video_reference.npy
uv run python convert_video_encoder.py export       # -> ../Models/VideoEncoder.mlpackage
uv run python validate_video.py                     # PASS/FAIL, Pearson r, abs error
```

## Validation results

`uv run python validate_video.py` (16 timesteps of the synthetic 8 s clip,
CoreML fp16 vs PyTorch fp32 reference):

```
inputs:  pixel_values_videos [1, 64, 3, 256, 256] float32
outputs: features [1, 2, 1408] float32
per-dim Pearson r: median=0.999995 min=0.999700   (target median > 0.99)  PASS
abs error: max=3.45e-01 mean=1.88e-03  (reference |max|=517.3, rel max err=6.7e-04)
wrapper (pre-conversion) vs neuralset aggregation path: max abs err = 0.0 (bit-exact)
```

The absolute errors look large only because group_mean features are sums of
token-meaned states with magnitudes up to ~500; relative error is ~7e-4, normal for
fp16 compute.

## Conversion notes

- Single-model conversion worked on the first attempt — no palettization, splitting,
  or flexible shapes needed. ~10 800 MIL ops, conversion ~1 min on M-series, package 1.9 GB
  (encoder only; the VJEPA2 predictor submodule is excluded from the trace).
- Only tracer warnings (all benign for fixed 64-frame input): the `num_frames <
  tubelet_size` branch, SDPA `is_causal` check, and the constant layer-index tensor.
- Attention uses SDPA → CoreML `scaled_dot_product_attention`, hence the macOS 15
  minimum target.
- moviepy `get_frame(8.0)` at exact end-of-clip logs a warning and returns the last
  valid frame — this matches neuralset behavior (same code path).
- Runtime: PyTorch CPU fp32 ≈ 40 s/clip; CoreML ≈ 7 s/clip on this machine.
