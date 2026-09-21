"""M3b: reproduce + export TRIBE v2 video feature extractor (VJEPA2 ViT-g) to CoreML.

Replicates the exact neuralset/tribev2 video feature path:
  neuralset.extractors.video.HuggingFaceVideo(
      image=HuggingFaceImage(model_name="facebook/vjepa2-vitg-fpc64-256",
                             layers=[0.5, 0.75, 1.0], layer_aggregation="group_mean",
                             token_aggregation="mean"),
      clip_duration=4.0, frequency=2.0, aggregation="sum")

Subcommands:
  make-clip   generate synthetic 8 s test clip (moving colored shapes) -> cache/test_clip.mp4
  reproduce   compute reference features with plain transformers calls   -> cache/video_reference.npy
  export      trace wrapper and convert to CoreML                        -> ../Models/VideoEncoder.mlpackage
"""

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

CACHE = Path(__file__).parent / "cache"
MODELS = Path(__file__).parent.parent / "Models"
MODEL_NAME = "facebook/vjepa2-vitg-fpc64-256"
CLIP_PATH = CACHE / "test_clip.mp4"

# tribev2 config (data.video_feature, cache/tribev2/config.yaml)
FREQUENCY = 2.0          # output grid, Hz
CLIP_DURATION = 4.0      # seconds of video per model call
NUM_FRAMES = 64          # vjepa2 frames per clip
LAYERS = [0.5, 0.75, 1.0]
CACHE_N_LAYERS = 20      # config: image.cache_n_layers (subselect before group_mean!)
CLIP_SECONDS = 8.0       # synthetic test clip duration
CLIP_FPS = 30
CLIP_W, CLIP_H = 640, 480


# ---------------------------------------------------------------- clip synthesis

def _synth_frame(t: float, w: int = CLIP_W, h: int = CLIP_H) -> np.ndarray:
    """One frame: moving colored shapes on a textured background."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    frame = np.stack(
        [xx / w * 60 + 30, yy / h * 60 + 30, (xx + yy) / (w + h) * 40 + 40], axis=-1
    )
    # red square moving left->right
    sq = 90
    cx = int(40 + (w - 140) * (t / CLIP_SECONDS))
    cy = h // 3
    mask = (np.abs(xx - cx) < sq / 2) & (np.abs(yy - cy) < sq / 2)
    frame[mask] = [220, 40, 40]
    # blue circle moving diagonally
    bx = int(w * 0.2 + w * 0.5 * (t / CLIP_SECONDS))
    by = int(h * 0.8 - h * 0.5 * (t / CLIP_SECONDS))
    mask = (xx - bx) ** 2 + (yy - by) ** 2 < 55**2
    frame[mask] = [40, 60, 230]
    # green bar sweeping vertically
    gy = int(h * (t / CLIP_SECONDS))
    mask = np.abs(yy - gy) < 12
    frame[mask] = [50, 210, 70]
    rng = np.random.default_rng(int(t * CLIP_FPS))
    frame += rng.normal(0, 4, frame.shape)
    return np.clip(frame, 0, 255).astype(np.uint8)


def make_clip() -> None:
    CACHE.mkdir(exist_ok=True)
    n = int(CLIP_SECONDS * CLIP_FPS)
    proc = subprocess.Popen(
        [
            "ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{CLIP_W}x{CLIP_H}", "-r", str(CLIP_FPS), "-i", "-",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", str(CLIP_PATH),
        ],
        stdin=subprocess.PIPE,
    )
    for k in range(n):
        proc.stdin.write(_synth_frame(k / CLIP_FPS).tobytes())
    proc.stdin.close()
    proc.wait()
    print(f"wrote {CLIP_PATH} ({n} frames @ {CLIP_FPS} fps)")


# ------------------------------------------------------------- neuralset replica

def neuralset_frame_times(duration: float) -> list[list[float]]:
    """Per output timestep, the 64 frame timestamps neuralset samples.

    HuggingFaceVideo._get_data: times = linspace(0, duration, n+1)[1:];
    for each t, frames at max(0, t - k/64*clip_duration) for k in reversed(range(64)).
    """
    expect = int(round(duration * FREQUENCY))
    times = np.linspace(0, duration, expect + 1)[1:]
    subtimes = [k / NUM_FRAMES * CLIP_DURATION for k in reversed(range(NUM_FRAMES))]
    return [[max(0.0, t - st) for st in subtimes] for t in times]


def load_model():
    from transformers import AutoModel, AutoVideoProcessor

    model = AutoModel.from_pretrained(MODEL_NAME, output_hidden_states=True)
    model.eval()
    processor = AutoVideoProcessor.from_pretrained(MODEL_NAME, do_rescale=True)
    return model, processor


def subselected_layer_indices(n_model_layers: int) -> list[int]:
    """neuralset BaseStatic._layer_subselection with cache_n_layers=20."""
    return [
        int(round(x))
        for x in np.linspace(0, n_model_layers - 1, CACHE_N_LAYERS)
    ]


def group_slices(n_sub: int = CACHE_N_LAYERS) -> list[slice]:
    """neuralset _aggregate_layers group_mean bounds on the subselected tensor."""
    bounds = np.unique([int(f * (n_sub - 1)) for f in LAYERS]).tolist()
    bounds[-1] += 1
    return [slice(l1, l2) for l1, l2 in zip(bounds[:-1], bounds[1:])]


def aggregate_hidden_states(hidden_states: tuple) -> np.ndarray:
    """neuralset _aggregate_tokens(mean) + _aggregate_layers(group_mean), with
    cache_n_layers=20 subselection (order matches video.py: subselect during
    _aggregate_tokens, group_mean later in _get_timed_arrays).

    For VJEPA2 ViT-g (41 states): selected original indices
    [0,2,4,6,8,11,13,15,17,19,21,23,25,27,29,32,34,36,38,40]; groups are
    original layers [19,21,23,25,27] and [29,32,34,36,38,40].
    Returns (2, 1408) float32.
    """
    sel = subselected_layer_indices(len(hidden_states))
    stacked = torch.stack([hidden_states[i][0] for i in sel])  # (20, tokens, D)
    tok = stacked.mean(dim=1)                                   # token mean
    latents = tok.cpu().float().numpy()
    groups = [latents[s].mean(0) for s in group_slices(latents.shape[0])]
    return np.stack(groups).astype(np.float32)


def reproduce(device: str = "cpu") -> np.ndarray:
    from moviepy import VideoFileClip

    model, processor = load_model()
    model.to(device)
    video = VideoFileClip(str(CLIP_PATH))
    print(f"clip: duration={video.duration}s fps={video.fps} size={video.size}")
    all_times = neuralset_frame_times(video.duration)
    feats = np.zeros((len(all_times), len(LAYERS) - 1, 1408), dtype=np.float32)
    for k, frame_ts in enumerate(all_times):
        frames = [video.get_frame(t) for t in frame_ts]  # neuralset _VideoImage._read
        inputs = processor(videos=list(frames), return_tensors="pt")
        pv = inputs["pixel_values_videos"]
        # neuralset _fix_pixel_values: NaN -> 0
        pv = torch.nan_to_num(pv, nan=0.0).float().to(device)
        with torch.inference_mode():
            out = model(pixel_values_videos=pv, skip_predictor=True)
        feats[k] = aggregate_hidden_states(out.hidden_states)
        print(f"  t={frame_ts[-1]:.2f}s done  (hidden layers: {len(out.hidden_states)}, "
              f"tokens: {out.hidden_states[0].shape[1]})")
        if k == len(all_times) - 1:
            np.save(CACHE / "validation_pixel_values.npy", pv.cpu().numpy())
            np.save(CACHE / "validation_frame_times.npy", np.array(frame_ts))
    video.close()
    np.save(CACHE / "video_reference.npy", feats)
    print(f"reference features {feats.shape} -> {CACHE / 'video_reference.npy'}")
    return feats


def save_inputs() -> None:
    """Processor-only: save model inputs for every 2Hz timestep (no model forward).

    Output: cache/clip_inputs.npy, shape (n_timesteps, 64, 3, 256, 256) float32,
    plus cache/clip_times.npy with the frame timestamps used per timestep.
    """
    from moviepy import VideoFileClip
    from transformers import AutoVideoProcessor

    processor = AutoVideoProcessor.from_pretrained(MODEL_NAME, do_rescale=True)
    video = VideoFileClip(str(CLIP_PATH))
    all_times = neuralset_frame_times(video.duration)
    inputs = np.zeros(
        (len(all_times), NUM_FRAMES, 3, 256, 256), dtype=np.float32
    )
    for k, frame_ts in enumerate(all_times):
        frames = [video.get_frame(t) for t in frame_ts]
        pv = processor(videos=list(frames), return_tensors="pt")["pixel_values_videos"]
        inputs[k] = torch.nan_to_num(pv, nan=0.0).float()[0].numpy()
    video.close()
    np.save(CACHE / "clip_inputs.npy", inputs)
    np.save(CACHE / "clip_times.npy", np.array(all_times))
    print(f"saved {inputs.shape} -> {CACHE / 'clip_inputs.npy'}")


# ----------------------------------------------------------------- CoreML export

class VJEPA2FeatureWrapper(nn.Module):
    """Encoder-only VJEPA2 -> tribev2 aggregated features.

    Mirrors neuralset exactly (verified against HuggingFaceImage._aggregate_tokens /
    _aggregate_layers with cache_n_layers=20):
      hidden_states = (embedding_out, layer_1_out, ..., layer_40_out)  (pre final LN,
      tie_last_hidden_states=False); token mean over all 8192 tokens; subselect 20
      equidistant states [0,2,...,40]; group_mean -> mean of original states
      [19,21,23,25,27] and [29,32,34,36,38,40].

    Input : pixel_values_videos (1, 64, 3, 256, 256) float32, processor-normalized.
    Output: features (1, 2, 1408) float32.
    """

    # original hidden-state indices selected by cache_n_layers=20 (n_model_layers=41)
    SEL = [0, 2, 4, 6, 8, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 32, 34, 36, 38, 40]

    def __init__(self, model: nn.Module):
        super().__init__()
        self.encoder = model.encoder

    def forward(self, pixel_values_videos: torch.Tensor) -> torch.Tensor:
        hs = self.encoder.embeddings(pixel_values_videos)
        states = [hs]
        for layer in self.encoder.layer:
            hs = layer(hs, None)[0]
            states.append(hs)
        sel = torch.tensor(self.SEL)
        stacked = torch.stack(states)          # (41, B, tokens, D)
        tok = stacked.mean(dim=2)              # (41, B, D)
        sub = tok[sel]                         # (20, B, D)
        g1 = sub[9:14].mean(dim=0)             # (B, D)
        g2 = sub[14:20].mean(dim=0)
        return torch.stack([g1, g2], dim=1)    # (B, 2, D)


def export() -> None:
    import coremltools as ct

    model, _ = load_model()
    wrapper = VJEPA2FeatureWrapper(model).eval()

    # sanity: wrapper must match the neuralset aggregation on the validation clip
    pv = torch.from_numpy(np.load(CACHE / "validation_pixel_values.npy")).float()
    ref = np.load(CACHE / "video_reference.npy")
    with torch.inference_mode():
        out = wrapper(pv)[0].numpy()
    err = np.abs(out - ref[-1]).max()
    print(f"wrapper vs neuralset-path max abs err: {err:.3e}")
    assert err < 1e-4, "wrapper does not reproduce neuralset features"

    print("tracing (this materializes 40 layers on a 1.2B-param model)...")
    with torch.inference_mode():
        traced = torch.jit.trace(wrapper, pv)
    print("converting to CoreML mlprogram fp16...")
    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="pixel_values_videos", shape=pv.shape, dtype=np.float32)],
        outputs=[ct.TensorType(name="features", dtype=np.float32)],
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.macOS15,
    )
    mlmodel.author = "previral (TRIBE v2 video encoder: facebook/vjepa2-vitg-fpc64-256)"
    mlmodel.short_description = (
        "VJEPA2 ViT-g encoder, 64-frame 4s clip @256px -> (2, 1408) tribev2 video features "
        "(token mean + group_mean of relative layers [0.5, 0.75, 1.0])"
    )
    MODELS.mkdir(exist_ok=True)
    out_path = MODELS / "VideoEncoder.mlpackage"
    mlmodel.save(str(out_path))
    print(f"saved {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cmd", choices=["make-clip", "save-inputs", "reproduce", "export", "all"])
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    if args.cmd in ("make-clip", "all"):
        make_clip()
    if args.cmd in ("save-inputs", "all"):
        save_inputs()
    if args.cmd in ("reproduce", "all"):
        reproduce(device=args.device)
    if args.cmd in ("export", "all"):
        export()


if __name__ == "__main__":
    sys.exit(main())
