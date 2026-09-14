"""Validate converted CoreML models against the PyTorch reference.

Usage:
    uv run python validate.py --stage head [--precision fp16|fp32] [--timesteps 200]

--stage head: FmriEncoder head — fixed-seed synthetic feature tensors, per-vertex
Pearson r (median/min), max abs error, output shape. Target: median r > 0.999.
Other stages (audio/video/text/e2e) belong to later milestones.
"""

import argparse
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_ROOT / "Models"


def validate_head(precision: str, timesteps: int, seed: int = 1234) -> bool:
    import coremltools as ct

    from convert_fmri_encoder import FmriEncoderExport
    from reference import load_reference_model

    model, build_args = load_reference_model()
    wrapper = FmriEncoderExport(model).eval()

    torch.manual_seed(seed)
    inputs = {
        m: torch.randn(1, L, D, timesteps)
        for m, (L, D) in build_args["feature_dims"].items()
    }
    with torch.inference_mode():
        ref = wrapper(inputs["text"], inputs["audio"], inputs["video"]).numpy()

    suffix = "" if precision == "fp16" else f"_{precision}"
    mlpackage = MODELS_DIR / f"FmriEncoder{suffix}.mlpackage"
    mlmodel = ct.models.MLModel(str(mlpackage), compute_units=ct.ComputeUnit.ALL)
    coreml_out = mlmodel.predict(
        {
            "text_features": inputs["text"].numpy(),
            "audio_features": inputs["audio"].numpy(),
            "video_features": inputs["video"].numpy(),
        }
    )["predictions"]
    coreml_out = np.asarray(coreml_out, dtype=np.float32)

    print(f"reference output shape: {ref.shape} dtype {ref.dtype}")
    print(f"coreml    output shape: {coreml_out.shape} dtype {coreml_out.dtype}")
    assert coreml_out.shape == ref.shape, (
        f"shape mismatch: {coreml_out.shape} vs {ref.shape}"
    )

    ref_flat = ref[0].T  # (T_out, n_vertices)
    out_flat = coreml_out[0].T
    max_abs_err = np.abs(out_flat - ref_flat).max()

    # Per-vertex Pearson r across output timesteps.
    ref_c = ref_flat - ref_flat.mean(axis=0, keepdims=True)
    out_c = out_flat - out_flat.mean(axis=0, keepdims=True)
    num = (ref_c * out_c).sum(axis=0)
    den = np.sqrt((ref_c**2).sum(axis=0) * (out_c**2).sum(axis=0)) + 1e-12
    r = num / den

    median_r = float(np.median(r))
    min_r = float(r.min())
    passed = median_r > 0.999
    print(f"per-vertex Pearson r: median {median_r:.6f}  min {min_r:.6f}")
    print(f"max abs error: {max_abs_err:.6f}")
    print(f"reference std: {ref.std():.6f}  coreml std: {coreml_out.std():.6f}")
    print(f"[head/{precision}] {'PASS' if passed else 'FAIL'} (target median r > 0.999)")
    return passed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=["head", "audio", "video", "text", "e2e"],
        required=True,
    )
    parser.add_argument("--precision", choices=["fp16", "fp32"], default="fp16")
    parser.add_argument("--timesteps", type=int, default=200)
    args = parser.parse_args()

    if args.stage != "head":
        raise NotImplementedError(
            f"stage {args.stage} is a later milestone (M3+); only 'head' exists in M1"
        )
    ok = validate_head(args.precision, args.timesteps)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
