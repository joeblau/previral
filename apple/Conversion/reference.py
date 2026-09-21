"""Load the TRIBE v2 FmriEncoder reference model exactly as TribeModel.from_pretrained does.

Used by both the conversion script and the validator so the CoreML model is always
compared against the true reference path.
"""

import os
from pathlib import Path

import torch

CACHE_DIR = Path(__file__).resolve().parent / "cache" / "tribev2"

# Silence neuralset/tribev2 logging noise unless debugging.
os.environ.setdefault("TRIBE_QUIET", "1")


def load_reference_model(cache_dir: str | Path = CACHE_DIR):
    """Return (model, model_build_args) with average_subjects baked in, eval mode, CPU fp32."""
    from tribev2 import TribeModel

    cache_dir = Path(cache_dir)
    xp = TribeModel.from_pretrained(
        checkpoint_dir=cache_dir,
        cache_folder=cache_dir / "feature_cache",
        device="cpu",
    )
    model = xp._model
    assert model.config.subject_layers.average_subjects is True
    assert model.predictor.average_subjects is True
    model.eval()
    build_args = {
        "feature_dims": model.feature_dims,
        "n_outputs": model.n_outputs,
        "n_output_timesteps": model.n_output_timesteps,
    }
    return model, build_args


if __name__ == "__main__":
    model, build_args = load_reference_model()
    print("build_args:", build_args)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"parameters: {n_params/1e6:.1f}M")

    # Reference forward on synthetic features via the SegmentData path.
    from neuralset.dataloader import SegmentData
    from neuralset.segments import Segment

    torch.manual_seed(0)
    T = 200
    data = {
        m: torch.randn(1, L, D, T) for m, (L, D) in build_args["feature_dims"].items()
    }
    batch = SegmentData(
        data=data, segments=[Segment(start=0.0, duration=100.0, timeline="synthetic")]
    )
    with torch.inference_mode():
        out = model(batch)
    print("output shape:", tuple(out.shape))
    print("output stats: mean %.4f std %.4f" % (out.mean(), out.std()))
