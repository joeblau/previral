"""Convert the TRIBE v2 FmriEncoder head to CoreML (.mlpackage).

Export wrapper takes the three modality feature tensors directly (no SegmentData /
pandas in the traced path) and bakes in the average-subject predictor row, so the
CoreML model needs no subject id.

Usage:
    uv run python convert_fmri_encoder.py [--timesteps 200] [--precision fp16|fp32]
        [--out ../Models/FmriEncoder.mlpackage]
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn

from reference import load_reference_model

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = REPO_ROOT / "Models" / "FmriEncoder.mlpackage"


class FmriEncoderExport(nn.Module):
    """Tensor-only wrapper around FmriEncoderModel (average subject, eval mode).

    Inputs per modality: (B, L_layers, D, T) — the exact layout the head's
    ``aggregate_features`` consumes. Output: pooled predictions
    (B, n_outputs, n_output_timesteps).
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, text, audio, video):
        m = self.model
        feats = {"text": text, "audio": audio, "video": video}
        tensors = []
        for modality, (n_layers, dim) in m.feature_dims.items():
            data = feats[modality].to(torch.float32)
            B, _, _, T = data.shape
            # layer_aggregation == "cat": "b l d t -> b (l d) t"
            data = data.reshape(B, n_layers * dim, T).transpose(1, 2)
            data = m.projectors[modality](data)  # B, T, hidden // n_modalities
            tensors.append(data)
        x = torch.cat(tensors, dim=-1)  # extractor_aggregation == "cat"
        x = m.transformer_forward(x, None)  # combiner + time_pos_embed + encoder
        x = x.transpose(1, 2)  # B, H, T
        x = m.low_rank_head(x.transpose(1, 2)).transpose(1, 2)  # B, 2048, T
        x = m.predictor(x, None)  # average-subject row; subject id unused
        return m.pooler(x)  # B, n_outputs, n_output_timesteps


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=200,
                        help="Input timesteps on the 2Hz feature grid (200 = 100s = 100 TRs).")
    parser.add_argument("--precision", choices=["fp16", "fp32"], default="fp16")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    import coremltools as ct

    model, build_args = load_reference_model()
    feature_dims = build_args["feature_dims"]
    T = args.timesteps
    print(f"build_args: {build_args}")
    print(f"exporting with B=1, T={T}, precision={args.precision}")

    wrapper = FmriEncoderExport(model).eval()

    torch.manual_seed(0)
    example = {
        m: torch.randn(1, L, D, T) for m, (L, D) in feature_dims.items()
    }

    # Sanity: wrapper must match the original SegmentData path exactly.
    from neuralset.dataloader import SegmentData
    from neuralset.segments import Segment

    batch = SegmentData(
        data=example,
        segments=[Segment(start=0.0, duration=T / 2.0, timeline="synthetic")],
    )
    with torch.inference_mode():
        ref = model(batch)
        wrapped = wrapper(example["text"], example["audio"], example["video"])
    max_diff = (ref - wrapped).abs().max().item()
    print(f"wrapper vs SegmentData path max abs diff: {max_diff:.3e}")
    assert max_diff == 0.0, "export wrapper diverges from reference path"

    with torch.inference_mode():
        traced = torch.jit.trace(
            wrapper, (example["text"], example["audio"], example["video"])
        )
        # verify trace numerics
        traced_out = traced(example["text"], example["audio"], example["video"])
    trace_diff = (traced_out - ref).abs().max().item()
    print(f"traced vs reference max abs diff: {trace_diff:.3e}")
    assert trace_diff < 1e-4, "tracing diverged from reference"

    inputs = [
        ct.TensorType(name="text_features", shape=example["text"].shape, dtype=np.float32),
        ct.TensorType(name="audio_features", shape=example["audio"].shape, dtype=np.float32),
        ct.TensorType(name="video_features", shape=example["video"].shape, dtype=np.float32),
    ]
    precision = (
        ct.precision.FLOAT16 if args.precision == "fp16" else ct.precision.FLOAT32
    )
    mlmodel = ct.convert(
        traced,
        convert_to="mlprogram",
        inputs=inputs,
        outputs=[ct.TensorType(name="predictions")],
        compute_precision=precision,
        compute_units=ct.ComputeUnit.ALL,
        minimum_deployment_target=ct.target.macOS15,
    )
    mlmodel.author = "previral (converted from facebook/tribev2, CC-BY-NC)"
    mlmodel.short_description = (
        "TRIBE v2 FmriEncoder head: multimodal 2Hz features -> fsaverage5 fMRI "
        "predictions (average subject)."
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    mlmodel.save(str(args.out))
    print(f"saved: {args.out}")

    spec = mlmodel.get_spec()
    for inp in spec.description.input:
        print("input:", inp.name, inp.type.multiArrayType)
    for out in spec.description.output:
        print("output:", out.name, out.type.multiArrayType)


if __name__ == "__main__":
    main()
