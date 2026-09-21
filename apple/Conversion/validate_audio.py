"""Validate Models/AudioEncoder.mlpackage against the true neuralset audio pipeline.

Runs the actual neuralset Wav2VecBert extractor (config-identical: layers
[0.5, 0.75, 1.0], group_mean, frequency 2.0, norm_audio) on a synthetic 60 s
test wav, and compares against the CoreML model on the raw waveform.

Usage:
    uv run python validate_audio.py [--precision fp16] [--duration 60]

Target: per-dim Pearson r median > 0.99.
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from convert_audio_encoder import (
    AudioEncoderExport,
    SR,
    make_test_wav,
    neuralset_reference,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_ROOT / "Models"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--precision", choices=["fp16", "fp32"], default="fp16")
    parser.add_argument("--duration", type=float, default=60.0)
    args = parser.parse_args()

    import coremltools as ct

    name = "AudioEncoder.mlpackage" if args.precision == "fp16" else "AudioEncoder_fp32.mlpackage"
    mlpackage = MODELS_DIR / name

    wav = make_test_wav(args.duration)
    ref = neuralset_reference(wav, args.duration)  # (2, 1024, n_out) fp32

    # torch wrapper fp32 (same code as exported) for an fp32 sanity datapoint
    wrapper = AudioEncoderExport(args.duration).eval()
    with torch.inference_mode():
        torch_out = wrapper(torch.from_numpy(wav)[None])[0].numpy()

    mlmodel = ct.models.MLModel(str(mlpackage), compute_units=ct.ComputeUnit.ALL)
    coreml_out = np.asarray(
        mlmodel.predict({"waveform": wav[None].astype(np.float32)})["audio_features"],
        dtype=np.float32,
    )[0]

    print(f"reference (neuralset) shape: {ref.shape}")
    print(f"torch wrapper shape:         {torch_out.shape}")
    print(f"coreml output shape:         {coreml_out.shape}")
    assert coreml_out.shape == ref.shape == torch_out.shape

    for label, out in (("torch-fp32", torch_out), ("coreml-fp16", coreml_out)):
        max_err = np.abs(out - ref).max()
        # per-dimension Pearson r across time for each (layer, dim)
        a = ref.reshape(2 * 1024, -1)
        b = out.reshape(2 * 1024, -1)
        a = a - a.mean(axis=1, keepdims=True)
        b = b - b.mean(axis=1, keepdims=True)
        r = (a * b).sum(1) / (np.sqrt((a**2).sum(1) * (b**2).sum(1)) + 1e-12)
        print(
            f"[{label}] per-dim Pearson r: median {np.median(r):.6f} "
            f"min {r.min():.6f} | max abs err {max_err:.4f} "
            f"(ref std {ref.std():.4f}, out std {out.std():.4f})"
        )
        if label == "coreml-fp16":
            passed = np.median(r) > 0.99
            print(f"[audio/{args.precision}] {'PASS' if passed else 'FAIL'} "
                  f"(target median r > 0.99)")
            raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
