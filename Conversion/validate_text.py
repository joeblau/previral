"""Validate Models/TextEncoder.mlpackage against the canonical fp32 reference.

The reference is the deterministic fp32 replica of the neuralset HuggingFaceText
pipeline (`canonical_reference_fp32` in convert_text_encoder.py). The true bf16
neuralset path is additionally reported for context but is itself unstable
run-to-run in bf16 on CPU (see NOTES_text.md).

Usage:
    uv run python validate_text.py [--precision fp16]

Target: per-word Pearson r median > 0.99.
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from convert_text_encoder import (
    TEST_WORDS,
    TextEncoderExport,
    build_batch_inputs,
    canonical_reference_fp32,
    neuralset_reference,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_ROOT / "Models"


def pearson_stats(ref: np.ndarray, out: np.ndarray) -> tuple[np.ndarray, float]:
    """Per-word Pearson r over flattened (2, 3072) features + max abs err."""
    rs = []
    for k in range(ref.shape[0]):
        a = ref[k].reshape(-1)
        b = out[k].reshape(-1)
        a = a - a.mean()
        b = b - b.mean()
        rs.append(float((a * b).sum() / (np.sqrt((a**2).sum() * (b**2).sum()) + 1e-12)))
    return np.array(rs), float(np.abs(ref - out).max())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--precision", choices=["fp16", "fp32"], default="fp16")
    parser.add_argument("--max-len", type=int, default=2048)
    args = parser.parse_args()

    import coremltools as ct
    from transformers import AutoTokenizer
    from convert_text_encoder import MODEL_NAME

    name = "TextEncoder.mlpackage" if args.precision == "fp16" else "TextEncoder_fp32.mlpackage"
    mlpackage = MODELS_DIR / name

    canon = canonical_reference_fp32()  # (13, 2, 3072) deterministic fp32
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, truncation_side="left")

    # torch fp32 wrapper (same code as exported)
    wrapper = TextEncoderExport().eval()
    torch_out = []
    with torch.inference_mode():
        for gi in range(len(TEST_WORDS)):
            ids, mask, weights = build_batch_inputs(gi, tok, max_len=args.max_len)
            torch_out.append(wrapper(ids, mask, weights)[0].numpy())
    torch_out = np.stack(torch_out)
    r, err = pearson_stats(canon, torch_out)
    print(f"[torch-fp32] per-word r: median {np.median(r):.6f} min {r.min():.6f} "
          f"| max abs err {err:.4f}")

    # CoreML model
    mlmodel = ct.models.MLModel(str(mlpackage), compute_units=ct.ComputeUnit.ALL)
    coreml_out = []
    for gi in range(len(TEST_WORDS)):
        ids, mask, weights = build_batch_inputs(gi, tok, max_len=args.max_len)
        pred = mlmodel.predict({
            "token_ids": ids.numpy().astype(np.int32),
            "attention_mask": mask.numpy().astype(np.int32),
            "target_weights": weights.numpy().astype(np.float32),
        })["text_features"]
        coreml_out.append(np.asarray(pred, dtype=np.float32)[0])
    coreml_out = np.stack(coreml_out)
    print(f"canonical shape {canon.shape}, coreml shape {coreml_out.shape}")
    assert coreml_out.shape == canon.shape

    r, err = pearson_stats(canon, coreml_out)
    print(f"[coreml-{args.precision}] per-word r: median {np.median(r):.6f} "
          f"min {r.min():.6f} | max abs err {err:.4f}")
    for k, (w, _, _) in enumerate(TEST_WORDS):
        print(f"  {w:>15}: r {r[k]:.6f}")
    passed = np.median(r) > 0.99
    print(f"[text/{args.precision}] {'PASS' if passed else 'FAIL'} "
          f"(target median r > 0.99)")

    # informational: CoreML vs the true bf16 neuralset path (unstable in bf16)
    ref_bf16 = neuralset_reference()
    r16, err16 = pearson_stats(ref_bf16, coreml_out)
    print(f"[info] coreml vs true bf16 neuralset path: per-word r median "
          f"{np.median(r16):.6f} min {r16.min():.6f} | max abs err {err16:.4f}")
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
