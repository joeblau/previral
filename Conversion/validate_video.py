"""M3b validation: PyTorch (neuralset replica) vs CoreML VideoEncoder on the same clip.

Runs Models/VideoEncoder.mlpackage on every 2Hz timestep of cache/test_clip.mp4
(inputs precomputed by convert_video_encoder.py save-inputs / reproduce) and compares
against the reference features cache/video_reference.npy.

Metrics over all timesteps: per-dimension Pearson r (median must be > 0.99),
max/mean absolute error. Prints PASS/FAIL.
"""

import sys
from pathlib import Path

import numpy as np

CACHE = Path(__file__).parent / "cache"
MODELS = Path(__file__).parent.parent / "Models"


def pearson_per_dim(pred: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """pred/ref: (n_samples, n_dims). Returns (n_dims,) Pearson correlations."""
    p = pred - pred.mean(axis=0, keepdims=True)
    r = ref - ref.mean(axis=0, keepdims=True)
    denom = np.sqrt((p**2).sum(axis=0) * (r**2).sum(axis=0))
    denom[denom == 0] = np.nan
    return (p * r).sum(axis=0) / denom


def main() -> int:
    import coremltools as ct

    ref = np.load(CACHE / "video_reference.npy")           # (T, 2, 1408)
    inputs = np.load(CACHE / "clip_inputs.npy")            # (T, 64, 3, 256, 256)
    print(f"reference {ref.shape}, inputs {inputs.shape}")

    mlmodel = ct.models.MLModel(str(MODELS / "VideoEncoder.mlpackage"))
    spec = mlmodel.get_spec()
    print("inputs:", [(i.name, list(i.type.multiArrayType.shape)) for i in spec.description.input])
    print("outputs:", [(o.name, list(o.type.multiArrayType.shape)) for o in spec.description.output])

    preds = np.zeros_like(ref)
    for k in range(inputs.shape[0]):
        out = mlmodel.predict({"pixel_values_videos": inputs[k : k + 1]})
        preds[k] = np.asarray(out["features"], dtype=np.float32)[0]
        if k == 0:
            print(f"first predict ok, output shape {np.asarray(out['features']).shape}")

    flat_p = preds.reshape(-1, preds.shape[-1])  # (T*2, 1408)
    flat_r = ref.reshape(-1, ref.shape[-1])
    r_per_dim = pearson_per_dim(flat_p, flat_r)
    abs_err = np.abs(flat_p - flat_r)

    med_r = float(np.nanmedian(r_per_dim))
    min_r = float(np.nanmin(r_per_dim))
    max_err = float(abs_err.max())
    mean_err = float(abs_err.mean())
    ref_scale = float(np.abs(flat_r).max())

    print(f"per-dim Pearson r: median={med_r:.6f} min={min_r:.6f} "
          f"(target median > 0.99)")
    print(f"abs error: max={max_err:.4e} mean={mean_err:.4e} "
          f"(ref |max|={ref_scale:.2f}, rel max err={max_err / ref_scale:.3e})")

    ok = med_r > 0.99
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
