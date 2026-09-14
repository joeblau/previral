"""Convert TRIBE v2's text feature extractor (Llama-3.2-3B) to CoreML.

Reproduces the exact neuralset HuggingFaceText pipeline (see NOTES_text.md):

  context string (ends with the target word)
  -> tokenize (add_special_tokens=False, LEFT pad with 128004 to batch-max length,
     batch = 4 consecutive words; truncation_side=left, never triggers at 131072)
  -> LlamaModel forward, 29 hidden states (embeddings + 28 layers)
  -> _layer_subselection to 20 layers: round(linspace(0, 28, 20))  [cache_n_layers=20]
  -> mean over the last n_target = padded_len - n_prefix tokens
     (INCLUDES left pads and overshoot into previous words — this is the true
     reference behavior, pad-id miscount: neuralset counts ==128001 but pads
     are 128004, so n_pads is always 0)
  -> group_mean: mean(hs20[9:14]), mean(hs20[14:20])  [layers 0.5/0.75/1.0]
  -> (2, 3072) per word

Weights: downloaded from the unsloth/Llama-3.2-3B mirror, sha256-verified against
the gated meta-llama/Llama-3.2-3B LFS pointers (byte-identical; the joeblau
account's gated-repo file access is pending approval).

Model I/O contract (single word per call, B=1, flexible seq len 1..2048):
  token_ids       int32  (1, L)  — left-padded with 128004 to the word's batch-max
  attention_mask  int32  (1, L)  — 0 on pads, 1 on real tokens
  target_weights  float32 (1, L) — 1/n_target on the last n_target positions
  text_features   float16 (1, 2, 3072) out

Usage:
    uv run python convert_text_encoder.py [--precision fp16] [--max-len 2048]
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_ROOT / "Models"

# sha256-verified byte-identical mirror of meta-llama/Llama-3.2-3B
MODEL_NAME = "unsloth/Llama-3.2-3B"
GATED_SHA256 = {
    "model-00001-of-00002.safetensors": "584d8d3e3f82f7964955174dfe5e3b1cf117a9d859f022cfdf7fcb884856e002",
    "model-00002-of-00002.safetensors": "4719a04514ec2f060240711b7c33ab21187cac730ecaba3040b7a0fd95a9cefb",
}

# hidden states: 29 (embeddings + 28 transformer layers); cache_n_layers=20
# subselection round(linspace(0, 28, 20)) applied inside _aggregate_tokens BEFORE
# the token mean; group_mean then runs on the 20 subselected layers.
N_HS = 29
SUBSELECT = [int(round(x)) for x in np.linspace(0, N_HS - 1, 20)]
_GROUP_IDX = [int(0.5 * 19), int(0.75 * 19), 19]  # [9, 14, 19] on 20 layers
GROUPS = [(_GROUP_IDX[0], _GROUP_IDX[1]), (_GROUP_IDX[1], _GROUP_IDX[2] + 1)]
# effective full-hidden-state indices: group1 = hs[13,15,16,18,19],
# group2 = hs[21,22,24,25,27,28]

# test words shared with validate_text.py
TEST_WORDS = [
    ("Hello", 0.0, 0.4), ("world", 0.4, 0.4), ("this", 0.8, 0.3), ("is", 1.1, 0.2),
    ("a", 1.3, 0.2), ("test", 1.5, 0.4), ("of", 1.9, 0.2), ("contextualized", 2.1, 0.6),
    ("word", 2.7, 0.3), ("embeddings", 3.0, 0.5), ("for", 3.5, 0.2), ("brain", 3.7, 0.3),
    ("encoding", 4.0, 0.5),
]
TEST_SENTENCE = (
    "Hello world this is a test of contextualized word embeddings for brain encoding."
)


def test_contexts() -> list[str]:
    """Assemble contexts exactly like AddContextToWords for the test sentence."""
    contexts = []
    for i, (w, _, _) in enumerate(TEST_WORDS):
        start = sum(len(x[0]) + 1 for x in TEST_WORDS[:i]) if i else 0
        contexts.append(TEST_SENTENCE[: TEST_SENTENCE.index(w, start) + len(w)])
    return contexts


def neuralset_reference(batch_size: int = 4) -> np.ndarray:
    """True neuralset HuggingFaceText features for the test words: (13, 2, 3072)."""
    from neuralset.events.etypes import Word
    from neuralset.extractors.text import HuggingFaceText

    ext = HuggingFaceText(
        model_name=MODEL_NAME, device="cpu", layers=[0.5, 0.75, 1.0],
        cache_n_layers=20, layer_aggregation="group_mean", token_aggregation="mean",
        batch_size=batch_size, contextualized=True, frequency=2.0, aggregation="sum",
        allow_missing=True, infra={"folder": None},
    )
    events = [
        Word(start=s, duration=d, timeline="t", text=w, context=c)
        for (w, s, d), c in zip(TEST_WORDS, test_contexts())
    ]
    t0 = TEST_WORDS[0][1]
    dur = TEST_WORDS[-1][1] + TEST_WORDS[-1][2] - t0
    return np.stack(
        [ta.data for ta in ext._get_timed_arrays(events, start=t0, duration=dur)]
    )


def canonical_reference_fp32(batch_size: int = 4) -> np.ndarray:
    """Deterministic fp32 reference following the reference loop structure exactly
    (batched forwards of `batch_size` consecutive words, bf16-free).

    The true neuralset path computes in bf16 and is NOT reproducible across runs
    on CPU (bf16 reduction-order noise amplified by LLaMA massive activations up
    to ~500; two identical runs diverge to per-word r as low as 0.76). fp32 CPU
    is bit-deterministic across processes (verified), so this is the canonical
    target. Same-process checks show the true bf16 path matches this reference
    up to bf16 noise (see NOTES_text.md).
    """
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_NAME, truncation_side="left")
    model = AutoModel.from_pretrained(MODEL_NAME, dtype=torch.float32).eval()
    contexts = test_contexts()
    feats = []
    with torch.no_grad():
        for b0 in range(0, len(TEST_WORDS), batch_size):
            bctx = contexts[b0 : b0 + batch_size]
            inputs = tok(bctx, add_special_tokens=False, return_tensors="pt",
                         padding=True, truncation=True)
            out = model(**inputs, output_hidden_states=True)
            hs = torch.stack(list(out.hidden_states))  # (29, B, L, 3072)
            L = inputs["input_ids"].shape[1]
            for i, w in enumerate(TEST_WORDS[b0 : b0 + batch_size]):
                n_pads = int((inputs["input_ids"][i] == tok.eos_token_id).sum())
                hs_i = hs[:, i]
                if n_pads:
                    hs_i = hs_i[:, :-n_pads]
                prefix = bctx[i][: -len(w[0])].rstrip()
                n_prefix = (
                    len(tok.encode(prefix, add_special_tokens=False)) if prefix else 0
                )
                n_target = max(1, L - n_pads - n_prefix)
                ws = hs_i[SUBSELECT][:, -n_target:]  # (20, n_target, 3072)
                feat = ws.mean(dim=1)  # token mean
                feats.append(
                    torch.stack([feat[a:b].mean(0) for a, b in GROUPS]).numpy()
                )
    return np.stack(feats)  # (13, 2, 3072)


class TextEncoderExport(nn.Module):
    """token ids + mask + target weights -> (1, 2, 3072) text feature."""

    def __init__(self):
        super().__init__()
        import transformers.masking_utils as mu
        from transformers import AutoModel

        # coremltools chokes on scalar ones (and has no new_ones/new_zeros);
        # True & x == x and False | x == x, so fold the init into the first mask
        def and_masks_fixed(*mask_functions):
            def and_mask(batch_idx, head_idx, q_idx, kv_idx):
                result = mask_functions[0](batch_idx, head_idx, q_idx, kv_idx)
                for mask in mask_functions[1:]:
                    result = result & mask(batch_idx, head_idx, q_idx, kv_idx).to(result.device)
                return result
            return and_mask

        def or_masks_fixed(*mask_functions):
            def or_mask(batch_idx, head_idx, q_idx, kv_idx):
                result = mask_functions[0](batch_idx, head_idx, q_idx, kv_idx)
                for mask in mask_functions[1:]:
                    result = result | mask(batch_idx, head_idx, q_idx, kv_idx).to(result.device)
                return result
            return or_mask

        mu.and_masks = and_masks_fixed
        mu.or_masks = or_masks_fixed

        self.model = AutoModel.from_pretrained(MODEL_NAME, dtype=torch.float32).eval()
        self.register_buffer(
            "sub_idx", torch.tensor(SUBSELECT, dtype=torch.long), persistent=False
        )

    def forward(self, token_ids, attention_mask, target_weights):
        out = self.model(
            input_ids=token_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        hs = torch.stack(out.hidden_states)  # (29, 1, L, 3072)
        hs = hs.index_select(0, self.sub_idx).squeeze(1)  # (20, L, 3072)
        # token mean over the last n_target positions (weights precomputed by caller)
        w = target_weights.to(hs.dtype).t()[None]  # (1, L, 1)
        feat = (hs * w).sum(dim=1)  # (20, 3072)
        g = torch.stack([feat[a:b].mean(0) for a, b in GROUPS])  # (2, 3072)
        return g.unsqueeze(0)


def build_batch_inputs(word_idx: int, tok, batch_size: int = 4, max_len: int = 2048):
    """Reference-exact model inputs for test word `word_idx`, given its batch.

    Batching mirrors the reference DataLoader: consecutive groups of `batch_size`
    words in event order, each row LEFT-padded with 128004 to the batch-max token
    length. target_weights implement the reference's `mean over the last
    n_target = padded_len - n_prefix tokens` (including pads/overshoot).

    Returns (token_ids, attention_mask, target_weights), each (1, L).
    """
    all_ctx = test_contexts()
    b0 = (word_idx // batch_size) * batch_size
    batch_idx = list(range(b0, min(b0 + batch_size, len(TEST_WORDS))))
    contexts = [all_ctx[i] for i in batch_idx]
    enc = tok(
        contexts, add_special_tokens=False, return_tensors="pt",
        padding=True, truncation=True,
    )
    ids, mask = enc["input_ids"], enc["attention_mask"]
    L = ids.shape[1]
    i = batch_idx.index(word_idx)
    w = TEST_WORDS[word_idx][0]
    prefix = contexts[i][: -len(w)].rstrip()
    n_prefix = len(tok.encode(prefix, add_special_tokens=False)) if prefix else 0
    n_pads = int((ids[i] == tok.eos_token_id).sum())  # always 0 (see docstring)
    n_target = max(1, L - n_pads - n_prefix)
    weights = torch.zeros(1, L)
    weights[0, L - n_target :] = 1.0 / n_target
    if L > max_len:
        raise ValueError(f"context {L} tokens exceeds max_len {max_len}")
    return ids[i : i + 1].int(), mask[i : i + 1].int(), weights


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--precision", choices=["fp16", "fp32"], default="fp16")
    parser.add_argument("--max-len", type=int, default=2048)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    import coremltools as ct
    from transformers import AutoTokenizer

    out_path = args.out or (
        MODELS_DIR / ("TextEncoder.mlpackage" if args.precision == "fp16"
                      else "TextEncoder_fp32.mlpackage")
    )

    wrapper = TextEncoderExport().eval()
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, truncation_side="left")

    # canonical deterministic fp32 reference (batched loop replica) vs wrapper
    canon = canonical_reference_fp32()
    mine = []
    with torch.inference_mode():
        for gi in range(len(TEST_WORDS)):
            ids, mask, weights = build_batch_inputs(gi, tok, max_len=args.max_len)
            mine.append(wrapper(ids, mask, weights)[0].numpy())
    mine = np.stack(mine)
    r = np.array([
        np.corrcoef(canon[k].reshape(-1), mine[k].reshape(-1))[0, 1]
        for k in range(len(TEST_WORDS))
    ])
    print(f"wrapper vs canonical fp32 reference: per-word r min {r.min():.6f} "
          f"median {np.median(r):.6f}, max abs diff {np.abs(canon - mine).max():.4f}")
    assert r.min() > 0.999, "wrapper diverges from canonical reference"

    # informational: vs the true bf16 neuralset path (itself run-to-run unstable
    # in bf16 on CPU — see canonical_reference_fp32 docstring)
    ref_bf16 = neuralset_reference()
    r16 = np.array([
        np.corrcoef(ref_bf16[k].reshape(-1), mine[k].reshape(-1))[0, 1]
        for k in range(len(TEST_WORDS))
    ])
    print(f"[info] wrapper vs true bf16 neuralset path: per-word r min "
          f"{r16.min():.6f} median {np.median(r16):.6f} "
          f"(bf16 run-to-run noise floor)")

    # trace on the longest test context, convert with flexible length
    ids, mask, weights = build_batch_inputs(12, tok, max_len=args.max_len)
    with torch.inference_mode():
        traced = torch.jit.trace(wrapper, (ids, mask, weights), check_trace=False)
        out0 = traced(ids, mask, weights).numpy()
    print("traced vs wrapper max abs diff:", np.abs(out0 - mine[12:13]).max())

    L = ct.RangeDim(1, args.max_len)
    mlmodel = ct.convert(
        traced,
        convert_to="mlprogram",
        inputs=[
            ct.TensorType(name="token_ids", shape=(1, L), dtype=np.int32),
            ct.TensorType(name="attention_mask", shape=(1, L), dtype=np.int32),
            ct.TensorType(name="target_weights", shape=(1, L), dtype=np.float32),
        ],
        outputs=[ct.TensorType(name="text_features")],
        compute_precision=(
            ct.precision.FLOAT16 if args.precision == "fp16" else ct.precision.FLOAT32
        ),
        compute_units=ct.ComputeUnit.ALL,
        minimum_deployment_target=ct.target.macOS15,
    )
    mlmodel.author = "previral (converted from Llama-3.2-3B via TRIBE v2; Meta Community License)"
    mlmodel.short_description = (
        "TRIBE v2 text encoder: Llama-3.2-3B contextualized word features "
        "(2 x 3072), one word-context per call."
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mlmodel.save(str(out_path))
    print(f"saved: {out_path}")
    spec = mlmodel.get_spec()
    for i in spec.description.input:
        print("input:", i.name,
              list(i.type.multiArrayType.shape) if i.type.HasField("multiArrayType") else "?",
              i.type.multiArrayType.dataType if i.type.HasField("multiArrayType") else "")
    for o in spec.description.output:
        print("output:", o.name, list(o.type.multiArrayType.shape),
              o.type.multiArrayType.dataType)


if __name__ == "__main__":
    main()
