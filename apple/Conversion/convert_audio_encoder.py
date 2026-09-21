"""Convert TRIBE v2's audio feature extractor (facebook/w2v-bert-2.0) to CoreML.

Reproduces the exact neuralset/tribev2 audio pipeline (see NOTES_audio.md):

  raw wav 16kHz mono
  -> Kaldi-style fbank (povey window 400, hop 160, FFT 512, power 2,
     preemphasis 0.97, kaldi mel 80 bins, natural log, floor 1.1920929e-07,
     per-mel-bin CMVN with ddof=1)              [SeamlessM4TFeatureExtractor]
  -> stride-2 frame pairing -> (T//2, 160)     [SeamlessM4TFeatureExtractor]
  -> Wav2Vec2BertModel, 25 hidden states        [neuralset HuggingFaceAudio]
  -> F.interpolate(mode="nearest") to 2 Hz      [neuralset BaseAudio._get_data]
  -> group_mean of relative layers [0.5, 0.75, 1.0] on all 25 hidden states
     = mean(hs[12:18]), mean(hs[18:25])         [neuralset _aggregate_layers]
  -> output (1, 2, 1024, 120) for a 60 s chunk

Verified against the true neuralset Wav2VecBert extractor (global r > 0.9999999).
cache_n_layers=20 in the config is a no-op for audio (layer subselection only
happens inside _aggregate_tokens, which the audio extractor never calls).

Usage:
    uv run python convert_audio_encoder.py [--duration 60] [--precision fp16]
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_ROOT / "Models"
CACHE_DIR = Path(__file__).resolve().parent / "cache"

SR = 16000
FRAME_LENGTH = 400
HOP = 160
N_FFT = 512
N_BINS = 257
N_MELS = 80
MEL_FLOOR = 1.192092955078125e-07
PREEMPHASIS = 0.97
OUT_FREQ = 2.0  # Hz feature grid
# group_mean groups on the 25 hidden states for layers=[0.5, 0.75, 1.0]:
# indices = [int(0.5*24), int(0.75*24), 24] = [12, 18, 24]; last += 1
GROUPS = [(12, 18), (18, 25)]


def make_test_wav(duration: float = 60.0, seed: int = 7) -> np.ndarray:
    """Synthetic speech-like test signal: sweeping harmonic stack + noise bursts."""
    rng = np.random.default_rng(seed)
    n = int(SR * duration)
    t = np.arange(n) / SR
    f0 = 100 + 80 * np.sin(2 * np.pi * 0.2 * t)
    phase = np.cumsum(2 * np.pi * f0 / SR)
    voiced = np.sin(phase) + 0.5 * np.sin(2 * phase) + 0.25 * np.sin(3 * phase)
    env = 0.5 + 0.5 * np.sin(2 * np.pi * 3.0 * t) * (rng.random(n) > 0.3)
    return (0.2 * voiced * env + 0.05 * rng.standard_normal(n)).astype(np.float32)


def neuralset_reference(wav: np.ndarray, duration: float) -> np.ndarray:
    """Run the TRUE neuralset Wav2VecBert extractor path on a wav array."""
    import soundfile as sf

    from neuralset.events.etypes import Audio
    from neuralset.extractors.audio import Wav2VecBert

    wav_path = CACHE_DIR / "test_audio.wav"
    wav_path.parent.mkdir(exist_ok=True)
    sf.write(wav_path, wav, SR)
    ext = Wav2VecBert(
        model_name="facebook/w2v-bert-2.0",
        device="cpu",
        layers=[0.5, 0.75, 1.0],
        cache_n_layers=20,
        layer_aggregation="group_mean",
        token_aggregation="mean",
        frequency=OUT_FREQ,
        norm_audio=True,
        normalized=True,
        aggregation="sum",
        allow_missing=True,
        infra={"folder": None},
    )
    event = Audio(start=0.0, duration=duration, timeline="test", filepath=str(wav_path))
    tas = list(ext._get_timed_arrays([event], start=0.0, duration=duration))
    return tas[0].data  # (2, 1024, n_timepoints)


class KaldiFrontend(nn.Module):
    """Torch replica of SeamlessM4TFeatureExtractor fbank (exact, see NOTES).

    The 2**15 Kaldi scaling is folded away: it only adds a per-mel-bin constant
    in log domain, which the per-bin CMVN removes exactly (verified numerically).
    The DFT is NOT pre-scaled for the same reason — the mel floor clamp
    (1.1920929e-07) is applied before the log, so any scaling changes which bins
    clamp. With raw |wav| <= 1 inputs and per-frame DC removal, DFT power values
    stay ~1e4 at most: safe for fp16 without scaling.
    """

    def __init__(self, n_samples: int):
        super().__init__()
        from transformers.audio_utils import mel_filter_bank, window_function

        window = torch.from_numpy(window_function(FRAME_LENGTH, "povey", periodic=False))
        mel_filters = torch.from_numpy(
            np.asarray(
                mel_filter_bank(
                    num_frequency_bins=N_BINS,
                    num_mel_filters=N_MELS,
                    min_frequency=20,
                    max_frequency=SR // 2,
                    sampling_rate=SR,
                    norm=None,
                    mel_scale="kaldi",
                    triangularize_in_mel_space=True,
                )
            )
        )
        n = torch.arange(FRAME_LENGTH, dtype=torch.float64)
        k = torch.arange(N_BINS, dtype=torch.float64)
        ang = 2 * torch.pi * k[:, None] * n[None, :] / N_FFT  # (257, 400)
        # Per-frame dc-offset removal + preemphasis are linear ops on the raw
        # frame; compose them into the DFT matrices so the whole frontend up to
        # the power spectrum is a single conv1d (coremltools has no unfold):
        #   d = f - mean(f); p[0] = 0.03*d[0]; p[i] = d[i] - 0.97*d[i-1]
        dc = torch.eye(FRAME_LENGTH, dtype=torch.float64) - 1.0 / FRAME_LENGTH
        pre = torch.eye(FRAME_LENGTH, dtype=torch.float64)
        pre[0, 0] = 1.0 - PREEMPHASIS
        pre[1:, :-1] -= PREEMPHASIS * torch.eye(FRAME_LENGTH - 1, dtype=torch.float64)
        a = pre @ dc  # (400, 400)
        w_re = a.T @ (torch.cos(ang) * window.double()[None, :]).T  # (400, 257)
        w_im = a.T @ (-torch.sin(ang) * window.double()[None, :]).T
        weight = torch.cat([w_re, w_im], dim=1).T.float().unsqueeze(1)  # (514, 1, 400)
        self.register_buffer("dft_weight", weight.contiguous())
        self.register_buffer("mel", mel_filters.float())  # (257, 80)
        self.n_frames = 1 + (n_samples - FRAME_LENGTH) // HOP

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        # wav (B, n_samples) -> input_features (B, n_frames//2, 160)
        spec = torch.nn.functional.conv1d(
            wav.unsqueeze(1), self.dft_weight, stride=HOP
        )  # (B, 514, T)
        re, im = spec[:, :N_BINS], spec[:, N_BINS:]
        power = re * re + im * im  # (B, 257, T)
        mel = torch.clamp(self.mel.T @ power, min=MEL_FLOOR)  # (B, 80, T)
        logmel = torch.log(mel).transpose(1, 2)  # natural log, (B, T, 80)
        mean = logmel.mean(dim=1, keepdim=True)
        var = logmel.var(dim=1, unbiased=True, keepdim=True)  # ddof=1
        logmel = (logmel - mean) / torch.sqrt(var + 1e-7)
        B, T, _ = logmel.shape
        return logmel.reshape(B, T // 2, N_MELS * 2)  # stride-2 pairing


def patch_relative_key_attention() -> None:
    """Rewrite w2v-bert's relative_key position bias with the skew trick.

    The original materializes distance_embedding(distance_matrix) — a per-layer
    (T, T, 64) constant (1.15 GB fp16 at T=2999) that coremltools folds into the
    weights blob (27 GB package) and would be a huge runtime activation anyway.
    bias[b,h,l,r] = scaling * q[b,h,l,:] . E[clamp(r-l,-64,8)+64, :] is Toeplitz
    in (r-l), so it equals matmul(q, E_ext.T) + relative shift with E_ext the
    clamped-extended table. Numerically identical (verified ~1e-6 vs original).
    """
    from transformers.models.wav2vec2_bert import modeling_wav2vec2_bert as mod

    def patched(module, query, key):
        T = query.shape[2]
        E = module.distance_embedding.weight  # (73, head_size)
        k = torch.arange(2 * T - 1, device=E.device)
        dist = torch.clamp(
            k - (T - 1),
            -module.left_max_position_embeddings,
            module.right_max_position_embeddings,
        )
        E_ext = E[dist + module.left_max_position_embeddings].to(query.dtype)
        qe = torch.matmul(query, E_ext.t())  # (b, h, l, 2T-1)
        b, h, l, S = qe.shape
        x = torch.cat([qe.new_zeros((b, h, l, 1)), qe], dim=-1)  # (b,h,l,S+1)
        x = x.view(b, h, S + 1, l)
        x = x[:, :, 1:].view(b, h, l, S)  # rel-shift: [l, r] = qe[l, r-l+T-1]
        bias = x[:, :, :, : key.shape[2]]
        return query, bias * module.scaling

    mod._apply_relative_key_position_encoding = patched


class AudioEncoderExport(nn.Module):
    """Raw 16kHz waveform -> (1, 2, 1024, n_timepoints) at 2 Hz."""

    def __init__(self, duration: float = 60.0):
        super().__init__()
        from transformers import Wav2Vec2BertModel

        patch_relative_key_attention()
        self.duration = duration
        self.n_samples = int(SR * duration)
        self.frontend = KaldiFrontend(self.n_samples)
        self.model = Wav2Vec2BertModel.from_pretrained("facebook/w2v-bert-2.0").eval()
        # nearest-resample indices, extracted from F.interpolate itself so the
        # sampling rule is identical to the reference (mode="nearest")
        t_native = self.frontend.n_frames // 2
        n_out = int(round(duration * OUT_FREQ))
        idx = torch.nn.functional.interpolate(
            torch.arange(t_native, dtype=torch.float32)[None, None, :], n_out
        )[0, 0].long()
        self.register_buffer("time_idx", idx)

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        feats = self.frontend(wav)  # (B, T', 160)
        out = self.model(feats, output_hidden_states=True)
        hs = torch.stack(out.hidden_states).squeeze(1)  # (25, T', 1024)
        groups = torch.stack([hs[a:b].mean(0) for a, b in GROUPS])  # (2, T', 1024)
        x = groups.transpose(1, 2)  # (2, 1024, T')
        x = x.unsqueeze(0)[:, :, :, self.time_idx]  # (1, 2, 1024, n_out)
        return x


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--precision", choices=["fp16", "fp32"], default="fp16")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    import coremltools as ct

    duration = args.duration
    n_samples = int(SR * duration)
    out_path = args.out or (
        MODELS_DIR / ("AudioEncoder.mlpackage" if args.precision == "fp16"
                      else "AudioEncoder_fp32.mlpackage")
    )

    wrapper = AudioEncoderExport(duration).eval()
    wav = torch.from_numpy(make_test_wav(duration))

    # Reference: true neuralset extractor on the same audio.
    ref = neuralset_reference(wav.numpy(), duration)
    with torch.inference_mode():
        wrapped = wrapper(wav[None])[0].numpy()
    assert wrapped.shape == ref.shape, (wrapped.shape, ref.shape)
    diff = np.abs(wrapped - ref)
    r = np.corrcoef(wrapped.reshape(-1), ref.reshape(-1))[0, 1]
    print(f"wrapper (fp32 torch) vs neuralset: shape {wrapped.shape} "
          f"max abs diff {diff.max():.2e}, global r {r:.8f}")
    assert r > 0.9999, "wrapper diverges from neuralset reference"

    with torch.inference_mode():
        traced = torch.jit.trace(wrapper, wav[None])
        traced_out = traced(wav[None]).numpy()
    td = np.abs(traced_out - wrapped).max()
    print(f"traced vs wrapper max abs diff: {td:.2e}")
    assert td < 1e-4

    n_out = wrapped.shape[-1]
    mlmodel = ct.convert(
        traced,
        convert_to="mlprogram",
        inputs=[ct.TensorType(name="waveform", shape=(1, n_samples), dtype=np.float32)],
        outputs=[ct.TensorType(name="audio_features")],
        compute_precision=(
            ct.precision.FLOAT16 if args.precision == "fp16" else ct.precision.FLOAT32
        ),
        compute_units=ct.ComputeUnit.ALL,
        minimum_deployment_target=ct.target.macOS15,
    )
    mlmodel.author = "previral (converted from facebook/w2v-bert-2.0 via TRIBE v2, CC-BY-NC)"
    mlmodel.short_description = (
        "TRIBE v2 audio encoder: 16 kHz mono waveform (60 s) -> w2v-bert-2.0 "
        "group-mean layer features (2 x 1024) on the 2 Hz grid."
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mlmodel.save(str(out_path))
    print(f"saved: {out_path}")
    spec = mlmodel.get_spec()
    for i in spec.description.input:
        print("input:", i.name, list(i.type.multiArrayType.shape), i.type.multiArrayType.dataType)
    for o in spec.description.output:
        print("output:", o.name, list(o.type.multiArrayType.shape), o.type.multiArrayType.dataType)
    print(f"expected output: (1, 2, 1024, {n_out})")


if __name__ == "__main__":
    main()
