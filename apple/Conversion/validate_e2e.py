"""M5 golden end-to-end validation: the app's full analysis pipeline vs the
PyTorch reference at the PREDICTION level.

Modality set matches the app's current capabilities exactly (the text/LLaMA
encoder is blocked on the HF gated license): real audio + video features from
the true neuralset extractors, text features zero-filled, then the FmriEncoder
head from best.ckpt. Features are zero-padded to the head's T=200 (100 s)
window like the reference dataloader, and the first ceil(duration) TRs are
compared — the app does the same in CoreML (see Conversion/NOTES_audio.md,
NOTES_video.md, and Previral/AnalysisPipeline.swift).

Usage:
    cd Conversion
    uv run python validate_e2e.py make-clip     # cache/e2e_clip.mp4 (8 s, A/V)
    uv run python validate_e2e.py reference     # -> cache/e2e_reference_predictions.npy
    # app side (Swift harness, run from the repo root):
    #   swiftc -swift-version 6 -strict-concurrency=complete -O \\
    #     -target arm64-apple-macos26.0 Previral/{BrainMesh,BrainActivity,BrainView,\\
    #     TimelineView,FmriEncoderModel,EncoderModels,AnalysisPipeline,AnalysisCache,\\
    #     AnalysisViewModel}.swift /tmp/previral-test/e2e/main.swift -o /tmp/previral-test/e2e/e2e
    #   /tmp/previral-test/e2e/e2e    # -> /tmp/previral-test/app_predictions.bin
    uv run python validate_e2e.py compare [/tmp/previral-test/app_predictions.bin]

Requires ffmpeg on PATH (muxing + audio decode of the test clip).
Target: median Pearson r > 0.95 (per-vertex across TRs and per-TR across vertices).
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

CACHE = Path(__file__).resolve().parent / "cache"
E2E_CLIP = CACHE / "e2e_clip.mp4"
E2E_AUDIO_WAV = CACHE / "e2e_audio_16k.wav"
REFERENCE_OUT = CACHE / "e2e_reference_predictions.npy"
CLIP_DURATION = 8.0
VERTEX_COUNT = 20484
DEFAULT_APP_BIN = "/tmp/previral-test/app_predictions.bin"

# --- 3-modality mode (text wired in) ---
E2E3_CLIP = CACHE / "e2e3_clip.mp4"
E2E3_SPEECH = CACHE / "e2e3_speech.aiff"
E2E3_AUDIO_WAV = CACHE / "e2e3_audio_16k.wav"
E2E3_REFERENCE_OUT = CACHE / "e2e3_reference_predictions.npy"
E2E3_FALLBACK_WORDS = CACHE / "e2e3_fallback_words.json"
DEFAULT_APP_BIN3 = "/tmp/previral-test/app_predictions3.bin"
DEFAULT_WORDS3 = "/tmp/previral-test/app_words3.json"
MODELS_DIR = Path(__file__).resolve().parent.parent / "Models"
# The clip's spoken text == convert_text_encoder.TEST_SENTENCE.
SPEECH_TEXT = (
    "Hello world this is a test of contextualized word embeddings for brain encoding."
)


def make_clip() -> None:
    """Mux the synthetic video clip with 8 s of the synthetic audio signal.

    Audio is stored as 48 kHz stereo AAC so the app side exercises its real
    extraction path (container decode + downmix + resample). The video stream
    is copied, so cache/video_reference.npy (computed from test_clip.mp4)
    remains the exact reference video features for this clip.
    """
    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg not found on PATH")

    if not (CACHE / "test_clip.mp4").exists():
        from convert_video_encoder import make_clip as make_test_clip

        make_test_clip()
    from convert_audio_encoder import SR, make_test_wav

    import soundfile as sf

    src_wav = CACHE / "e2e_audio_src.wav"
    sf.write(src_wav, make_test_wav(CLIP_DURATION), SR)

    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", str(CACHE / "test_clip.mp4"),
            "-i", str(src_wav),
            "-map", "0:v", "-map", "1:a", "-t", str(CLIP_DURATION),
            "-af", "aresample=48000,pan=stereo|c0=c0|c1=c0",
            "-c:v", "copy", "-c:a", "aac",
            str(E2E_CLIP),
        ],
        check=True,
    )
    print(f"wrote {E2E_CLIP}")


def reference(recompute_video: bool = False) -> np.ndarray:
    """True neuralset features + torch FmriEncoder head -> (20484, n_tr)."""
    import torch

    from reference import load_reference_model

    if not E2E_CLIP.exists():
        make_clip()

    # --- video features: reuse the M3 reference (identical video stream) unless asked
    if not recompute_video and (CACHE / "video_reference.npy").exists():
        video_feats = np.load(CACHE / "video_reference.npy")  # (T, 2, 1408)
        print(f"video features: reusing {CACHE / 'video_reference.npy'} {video_feats.shape}")
    else:
        video_feats = _video_reference_from_scratch()
    video_grid = video_feats.transpose(1, 2, 0)  # (2, 1408, T)

    # --- audio features: true neuralset Wav2VecBert on the clip's decoded audio
    audio_grid = _audio_reference()  # (2, 1024, T)

    n_time = round(CLIP_DURATION * 2)
    assert video_grid.shape[-1] == n_time, f"video T={video_grid.shape[-1]} != {n_time}"
    assert audio_grid.shape[-1] == n_time, f"audio T={audio_grid.shape[-1]} != {n_time}"

    # --- assemble head inputs: T=200 window, zero-padded tail, zero text
    T = 200
    text = np.zeros((1, 2, 3072, T), dtype=np.float32)  # LLaMA path blocked: zeros
    audio = np.zeros((1, 2, 1024, T), dtype=np.float32)
    video = np.zeros((1, 2, 1408, T), dtype=np.float32)
    audio[..., :n_time] = audio_grid
    video[..., :n_time] = video_grid

    model, _ = load_reference_model()
    from neuralset.dataloader import SegmentData
    from neuralset.segments import Segment

    batch = SegmentData(
        data={
            "text": torch.from_numpy(text),
            "audio": torch.from_numpy(audio),
            "video": torch.from_numpy(video),
        },
        segments=[Segment(start=0.0, duration=T / 2.0, timeline="e2e")],
    )
    with torch.inference_mode():
        out = model(batch)  # (1, 20484, 100)
    print(f"head output: {tuple(out.shape)}")

    n_tr = int(np.ceil(CLIP_DURATION))
    preds = out[0, :, :n_tr].numpy().astype(np.float32)  # (20484, n_tr)
    np.save(REFERENCE_OUT, preds)
    np.savez(
        CACHE / "e2e_reference_features.npz",
        audio=audio_grid,
        video=video_grid,
    )
    print(f"reference predictions {preds.shape} -> {REFERENCE_OUT}")
    return preds


def _video_reference_from_scratch() -> np.ndarray:
    """Run the true VJEPA2 + neuralset aggregation path on the e2e clip."""
    import torch
    from moviepy import VideoFileClip

    from convert_video_encoder import (
        aggregate_hidden_states,
        load_model,
        neuralset_frame_times,
    )

    model, processor = load_model()
    video = VideoFileClip(str(E2E_CLIP))
    all_times = neuralset_frame_times(video.duration)
    feats = np.zeros((len(all_times), 2, 1408), dtype=np.float32)
    for k, frame_ts in enumerate(all_times):
        frames = [video.get_frame(t) for t in frame_ts]
        pv = processor(videos=list(frames), return_tensors="pt")["pixel_values_videos"]
        pv = torch.nan_to_num(pv, nan=0.0).float()
        with torch.inference_mode():
            out = model(pixel_values_videos=pv, skip_predictor=True)
        feats[k] = aggregate_hidden_states(out.hidden_states)
        print(f"  video t={frame_ts[-1]:.2f}s done")
    video.close()
    np.save(CACHE / "video_reference.npy", feats)
    return feats


def _audio_reference() -> np.ndarray:
    """True neuralset Wav2VecBert on the clip's audio (ffmpeg-decoded to 16 kHz mono)."""
    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg not found on PATH")
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", str(E2E_CLIP),
            "-vn", "-ac", "1", "-ar", "16000", "-f", "wav",
            str(E2E_AUDIO_WAV),
        ],
        check=True,
    )

    from neuralset.events.etypes import Audio
    from neuralset.extractors.audio import Wav2VecBert

    ext = Wav2VecBert(
        model_name="facebook/w2v-bert-2.0",
        device="cpu",
        layers=[0.5, 0.75, 1.0],
        cache_n_layers=20,
        layer_aggregation="group_mean",
        token_aggregation="mean",
        frequency=2.0,
        norm_audio=True,
        normalized=True,
        aggregation="sum",
        allow_missing=True,
        infra={"folder": None},
    )
    event = Audio(start=0.0, duration=CLIP_DURATION, timeline="e2e", filepath=str(E2E_AUDIO_WAV))
    tas = list(ext._get_timed_arrays([event], start=0.0, duration=CLIP_DURATION))
    return tas[0].data  # (2, 1024, T)


def _pearson(a: np.ndarray, b: np.ndarray, axis: int) -> np.ndarray:
    a = a - a.mean(axis=axis, keepdims=True)
    b = b - b.mean(axis=axis, keepdims=True)
    denom = np.sqrt((a**2).sum(axis=axis) * (b**2).sum(axis=axis))
    denom[denom == 0] = np.nan
    return (a * b).sum(axis=axis) / denom


def compare(app_bin: str) -> int:
    ref = np.load(REFERENCE_OUT)  # (20484, n_tr)
    raw = np.fromfile(app_bin, dtype=np.float32)
    n_tr = ref.shape[1]
    if raw.size != VERTEX_COUNT * n_tr:
        print(f"app bin has {raw.size} floats, expected {VERTEX_COUNT * n_tr} "
              f"(vertexCount x trCount row-major)")
        return 1
    app = raw.reshape(VERTEX_COUNT, n_tr)

    r_per_vertex = _pearson(app, ref, axis=1)   # across TRs, per vertex
    r_per_tr = _pearson(app, ref, axis=0)       # across vertices, per TR
    abs_err = np.abs(app - ref)

    print(f"reference {ref.shape}, app {app.shape}")
    print(f"per-vertex r across TRs:  median {np.nanmedian(r_per_vertex):.6f} "
          f"min {np.nanmin(r_per_vertex):.6f} "
          f"(NaN vertices: {np.isnan(r_per_vertex).sum()})")
    print(f"per-TR r across vertices: median {np.nanmedian(r_per_tr):.6f} "
          f"min {np.nanmin(r_per_tr):.6f}")
    print(f"abs err: max {abs_err.max():.4f} mean {abs_err.mean():.6f} "
          f"(ref std {ref.std():.4f})")
    print("per-TR r:", np.array2string(r_per_tr, precision=4, floatmode="fixed"))

    ok = np.nanmedian(r_per_vertex) > 0.95 and np.nanmedian(r_per_tr) > 0.95
    print("PASS" if ok else "FAIL", "(target median r > 0.95)")
    return 0 if ok else 1


# ---------------------------------------------------------------------------
# 3-modality mode (text wired in). The clip's audio is real speech (macOS
# `say` on convert_text_encoder.TEST_SENTENCE); word timings come from the
# app's own Speech-framework transcription, dumped by the Swift harness
# (app_words3.json), so the comparison isolates tokenizer + encoder + grid
# placement, not ASR. Text features use the canonical deterministic fp32
# replica of the reference computation (the true neuralset path is bf16 and
# not run-to-run deterministic — see NOTES_text.md).
# ---------------------------------------------------------------------------


def make_clip3() -> None:
    """Speech clip: same video stream as the 2-modality clip, `say` speech audio."""
    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg not found on PATH")
    if not (CACHE / "test_clip.mp4").exists():
        from convert_video_encoder import make_clip as make_test_clip

        make_test_clip()
    subprocess.run(["say", "-o", str(E2E3_SPEECH), SPEECH_TEXT], check=True)
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", str(CACHE / "test_clip.mp4"),
            "-i", str(E2E3_SPEECH),
            "-map", "0:v", "-map", "1:a", "-t", str(CLIP_DURATION),
            "-af", "aresample=48000,pan=stereo|c0=c0|c1=c0,apad=whole_dur=" + str(CLIP_DURATION),
            "-c:v", "copy", "-c:a", "aac",
            str(E2E3_CLIP),
        ],
        check=True,
    )
    print(f"wrote {E2E3_CLIP}")

    # Fallback word timings for headless runs where Speech transcription is
    # unavailable (TCC): uniform spread of the sentence's words over the
    # speech audio. Both sides of the comparison share these, so the golden
    # test still isolates tokenizer/encoder/placement.
    import json as _json

    import soundfile as sf

    speech_seconds = sf.info(str(E2E3_SPEECH)).duration
    usable = min(speech_seconds, CLIP_DURATION)
    words = SPEECH_TEXT.split()
    t0, t1 = 0.2, max(0.2, usable - 0.2)
    per = (t1 - t0) / len(words)
    transcript = SPEECH_TEXT
    entries = []
    cursor = 0
    for i, w in enumerate(words):
        entries.append({
            "text": w,
            "start": t0 + i * per,
            "duration": per,
            "charStart": cursor,
            "charEnd": cursor + len(w),
        })
        cursor += len(w) + 1
    E2E3_FALLBACK_WORDS.write_text(_json.dumps({"text": transcript, "words": entries}, indent=2))
    print(f"wrote fallback words {E2E3_FALLBACK_WORDS}")


def _contexts_from_words(transcript_text: str, words: list[dict]) -> list[str]:
    """AddContextToWords replica: fragments tile the transcript; context =
    transcript[end of word i-1024 ..< end of word i] (see NOTES_text.md)."""
    chars = list(transcript_text)
    out = []
    for i, w in enumerate(words):
        start = words[i - 1024]["charEnd"] if i >= 1024 else 0
        out.append("".join(chars[start : w["charEnd"]]))
    return out


def _canonical_text_features(words: list[str], contexts: list[str]) -> np.ndarray:
    """Deterministic fp32 replica of the reference text computation (same
    batching/span semantics as convert_text_encoder.canonical_reference_fp32,
    generalized to an arbitrary word list)."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    from convert_text_encoder import GROUPS, MODEL_NAME, SUBSELECT

    tok = AutoTokenizer.from_pretrained(str(MODELS_DIR), truncation_side="left")
    model = AutoModel.from_pretrained(MODEL_NAME, dtype=torch.float32).eval()
    feats = []
    with torch.no_grad():
        for b0 in range(0, len(words), 4):
            bctx = contexts[b0 : b0 + 4]
            inputs = tok(bctx, add_special_tokens=False, return_tensors="pt",
                         padding=True, truncation=True)
            out = model(**inputs, output_hidden_states=True)
            hs = torch.stack(list(out.hidden_states))  # (29, B, L, 3072)
            L = inputs["input_ids"].shape[1]
            for i, w in enumerate(words[b0 : b0 + 4]):
                n_pads = int((inputs["input_ids"][i] == tok.eos_token_id).sum())  # always 0
                hs_i = hs[:, i]
                if n_pads:
                    hs_i = hs_i[:, :-n_pads]
                prefix = bctx[i][: -len(w)].rstrip()
                n_prefix = len(tok.encode(prefix, add_special_tokens=False)) if prefix else 0
                n_target = max(1, L - n_pads - n_prefix)
                ws = hs_i[SUBSELECT][:, -n_target:]
                feat = ws.mean(dim=1)
                feats.append(torch.stack([feat[a:b].mean(0) for a, b in GROUPS]).numpy())
    return np.stack(feats)  # (n_words, 2, 3072)


def reference3(words_path: str) -> np.ndarray:
    """3-modality reference: real audio + video + text features -> head."""
    import json as _json

    import torch

    from reference import load_reference_model

    if not E2E3_CLIP.exists():
        make_clip3()

    words_json = _json.loads(Path(words_path).read_text())
    words = words_json["words"]
    transcript_text = words_json["text"]
    print(f"words: {len(words)} from {words_path}")
    for w in words:
        print(f"  [{w['start']:5.2f}+{w['duration']:4.2f}] {w['text']!r}")

    # --- text features (canonical fp32 replica) + grid placement
    contexts = _contexts_from_words(transcript_text, words)
    text_feats = _canonical_text_features([w["text"] for w in words], contexts)
    print(f"text features: {text_feats.shape}")

    n_time = round(CLIP_DURATION * 2)
    text_grid = np.zeros((2, 3072, n_time), dtype=np.float32)
    for feat, w in zip(text_feats, words):
        # TimedArray._overlap_slice: round() is Python half-to-even
        start_bin = int(round(2 * w["start"]))
        n_bins = max(1, int(round(2 * w["duration"])))
        if start_bin > n_time - n_bins:
            start_bin = n_time - n_bins
        if start_bin < 0:
            continue
        text_grid[:, :, start_bin : start_bin + n_bins] += feat[:, :, None]

    # --- audio features: true neuralset Wav2VecBert on the clip's decoded audio
    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg not found on PATH")
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", str(E2E3_CLIP),
            "-vn", "-ac", "1", "-ar", "16000", "-f", "wav",
            str(E2E3_AUDIO_WAV),
        ],
        check=True,
    )
    from neuralset.events.etypes import Audio
    from neuralset.extractors.audio import Wav2VecBert

    ext = Wav2VecBert(
        model_name="facebook/w2v-bert-2.0", device="cpu",
        layers=[0.5, 0.75, 1.0], cache_n_layers=20,
        layer_aggregation="group_mean", token_aggregation="mean",
        frequency=2.0, norm_audio=True, normalized=True,
        aggregation="sum", allow_missing=True, infra={"folder": None},
    )
    event = Audio(start=0.0, duration=CLIP_DURATION, offset=0.0, frequency=16000,
                  timeline="e2e3", filepath=str(E2E3_AUDIO_WAV))
    audio_grid = list(ext._get_timed_arrays([event], start=0.0, duration=CLIP_DURATION))[0].data
    assert audio_grid.shape == (2, 1024, n_time), audio_grid.shape

    # --- video features: same copied stream as the 2-modality clip
    video_grid = np.load(CACHE / "video_reference.npy").transpose(1, 2, 0)  # (2, 1408, T)
    assert video_grid.shape == (2, 1408, n_time)

    # --- head
    T = 200
    text = np.zeros((1, 2, 3072, T), dtype=np.float32)
    audio = np.zeros((1, 2, 1024, T), dtype=np.float32)
    video = np.zeros((1, 2, 1408, T), dtype=np.float32)
    text[..., :n_time] = text_grid
    audio[..., :n_time] = audio_grid
    video[..., :n_time] = video_grid

    model, _ = load_reference_model()
    from neuralset.dataloader import SegmentData
    from neuralset.segments import Segment

    batch = SegmentData(
        data={
            "text": torch.from_numpy(text),
            "audio": torch.from_numpy(audio),
            "video": torch.from_numpy(video),
        },
        segments=[Segment(start=0.0, duration=T / 2.0, timeline="e2e3")],
    )
    with torch.inference_mode():
        out = model(batch)
    n_tr = int(np.ceil(CLIP_DURATION))
    preds = out[0, :, :n_tr].numpy().astype(np.float32)
    np.save(E2E3_REFERENCE_OUT, preds)
    np.savez(CACHE / "e2e3_reference_features.npz",
             text=text_grid, audio=audio_grid, video=video_grid)
    print(f"3-modality reference predictions {preds.shape} -> {E2E3_REFERENCE_OUT}")
    return preds


def compare3(app_bin: str) -> int:
    ref = np.load(E2E3_REFERENCE_OUT)
    raw = np.fromfile(app_bin, dtype=np.float32)
    n_tr = ref.shape[1]
    if raw.size != VERTEX_COUNT * n_tr:
        print(f"app bin has {raw.size} floats, expected {VERTEX_COUNT * n_tr}")
        return 1
    app = raw.reshape(VERTEX_COUNT, n_tr)

    r_per_vertex = _pearson(app, ref, axis=1)
    r_per_tr = _pearson(app, ref, axis=0)
    abs_err = np.abs(app - ref)

    print(f"3-modality: reference {ref.shape}, app {app.shape}")
    print(f"per-vertex r across TRs:  median {np.nanmedian(r_per_vertex):.6f} "
          f"min {np.nanmin(r_per_vertex):.6f}")
    print(f"per-TR r across vertices: median {np.nanmedian(r_per_tr):.6f} "
          f"min {np.nanmin(r_per_tr):.6f}")
    print(f"abs err: max {abs_err.max():.4f} mean {abs_err.mean():.6f} "
          f"(ref std {ref.std():.4f})")
    print("per-TR r:", np.array2string(r_per_tr, precision=4, floatmode="fixed"))
    print("note: the true neuralset text path is bf16 and not run-to-run "
          "deterministic (rerun r dips to ~0.76 on pad-heavy words); this "
          "reference is the canonical fp32 replica — see NOTES_text.md.")

    ok = np.nanmedian(r_per_vertex) > 0.95 and np.nanmedian(r_per_tr) > 0.95
    print("PASS" if ok else "FAIL", "(target median r > 0.95)")
    return 0 if ok else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("make-clip")
    ref_parser = sub.add_parser("reference")
    ref_parser.add_argument("--recompute-video", action="store_true",
                            help="rerun VJEPA2 on the clip instead of reusing video_reference.npy")
    cmp_parser = sub.add_parser("compare")
    cmp_parser.add_argument("app_bin", nargs="?", default=DEFAULT_APP_BIN)

    sub.add_parser("make-clip3")
    ref3_parser = sub.add_parser("reference3")
    ref3_parser.add_argument("--words", default=DEFAULT_WORDS3,
                             help="words JSON dumped by the Swift harness "
                                  "(or the fallback from make-clip3)")
    cmp3_parser = sub.add_parser("compare3")
    cmp3_parser.add_argument("app_bin", nargs="?", default=DEFAULT_APP_BIN3)
    args = parser.parse_args()

    if args.command == "make-clip":
        make_clip()
        return 0
    if args.command == "reference":
        reference(recompute_video=args.recompute_video)
        return 0
    if args.command == "compare":
        return compare(args.app_bin)
    if args.command == "make-clip3":
        make_clip3()
        return 0
    if args.command == "reference3":
        reference3(words_path=args.words)
        return 0
    return compare3(args.app_bin)


if __name__ == "__main__":
    sys.exit(main())
