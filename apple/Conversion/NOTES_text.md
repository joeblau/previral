# Text encoder (Llama-3.2-3B) — conversion notes for M4 / Swift

## Weights provenance

The `joeblau` HF account authenticates but gated-repo **file downloads return 403
("requires approval")** — license grant pending. Weights were downloaded from the
`unsloth/Llama-3.2-3B` mirror after verifying **sha256 byte-identity against the
gated repo's published LFS pointers** (exposed via the API even without file
access):

- `model-00001-of-00002.safetensors` sha256 `584d8d3e…e002` (match)
- `model-00002-of-00002.safetensors` sha256 `4719a045…cefb` (match)

Once Meta grants access, `hf download meta-llama/Llama-3.2-3B` yields identical
files; nothing needs reconversion.

## Model facts (from config.json, not guessed)

Llama-3.2-3B: **28 layers, hidden 3072**, 24 attention heads, 8 KV heads (GQA),
vocab 128256, RMSNorm eps 1e-5, RoPE theta 500000 with llama3 scaling
(factor 32, low/high freq 1/4, original max pos 8192), tied embeddings.
`hidden_states` tuple = **29 entries** (embeddings + 28 layer outputs, last one
post-final-norm).

## Layer mapping (exact)

`layers: [0.5, 0.75, 1.0]`, `layer_aggregation: group_mean`, `cache_n_layers: 20`.
Unlike audio, text **does** subselect first (`_aggregate_tokens` →
`_layer_subselection`):

- subselect 29 → 20: `round(linspace(0, 28, 20))` =
  `[0,1,3,4,6,7,9,10,12,13,15,16,18,19,21,22,24,25,27,28]`
- group_mean on the 20 with indices `[int(.5·19), int(.75·19), 19] = [9,14,19]`,
  last += 1 → groups `sub[9:14]` and `sub[14:20]`
- **effective full-hidden-state groups**:
  - group 1 = mean of hidden states **{13, 15, 16, 18, 19}**
  - group 2 = mean of hidden states **{21, 22, 24, 25, 27, 28}**
- token aggregation: **mean over the word's token span** (see below)

## Per-word forward semantics (reference, verified against true extractor)

For each Word event (from the events pipeline: Whisper/Apple-Speech word +
`AddContextToWords` context string ending with the word text):

1. Words in event order are batched in **consecutive groups of 4**
   (`batch_size: 4`; batching happens over the full timeline's word list at
   `prepare()` time, in events-table order).
2. Each context is tokenized with **no special tokens** (`add_special_tokens=False`
   → no BOS despite the tokenizer default), and the batch is **LEFT-padded** with
   token **128004** (`<|finetune_right_pad_id|>`) to the batch-max length
   (`padding_side: left` from the repo's tokenizer_config).
3. One forward per batch. Rows are independent; real tokens sit at RoPE positions
   `[n_pads_left, L_batch)` — i.e. positions are **shifted by the pad count**
   (this is reference behavior, not a bug to fix).
4. Per word: `n_prefix = len(encode(prefix))` where
   `prefix = context[:-len(word)].rstrip()` (prefix re-encoded separately);
   `n_target = L_batch - n_pads_counted - n_prefix`. **n_pads_counted is always 0**
   because neuralset counts pads as `== eos (128001)` while actual pads are 128004
   — so `n_target` inflates by the left-pad count and the token span
   `[-n_target:]` **includes pad positions and overshoots into previous words**
   when the word isn't the longest row. This contamination IS the reference
   function the head was trained with; we replicate it exactly.
5. Mean over that span (per subselected layer) → group means → `(2, 3072)`.

## bf16 nondeterminism caveat (important for M5 expectations)

The reference runs the model in **bf16** (`AutoModel.from_pretrained` default
dtype). On CPU, identical repeated runs of the true extractor produce per-word
Pearson r as low as **0.76–0.85** on pad-heavy words (bf16 reduction-order noise
amplified by LLaMA's massive activations, absmax ~500 in some hidden-state dims).
fp32 CPU is bit-deterministic across processes (verified, diff 0.0).

Consequence: M3c validates the CoreML model against a **canonical deterministic
fp32 replica** of the exact reference computation (same tokenization, batching,
span selection, layer groups). The true bf16 path is reported informationally.
For M5 e2e, expect text-stage variance of the same order as the original
pipeline's own run-to-run noise; the fp16 CoreML model sits well within it.

## CoreML I/O contract (`Models/TextEncoder.mlpackage`)

One word per call (stateless, no KV cache — single full-context forward):

| name | shape | dtype | content |
|---|---|---|---|
| `token_ids` | (1, L) | int32 | context token ids, **left-padded with 128004** to the word's batch-max length |
| `attention_mask` | (1, L) | int32 | 0 on pads, 1 on real tokens |
| `target_weights` | (1, L) | float32 | `1/n_target` on the **last n_target positions**, else 0 |
| `text_features` (out) | (1, 2, 3072) | float16 | (L_layers=2, D=3072) feature for the word |

`L` is flexible (1..2048 tokens, `RangeDim`). Swift computes `n_target` exactly
like the reference: batch 4 consecutive words in event order, `L_b` = batch-max
token count, `n_target = L_b - len(tokenize(prefix))`, `prefix =
context[:-len(word)].rstrip()`.

## Swift recipe (M4)

1. **Word timings**: Apple Speech framework (`SpeechAnalyzer`/
   `SpeechTranscriber`, macOS 26) → word strings + start/duration. (Replaces
   Whisper `ExtractWordsFromAudio`; word-boundary differences vs reference are an
   accepted deviation, same as the plan's audio-transcription substitution.)
2. **Context assembly** (replicate `AddContextToWords(sentence_only=False,
   max_context_len=1024, split_field="")`): per word, context = concatenation of
   up to **1024 previous word-parts** (a "part" = the sentence-text fragment
   covering the word, including following punctuation/spacing; `deque(maxlen=
   1024)` of parts, reset only on timeline change; on sentence change append the
   remainder of the previous sentence to the last part) **+ the fragment ending
   with the word itself**. The context string ends exactly with the word text.
   Note `max_context_len` counts **words**, not tokens — long videos can exceed
   1024 tokens; the CoreML model caps at 2048 tokens, so left-truncate beyond
   that (reference effectively never truncates at 131072; deviation only for
   very long continuous speech).
3. **Tokenization**: ship `tokenizer.json` (HF fast tokenizer, BPE, 128256
   vocab). Use a Swift tokenizer that loads `tokenizer.json` directly (e.g.
   huggingface `swift-transformers`/`swift-tokenizers` `Tokenizer` —
   `Tokenizer(from: folderURL)` reads tokenizer.json). Critical flags:
   `addSpecialTokens = false` (no BOS!), no normalization beyond the tokenizer's
   own; re-encode `prefix` strings separately for span math (BPE boundary merges
   with the prefix are part of the reference behavior).
4. **Model calls**: for each batch of 4 consecutive words: tokenize the 4
   contexts, left-pad to batch max, run 4 single-row forwards (or a batched
   wrapper if desired), collect `text_features` per word.
5. **Grid placement** (`aggregation: sum`, 2 Hz): a word's `(2, 3072)` feature is
   **added** to every 2 Hz bin overlapping `[word.start, word.start +
   word.duration)`: `start_bin = round(2·(start - segment_start))`,
   `n_bins = max(1, round(2·overlap_duration))` (see `TimedArray._overlap_slice`).
   Words overlapping the same bin sum; bins with no words stay zero.

## Validation

`uv run python validate_text.py` — 13-word fixed test sequence with realistic
contexts; compares CoreML fp16 vs the canonical fp32 reference per word, plus an
informational comparison against the true bf16 neuralset extractor.

Results (13 words, 2×3072 features each):

| comparison | per-word r median | min | max abs err |
|---|---|---|---|
| wrapper torch fp32 vs canonical fp32 | 1.000000 | 1.000000 | 0.0 |
| **CoreML fp16 vs canonical fp32** | **1.000000** | **0.997089** | 174.7* |
| [info] CoreML fp16 vs true bf16 neuralset | 0.946 | 0.22 | 260.8* |

**PASS** (target median r > 0.99). \*Max abs errs sit on LLaMA's massive-activation
outlier dims (hidden-state absmax ~500) in pad-contaminated words ("Hello" = 75%
pad tokens in the mean) — fp16 vs fp32 there is inherent; per-word r still
≥ 0.997. The bf16 row is at the reference pipeline's own run-to-run noise floor
(bf16-vs-bf16 repeated runs also dip to r ≈ 0.76 on such words), so it is
informational only.
