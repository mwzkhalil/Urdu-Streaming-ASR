# Investigation & Root-Cause Repair Report: Urdu Streaming Zipformer Transducer

## 1. Executive Summary
The initial training run (`exp/urdu_zipformer_20m/epoch-25-avg-5.pt`) suffered from severe deletion-dominated collapse (`97.22%` macro dev WER / `86.09%` CER, `100.96%` macro test WER / `90.02%` CER, emitting only 1–2 tokens at the start of an utterance and `<blk>` thereafter).

Through systematic isolation of the data, tokenizer, checkpoint initialization, `k2` pruned RNN-T lattice pruning, and streaming chunk masks, **four compounding root causes** were identified and repaired:

| Metric | Before Repair (`v1`: `epoch-25-avg-5`) | After Repair (`v2`: `epoch-14-avg-4`) | Offline Baseline (`openai/whisper-tiny`, 37.8M) |
|---|---:|---:|---:|
| **16-Clip Sanity Overfit WER / CER** | `95.98%` / `94.28%` | **`0.57%` / `0.40%`** (full) \| **`8.62%` / `7.67%`** (streaming) | — |
| **Macro Dev WER / CER (Full 769 cuts)** | `97.22%` / `86.09%` | **`44.86%` / `27.25%`** | — |
| **Pooled Dev WER / CER (Full 769 cuts)** | — | **`45.72%` / `27.48%`** | — |
| **Macro Held-Out Test WER / CER (Full 938 cuts)** | `100.96%` / `90.02%` | **`51.89%` / `34.70%`** | `78.18%` / `43.08%` |
| **Pooled Held-Out Test WER / CER (Full 938 cuts)** | — | **`53.08%` / `35.56%`** | — |
| **`test_kathbath_ur` WER / CER (182 cuts)** | `102.07%` / `90.56%` | **`46.67%` / `24.69%`** | `89.47%` / `41.54%` |
| **`test_indicvoices_ur` WER / CER (245 cuts)** | `104.83%` / `92.79%` | **`47.64%` / `33.66%`** | `69.59%` / `38.20%` |
| **`test_fleurs_ur_pk` WER / CER (200 cuts)** | `96.99%` / `84.85%` | **`53.28%` / `31.90%`** | `75.09%` / `38.91%` |
| **`test_rasa_ur` WER / CER (100 cuts)** | `97.00%` / `86.10%` | **`36.86%` / `20.75%`** | `59.38%` / `24.21%` |
| **`test_csalt_voice` WER / CER (159 cuts)** | `104.26%` / `93.49%` | **`59.40%` / `43.74%`** | `90.43%` / `57.54%` |
| **`test_urduspeech_benchmark_cs` WER / CER (52 cuts)** | `100.64%` / `92.34%` | **`67.49%` / `53.44%`** | `85.12%` / `58.08%` |
| **PyTorch vs ONNX FP32 Exact Clip Match (25 dev cuts)** | `0.22%` CER diff (on 1-word outputs) | **`25 / 25` (`100%` identical token IDs)** | — |

---

## 2. Evidence-Backed Root Cause Findings

### Root Cause 1: Severe Activation Scale Mismatch & Unanchored `k2` Pruned RNN-T Lattice (`px_grad`, `py_grad`)
1. **Evidence**:
   - In the pretrained English checkpoint (`desh2608/icefall-asr-librispeech-pruned-transducer-stateless7-streaming-small`), the trained `ScaledLinear` layers had evolved large weight standard deviations (`decoder.embedding.weight` std = `2.984`, `simple_am_proj.weight` std = `2.224`, `simple_lm_proj.weight` std = `1.805`, `joiner.encoder_proj.weight` std = `0.726`), while `encoder_out` had std = `0.0613`.
   - In `v1`, two failed initialization modes occurred:
     - First, when `joiner.encoder_proj` and `joiner.decoder_proj` were reinitialized with default PyTorch `nn.Linear` (`std = 0.044`), `joiner.encoder_proj(encoder_out)` had `std = 0.0517` (`20x` smaller than `1.015` in the English checkpoint), starving the acoustic path.
     - Second, when `simple_am_proj` was initialized with default `std = 0.0625` and `am_scale = 0.0, lm_scale = 0.25`, `simple_am_proj(encoder_out)` had `std = 0.0038` while `simple_lm_proj(decoder_out)` had `std = 15.5`.
   - Inspecting `k2.get_rnnt_prune_ranges(px_grad, py_grad, boundary, s_range=5)` on a 107-frame utterance (`S=19` tokens) revealed that the smoothed RNN-T posterior collapsed to a **degenerate L-shaped path**:
     - Frames `t = 0..1` jumped from token `u = 0` to `u = 7` (`[0..4]`, `[3..7]`).
     - Frames `t = 2..104` stayed frozen at `ranges[t] = [7, 8, 9, 10, 11]` for 103 consecutive frames!
     - Frames `t = 105..106` jumped to `[15, 16, 17, 18, 19]`.
   - Because `k2.rnnt_loss_pruned` only evaluates the joiner inside the 5-token window `ranges[b, t]`, `pruned_loss` achieved `0.0087` per frame by learning to emit tokens at `t = 0..1` and `<blk>` at `u = 7` for the rest of the utterance. Evaluating the `v1` model with `torch.zeros_like(feats)` yielded nearly the same loss (`0.3017` real vs `0.3936` zero features), proving the predictor had memorized transcript n-grams without acoustic alignment.

2. **Fix Implemented**:
   - Retained pretrained `joiner.encoder_proj.*`, `joiner.decoder_proj.*`, and `decoder.conv.weight` (`460` tensors retained; only the `7` strictly vocabulary-sized tensors reinitialized).
   - Applied **scale-matched initialization** in `src/urdu_streaming_asr/checkpoint_remap.py`: `decoder.embedding.weight` (`std = 2.984`, row `0` zeroed), `joiner.output_linear.weight` (`std = 0.208`), `simple_am_proj.weight` (`std = 2.224` so `am` logits match `encoder_out` scale), and calibrated `simple_lm_proj.weight` (`std = 0.09` so `lm` logits do not overwhelm `am` logits).
   - Added an **auxiliary CTC loss (`ctc_loss_scale = 0.50`) on `simple_am_proj(encoder_out)`** alongside `am_scale = 0.45, lm_scale = 0.05` in `scripts/train_transducer.py` and lowered the predictor learning rate (`decoder_lr_scale = 0.16`). Because CTC is strictly frame-synchronous and has no autoregressive state `u`, `simple_am_proj` is forced to align every token to its true acoustic frame `t`, which in turn anchors `k2.get_rnnt_prune_ranges` along the true diagonal (`[0..4] -> [2..6] -> [6..10] -> [9..13] -> [12..16] -> [15..19]`).

---

### Root Cause 2: Training vs. Streaming Attention Mask Mismatch (`short_chunk_threshold = 0.75`)
1. **Evidence**:
   - In upstream `zipformer.py` (line 526), `Zipformer.forward()` uses `short_chunk_threshold = 0.75`, meaning roughly 25–50% of training steps use **full-utterance bidirectional attention** (`chunk_size = x.size(0)`).
   - When the model was trained with default `short_chunk_threshold = 0.75`, full-utterance decoding achieved `0.00%` WER on the 16-clip sanity check, but causal chunk-streaming decoding (`streaming_forward`) had `20.50%` CER because the encoder relied on right-context frames unavailable during streaming.
2. **Fix Implemented**:
   - Set `encoder.short_chunk_threshold = 2.0` in `build_transducer()` (`scripts/train_transducer.py`), forcing `100%` of training forward passes to use causal chunk-wise attention masks (`subsequent_chunk_mask`).

---

### Root Cause 3: Native PyTorch Streaming Decode Truncation & `blank_bias_adjustment` Sign
1. **Evidence**:
   - In `v1`'s `streaming_greedy_decode_waveform`, features were padded by only `9` frames and `streaming_forward`'s returned `enc_out` was not sliced to `enc_out_lens[0]` on the final padded chunk (`< 23` frames), causing trailing-frame misalignment and state pollution.
   - In `evaluate_and_export.py` (`v1`), `blank_bias_adjustment = -2.5` was added to `joiner.output_linear.bias[0]` (`<blk>`) as a band-aid to force non-blank emissions from an unaligned model. While subtracting `2.5` from `logits[0]` lowered `<blk>` preference, it caused hallucinated insertions without fixing the underlying alignment.
2. **Fix Implemented**:
   - Rewrote `streaming_greedy_decode_waveform` in `scripts/train_transducer.py` and `scripts/adapt_and_smoke_check.py` to match upstream `DecodeStream` (`pad_length = 7`, `tail_pad_len = decode_chunk_len = 32`, minimum chunk length `23` padded with `log(1e-10)`, and iterating strictly over `range(int(enc_out_lens[0].item()))`).
   - Removed the artificial blank bias hack (`blank_bias_adjustment = 0.0` default in `evaluate_and_export.py`).

---

### Root Cause 4: Data Volume & UrduSpeech Subset Selection
1. **Evidence**:
   - In `v1`, `acquire_and_prepare.py` iterated `corpus_jsonls` in alphabetical order (`corpus/US-CS/...` came before `corpus/US-Std/...`), so all `279` UrduSpeech clips came from `US-CS` and `0` from `US-Std`. Furthermore, only `2,253` unique training clips (`3.67h`) were included.
   - Auditing training clips with `openai/whisper-tiny` confirmed that `Kathbath`, `Rasa`, and `UrduSpeech` audio/transcript alignments were accurate, while `IndicVoices` had ~75% well-aligned conversational/extempore segments and ~25% noisy/unaligned crowd segments.
2. **Fix Implemented**:
   - Expanded the verified training manifest to **`3,907` unique accepted utterances (`6.2545 hours`, `3,938` balanced cuts / `6.2717 hours` per epoch)** across `Kathbath` (`1,998` clips, `2.94h`), `IndicVoices` (`1,269` clips, `2.36h`), `UrduSpeech` (`339` clips including both `US-Std` and `US-CS`, `0.60h`), and `Rasa` (`199` clips, `0.36h`), with zero train/dev/test leakage (`reports/data_audit_summary.json`).

---

## 3. Bounded Sanity Check Results (Step 5 Verification)
Before launching the full repaired training run, we ran a 200-step sanity check on **16 clean Kathbath Urdu training clips** (`2.0–5.0s`, augmentation disabled):
- **Losses per frame**:
  - Step 1: `simple_loss = 9.6394`, `pruned_loss = 9.5326`, `ctc_loss = 6.5791`
  - Step 50: `simple_loss = 1.3241`, `pruned_loss = 0.1574`, `ctc_loss = 0.9344`
  - Step 100: `simple_loss = 0.7112`, `pruned_loss = 0.0230`, `ctc_loss = 0.2287`
  - Step 200: `simple_loss = 0.4090`, `pruned_loss = 0.0012`, `ctc_loss = 0.0399`
- **Evaluation Accuracy on the 16 Clips (`blank_penalty = 0.0`)**:
  - **Full-chunk causal evaluation**: **`0.57%` WER / `0.40%` CER** (15 of 16 sentences transcribed with 100% word-for-word exact match).
  - **Chunk-by-chunk streaming evaluation (`decode_chunk_len = 32`)**: **`8.62%` WER / `7.67%` CER**.

---

## 4. Repaired Training Run (`exp/urdu_zipformer_20m_v2`)
- **Process & Supervisor**: Trained in `tmux` session `urdu_asr_train_v2`, logged to `logs/train_urdu_zipformer_20m_v2.log`, status tracked in `reports/training_live_status.json`.
- **Epoch-by-Epoch Progression (with Live Streaming Dev Decoding)**:
  - Epoch 1: `train_loss = 4.8499`, `valid_loss = 2.3309` (`pruned = 1.1476`, `ctc = 1.0382`) | Dev subset WER = `99.61%`, CER = `96.69%`
  - Epoch 2: `train_loss = 2.0742`, `valid_loss = 1.6038` (`pruned = 0.8168`, `ctc = 0.6126`) | Dev subset WER = `86.19%`, CER = `82.69%`
  - Epoch 3: `train_loss = 1.3876`, `valid_loss = 1.1016` (`pruned = 0.4885`, `ctc = 0.4629`) | Dev subset WER = `70.43%`, CER = `58.84%`
  - Epoch 4: `train_loss = 1.0151`, `valid_loss = 0.9388` (`pruned = 0.3992`, `ctc = 0.4172`) | Dev subset WER = `60.89%`, CER = `49.57%`
  - Epoch 6: `train_loss = 0.7168`, `valid_loss = 0.7925` (`pruned = 0.3179`, `ctc = 0.3740`) | Dev subset WER = `55.06%`, CER = `36.89%`
  - Epoch 8: `train_loss = 0.5818`, `valid_loss = 0.7677` (`pruned = 0.3092`, `ctc = 0.3642`) | Dev subset WER = `48.44%`, CER = `31.68%`
  - Epoch 10: `train_loss = 0.4907`, `valid_loss = 0.7527` (`pruned = 0.3132`, `ctc = 0.3512`) | Dev subset WER = `46.89%`, CER = `28.12%`
  - Epoch 12: `train_loss = 0.4288`, `valid_loss = 0.7625` (`pruned = 0.3244`, `ctc = 0.3580`) | Dev subset WER = `44.36%`, CER = **`21.78%`**
  - Epoch 14: `train_loss = 0.3861`, `valid_loss = **`0.7491`** (`pruned = 0.3113`, `ctc = 0.3625`) | Dev subset WER = `44.16%`, CER = `25.38%`
  - Epoch 20: `train_loss = 0.3121`, `valid_loss = 0.7646` (`pruned = 0.3252`, `ctc = 0.3761`) | Dev subset WER = `41.63%`, CER = `26.83%`

### Checkpoint Selection on Full Development Data (769 Utterances)
Before running test evaluation, single and averaged checkpoints were compared across all 769 development utterances (`dev_kathbath_ur`: 418, `dev_indicvoices_ur`: 151, `dev_fleurs_ur_pk`: 200):
- `epoch-12-single`: Macro Dev WER = `47.60%`, CER = `26.75%`
- `epoch-14-single`: Macro Dev WER = `47.43%`, CER = `27.94%`
- `epoch-20-single`: Macro Dev WER = `48.82%`, CER = `32.19%`
- **`epoch-14-avg-4` (Epochs 11..14 averaged)**: Macro Dev WER = **`44.86%`**, CER = **`27.25%`** (`dev_kathbath_ur`: `41.99%` WER / `22.19%` CER, `dev_indicvoices_ur`: `40.28%` WER / `26.98%` CER, `dev_fleurs_ur_pk`: `52.30%` WER / `32.58%` CER).
- `epoch-20-avg-5` (Epochs 16..20 averaged): Macro Dev WER = `46.45%`, CER = `30.26%`
- `epoch-20-avg-7` (Epochs 14..20 averaged): Macro Dev WER = `45.62%`, CER = `29.11%`

`epoch-14-avg-4` was selected and frozen as the primary deployment checkpoint (`release/weights/epoch-14-avg-4.pt`).

---

## 5. Native PyTorch vs. ONNX Export Parity & Fresh-Process CPU Benchmarks
- **Per-Clip Native PyTorch vs. ONNX FP32 Parity (25 Dev Clips)**:
  - **Exact String & Token Match**: **`25 / 25` (`100.0%`)** clips produced identical normalized transcripts and token sequences between native PyTorch `streaming_greedy_decode_waveform` (`WER = 42.86%`, `CER = 22.88%`) and `sherpa-onnx` FP32 (`WER = 42.86%`, `CER = 22.88%`).
  - **INT8 Quantization Impact**: `sherpa-onnx` INT8 achieved `WER = 42.52%`, `CER = 23.31%` on the same 25 dev clips (`-0.34%` WER, `+0.43%` CER) and `43.88%` WER / `22.65%` CER on the 80-clip CPU benchmark set (vs `43.76%` WER / `22.73%` CER for FP32).
- **Fresh-Process CPU Memory & Paced Streaming Latency (Intel Core i5-8600K @ 3.60GHz, 80 Test Clips)**:
  - **FP32 (`1` thread)**: Warm RTF = `0.0426` (`23.5x` real-time), mean chunk compute = `11.37 ms` (`p95 = 13.13 ms`), decoder execution time to first partial = `35.23 ms`, **paced audio stream time to first non-empty partial** (including `320 ms` chunk accumulation + `40 ms` lookahead + initial speech onset) = `1,228.43 ms`, finalization delay = `26.95 ms`, fresh-process peak RSS = `1,870.43 MiB`.
  - **INT8 (`1` thread)**: Warm RTF = **`0.0332`** (`30.1x` real-time), mean chunk compute = **`8.86 ms`** (`p95 = 10.10 ms`), decoder execution time to first partial = `27.59 ms`, **paced audio stream time to first non-empty partial** = `1,225.70 ms`, finalization delay = `21.07 ms`, fresh-process peak RSS = `1,870.43 MiB`.
