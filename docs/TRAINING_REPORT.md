# Training, Adaptation, Export, and Evaluation Report

## 1. Pretrained Checkpoint Lineage & Scale-Matched Vocabulary Adaptation
- **Base Pretrained Checkpoint**: `desh2608/icefall-asr-librispeech-pruned-transducer-stateless7-streaming-small` (`exp/pretrained.pt`, revision `be162ecc09bade73063a671fad9d18220149d25b`, SHA256 `3222015e7c994de8bc9fd7f03414d6e34e9f81345380d6ee247f55457ac1ff67`).
- **Tensor Transfer & Scale-Matched Reinitialization (`reports/checkpoint_lineage.json`)**:
  - **Retained (`460` tensors)**: `100%` of causal Zipformer `encoder.*` parameters plus vocabulary-independent `decoder.conv.weight`, `joiner.encoder_proj.*`, and `joiner.decoder_proj.*`.
  - **Reinitialized (`7` vocabulary-sized tensors)**: `decoder.embedding.weight` (`std=2.984`, row `0` zeroed), `joiner.output_linear.*` (`std=0.208`), `simple_am_proj.*` (`std=2.224`), and `simple_lm_proj.*` (`std=0.09`, calibrated so initial LM logits do not overwhelm AM logits during `k2` pruning).
  - **Deployed Parameter Count**: **`20,849,663`** (`encoder`: `19,533,311`, `decoder`: `791,040`, `joiner`: `525,312`).

## 2. Repaired Training Run (`exp/urdu_zipformer_20m_v2`)
- **Dataset Mixture (`6.25h` unique / `3,907` utterances; `6.27h` balanced / `3,938` cuts per epoch)**:
  - `kathbath_ur:train`: `1,998` utterances (`2.94h`)
  - `indicvoices_ur:train`: `1,269` utterances (`2.36h`)
  - `urduspeech_aslp:train`: `339` utterances (`0.60h` across `US-Std` and `US-CS`)
  - `rasa_ur:train`: `199` utterances (`0.36h`)
- **Key Training Repairs**:
  1. Auxiliary CTC loss (`ctc_loss_scale = 0.50`) on `simple_am_proj(encoder_out)` + `am_scale = 0.45, lm_scale = 0.05` to anchor `k2.get_rnnt_prune_ranges` along the true acoustic diagonal.
  2. `encoder.short_chunk_threshold = 2.0` so 100% of training forward passes use causal chunk-wise attention masks matching streaming inference.
  3. Three-group `ScaledAdam` learning rates (`encoder = 0.008`, `am_joiner = 0.025`, `decoder_lm = 0.004`).
- **Convergence**:
  - Trained for `20` epochs (`3,529` optimizer updates) in `tmux` session `urdu_asr_train_v2`.
  - Train loss decreased from `4.8499` (Epoch 1) to `0.3121` (Epoch 20); validation loss reached `0.7491` at Epoch 14 (`valid_pruned_loss = 0.3113`, `valid_ctc_loss = 0.3625`), and development streaming CER improved from `96.69%` (Epoch 1) to `21.78%` (Epoch 12).
  - Peak GPU VRAM allocated: `2,771.21 MiB`; free disk space maintained at `131 GiB`.

## 3. Full Held-Out Evaluation (`epoch-14-avg-4`, `blank_bias_adjustment = 0.0`)
*(Disclosure: Full official development and test partitions are scored below; the first 50 utterances per test partition were previously inspected during the v1 failure diagnosis.)*

| Partition | Utterances (Hours) | Before Repair (`v1`) Norm WER / CER | Repaired (`v2` `epoch-14-avg-4`) Raw WER / CER | Repaired (`v2` `epoch-14-avg-4`) Norm WER / CER | S / D / I (Normalized) | Offline `whisper-tiny` (37.8M, 50-cut ref) Norm WER / CER |
|---|---:|---:|---:|---:|---:|---:|
| `dev_kathbath_ur` | 418 (`0.613h`) | `97.38%` / `87.80%` | `41.99%` / `22.19%` | **`41.99%` / `22.19%`** | `1495 / 474 / 125` | — |
| `dev_indicvoices_ur` | 151 (`0.337h`) | `98.10%` / `85.95%` | `44.07%` / `27.85%` | **`40.28%` / `26.98%`** | `652 / 499 / 36` | — |
| `dev_fleurs_ur_pk` | 200 (`0.569h`) | `96.19%` / `84.51%` | `56.60%` / `33.90%` | **`52.30%` / `32.58%`** | `1700 / 875 / 61` | — |
| **Macro Dev Average** | **769 (`1.518h`)** | `97.22%` / `86.09%` | `47.55%` / `27.98%` | **`44.86%` / `27.25%`** | — | — |
| **Pooled Dev Total** | **769 (`1.518h`)** | — | `47.87%` / `28.16%` | **`45.72%` / `27.48%`** | `3847 / 1848 / 222` | — |
| `test_kathbath_ur` | 182 (`0.243h`) | `102.07%` / `90.56%` | `46.67%` / `24.69%` | **`46.67%` / `24.69%`** | `612 / 224 / 68` | `89.47%` / `41.54%` |
| `test_indicvoices_ur` | 245 (`0.443h`) | `104.83%` / `92.79%` | `50.04%` / `34.42%` | **`47.64%` / `33.66%`** | `1058 / 1040 / 41` | `69.59%` / `38.20%` |
| `test_fleurs_ur_pk` | 200 (`0.520h`) | `96.99%` / `84.85%` | `57.88%` / `33.18%` | **`53.28%` / `31.90%`** | `1862 / 921 / 107` | `75.09%` / `38.91%` |
| `test_rasa_ur` | 100 (`0.180h`) | `97.00%` / `86.10%` | `41.81%` / `22.31%` | **`36.86%` / `20.75%`** | `388 / 188 / 33` | `59.38%` / `24.21%` |
| `test_csalt_voice` | 159 (`0.407h`) | `104.26%` / `93.49%` | `59.40%` / `43.74%` | **`59.40%` / `43.74%`** | `1238 / 1518 / 54` | `90.43%` / `57.54%` |
| `test_urduspeech_benchmark_cs` | 52 (`0.196h`) | `100.64%` / `92.34%` | `70.88%` / `54.71%` | **`67.49%` / `53.44%`** | `666 / 849 / 11` | `85.12%` / `58.08%` |
| **Macro Test Average** | **938 (`1.988h`)** | `100.96%` / `90.02%` | `54.45%` / `35.51%` | **`51.89%` / `34.70%`** | — | `78.18%` / `43.08%` |
| **Pooled Test Total** | **938 (`1.988h`)** | — | `55.58%` / `36.36%` | **`53.08%` / `35.56%`** | `5824 / 4740 / 314` | — |
