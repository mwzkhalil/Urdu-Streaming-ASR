# Implementation & Training Plan: Urdu-Streaming-ASR

## 1. Objective & Scope
Build, verify, train, and export a compact, causal, truly streaming Urdu (with natural Urdu-English code-switching) transducer ASR model using the `icefall` / `k2` / `Lhotse` ecosystem and export it to `sherpa-onnx` (FP32 and INT8).
- **Target deployment footprint**: ~21.23M parameters (within the 20–35M target).
- **Streaming mechanism**: Causal chunk-wise Zipformer encoder (`short_chunk_size=50`, `num_left_chunks=4`, `decode_chunk_len=32` frames at 10ms frame shift -> 320ms chunk with 40ms subsampling lookahead), stateless predictor (`context_size=2`), and linear joiner. Produces incremental partial hypotheses while audio frames arrive.

## 2. Architecture & Pretrained Checkpoint Selection
- **Upstream recipe**: `icefall` (`egs/librispeech/ASR/pruned_transducer_stateless7_streaming`), pinned at commit `bfde1030136ffc6770857d3e92bd6762497a8f54`.
- **Pretrained source checkpoint**: `desh2608/icefall-asr-librispeech-pruned-transducer-stateless7-streaming-small` (`exp/pretrained.pt`, repo revision `be162ecc09bade73063a671fad9d18220149d25b`).
  - Original English BPE-500 parameter count: `20,697,573`.
  - Encoder config: `num_encoder_layers=2,2,2,2,2`, `encoder_dims=256,256,256,256,256`, `attention_dims=192,192,192,192,192`, `encoder_unmasked_dims=192,192,192,192,192`, `feedforward_dims=768,768,768,768,768`, `nhead=8,8,8,8,8`, `zipformer_downsampling_factors=1,2,4,8,2`, `cnn_module_kernels=31,31,31,31,31`.
  - Decoder/Joiner config: `decoder_dim=512`, `joiner_dim=512`, `context_size=2`.
- **Cross-language adaptation strategy**:
  1. Verify exact loading and streaming English reference decoding on the 3 test wavs shipped with `desh2608/icefall-asr-librispeech-pruned-transducer-stateless7-streaming-small`.
  2. Train a source-balanced SentencePiece BPE tokenizer (`vocab_size=1024`, `blank_id=0`, `unk_id=2`) strictly on normalized training transcripts, preserving Urdu script distinctions (`ھ`, `ہ`, `ں`, `ڑ`, `ٹ`, `ڈ`, `ء`, `ئ`, `ؤ`, `ے`, `ی`) plus Latin characters for English code-switching and digits `0-9`.
  3. Transfer 100% of `encoder.*` parameters plus vocabulary-independent `decoder.conv.weight`, `joiner.encoder_proj.*`, and `joiner.decoder_proj.*`.
  4. Reinitialize vocabulary-dependent tensors (`decoder.embedding.weight`, `joiner.output_linear.*`, `simple_am_proj.*`, `simple_lm_proj.*`), explicitly copying row `0` (`<blk>`) where appropriate or initializing cleanly with verified blank/token ID consistency.
  5. Use differential learning rates during fine-tuning: lower encoder learning rate (`encoder_lr_scale=0.2` relative to `base_lr`) and full `base_lr` on newly initialized predictor/joiner/projection heads, with `batch_count` offset (`+100000`) as in upstream `icefall` `finetune.py` so pretrained encoder `ScheduledFloat` and layer-drop modules stay in post-warmup state.

## 3. Data Strategy, Quality Control & Leakage Prevention
- **Storage budget**: Remote host `/` has 149 GiB free out of 468 GiB. Maintain a hard floor of >= 40 GiB free at all times (targeting <= 45 GiB total new disk usage across environment, checkpoints, and bounded parquet/FLAC data).
- **Accessible Training Sources**:
  1. **Kathbath Urdu** (`ai4bharat/Kathbath`, human-labeled read speech, `urdu/train` & `urdu/valid`).
  2. **IndicVoices Urdu** (`ai4bharat/IndicVoices`, human-labeled spontaneous/extempore/conversational/read speech across diverse speakers and scenarios, `urdu/train` & `urdu/valid`).
  3. **ASLP UrduSpeech** (`ASLP-lab/UrduSpeech`, natural Pakistani Urdu `US-Std` and Urdu-English code-switched `US-CS`, plus capped `US-EngPk`; treated as SILVER labels; strict exclusion of all `US-benchmark` recordings and any overlapping source IDs).
  4. **Rasa Urdu** (`ai4bharat/Rasa`, studio expressive speech; capped at <= 10% sampling weight).
- **Blocked/Pending Source**:
  - **Common Voice Scripted Speech 27.0 - Urdu** (`cmu5x1w3400cyo107st8cgsoq` on Mozilla Data Collective): Adapter is fully implemented with SHA256 verification (`12ad5f359fd94805db1bdaeb1441eb35720221f3c99229034069375fa246d990`), but `POST /api/datasets/cmu5x1w3400cyo107st8cgsoq/download` returns HTTP 401 (`{"error":"Authentication required"}`) without a user-supplied `MDC_API_KEY`. Recorded as gated/pending; pipeline proceeds with accessible human + silver sources.
- **Evaluation-Only Benchmarks (Sealed from Training & Tokenizer)**:
  - `Kathbath` Urdu official `valid` (partitioned deterministically by speaker into `dev` and `test` since public parquet release provides `train` and `valid`).
  - `IndicVoices` Urdu official `valid` (partitioned deterministically by speaker/session into `dev` and `test`).
  - `ASLP UrduSpeech` `US-benchmark` (`US-benchmark-Std`, `US-benchmark-CS`, `US-benchmark-EngPk` manually corrected benchmark).
  - `google/fleurs` (`ur_pk` `dev` and `test`, external domain benchmark).
  - `urdu-asr/csalt-voice` (conversational benchmark from `sameearif/WER-We-Stand`).

## 4. Verification & Execution Sequence
1. Establish isolated conda environment (`urdu_asr_stream`) with Python 3.11, PyTorch 2.11.0+cu128, torchaudio 2.11.0+cu128, prebuilt `k2` CUDA 12.8 wheel, `lhotse`, `sentencepiece`, `onnx`, `onnxruntime`, and `sherpa-onnx`.
2. Verify CUDA RNN-T pruned loss forward/backward pass.
3. Run incremental data acquisition, audio/text validation, deduplication, leakage sealing, stratified 40-clip quality audit, and SentencePiece training.
4. Run smoke checks: unit tests, English checkpoint reference decode, Urdu vocab adaptation report, 24-clip overfit loss reduction check, checkpoint save/resume verification, and stateful streaming decode check.
5. Launch persistent `tmux` training run on RTX 3060 (`12 GB VRAM`), verify step advancement and checkpoints, evaluate on dev/test sets, export FP32 & INT8 ONNX models for `sherpa-onnx`, benchmark CPU RTF/latency (1 and 4 threads), and assemble the local HF release folder.
