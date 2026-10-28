# Evaluation Protocol

## 1. Development vs. Final Test Separation
- All checkpoint selection, epoch averaging window (`epoch`, `avg`), decoding method (`greedy_search` vs `modified_beam_search`), and normalizer rules are selected strictly on **development** splits (`kathbath_dev`, `indicvoices_dev`).
- Final held-out **test** sets (`kathbath_test`, `indicvoices_test`, `urduspeech_benchmark_std`, `urduspeech_benchmark_cs`, `fleurs_ur_pk_test`, `csalt_voice_val`) are sealed from tokenizer training, vocabulary construction, and hyperparameter selection.

## 2. Metrics & Reporting
- **Primary Metric**: Macro-averaged (source-balanced) Normalized WER and CER across held-out domains, reported alongside per-source Raw and Normalized WER, CER, substitutions, deletions, insertions, utterance counts, and total audio hours.
- **Code-Switching Breakdown**: Standard Urdu vs. Urdu-English code-switched utterances are scored both in aggregate and separately without translating or stripping Latin words.
- **Streaming vs. Full-Context Parity**: Evaluated using the causal streaming Zipformer chunk configuration (`decode_chunk_len=32`, `num_left_chunks=4`).
- **Deployment Benchmarks**:
  - FP32 vs. INT8 ONNX accuracy on held-out sets.
  - CPU Real-Time Factor (RTF), peak RSS memory, time-to-first-partial hypothesis, finalization delay, and algorithmic lookahead (`320ms` chunk + `40ms` subsampling lookahead) measured at `num_threads=1` and `num_threads=4`.
- **External Baseline Comparison**: Evaluated against `openai/whisper-tiny` (or `whisper-base`) on the identical evaluation manifests and normalizer, explicitly noting that Whisper is an offline chunked/windowed encoder-decoder rather than a causal frame-synchronous streaming transducer.
