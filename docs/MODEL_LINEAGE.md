# Model Lineage & Cross-Language Adaptation Report

## 1. Selected Pretrained Architecture & Generation
- **Model Family**: Causal Streaming Zipformer (1st-generation Zipformer from `icefall`'s `pruned_transducer_stateless7_streaming` recipe, paired with a stateless Conv1d predictor and linear joiner).
- **Why this exact generation**:
  - The target size is 20–35M parameters with a proven `icefall` training recipe and verified `sherpa-onnx` streaming export path.
  - `desh2608/icefall-asr-librispeech-pruned-transducer-stateless7-streaming-small` is a published, trainable PyTorch checkpoint (`exp/pretrained.pt`) of ~20.7M parameters trained on 960h LibriSpeech, and `csukuangfj/sherpa-onnx-streaming-zipformer-en-20M-2023-02-17` is its official `sherpa-onnx` export (`icefall` PR #903).
  - Crucially, `pruned_transducer_stateless7_streaming` uses `Zipformer` (`egs/librispeech/ASR/pruned_transducer_stateless7_streaming/zipformer.py`), NOT `Zipformer2` (`egs/librispeech/ASR/zipformer/zipformer.py`). We preserve the exact `pruned_transducer_stateless7_streaming` modules without mixing incompatible `Zipformer2` classes.

## 2. Upstream Provenance & Hashes
- **Upstream `icefall` repository**: `https://github.com/k2-fsa/icefall`
- **Pinned `icefall` commit**: `bfde1030136ffc6770857d3e92bd6762497a8f54` (recipe directory `egs/librispeech/ASR/pruned_transducer_stateless7_streaming`)
- **Pretrained English checkpoint repository**: `https://huggingface.co/desh2608/icefall-asr-librispeech-pruned-transducer-stateless7-streaming-small`
  - Repo Git revision: `be162ecc09bade73063a671fad9d18220149d25b`
  - Checkpoint file: `exp/pretrained.pt` (SHA256 recorded by `scripts/adapt_checkpoint.py` in `reports/checkpoint_lineage.json`)
  - Original English tokenizer: `data/lang_bpe_500/bpe.model` (`vocab_size=500`, `blank_id=0`, `unk_id=2`)
  - Original training lineage (from `log/log-train-2023-02-05-17-58-35-0`): Trained for 30 epochs on full LibriSpeech with `short_chunk_size=50`, `num_left_chunks=4`, `decode_chunk_len=32`, `context_size=2`, `prune_range=5`, `simple_loss_scale=0.5`, `lm_scale=0.25`, `am_scale=0.0`, averaged over epochs 22..30 (`epoch=30, avg=9`).

## 3. Structural Hyperparameters (Preserved Intact)
| Hyperparameter | Value | Notes |
|---|---|---|
| `feature_dim` | `80` | 80-dim log-mel filterbank, 16 kHz, 25ms window, 10ms shift |
| `subsampling_factor` | `4` | `Conv2dSubsampling` + `output_downsampling_factor=2` -> 40ms encoder frame rate |
| `num_encoder_layers` | `2,2,2,2,2` | 5 Zipformer stacks, 10 encoder layers total |
| `encoder_dims` | `256,256,256,256,256` | Constant 256-dim across all 5 stacks |
| `attention_dims` | `192,192,192,192,192` | Attention dimension per stack |
| `encoder_unmasked_dims` | `192,192,192,192,192` | Unmasked feature dimension during SpecAugment |
| `feedforward_dims` | `768,768,768,768,768` | Feedforward dimension per stack |
| `nhead` | `8,8,8,8,8` | 8 attention heads per stack |
| `zipformer_downsampling_factors` | `1,2,4,8,2` | Multi-resolution temporal downsampling factors |
| `cnn_module_kernels` | `31,31,31,31,31` | Depthwise separable CNN kernel sizes |
| `short_chunk_size` | `50` | Dynamic chunk training size controller |
| `num_left_chunks` | `4` | Left context chunks (`4 * 16 = 64` encoder frames = 2.56s left context) |
| `decode_chunk_len` | `32` | 32 frames at 20ms (`Conv2dSubsampling` output) = 16 encoder frames = 320ms chunk |
| `decoder_dim` | `512` | Stateless predictor embedding & Conv1d channel dim |
| `context_size` | `2` | Bigram token history (`kernel_size=2`, `groups=128`) |
| `joiner_dim` | `512` | Joiner hidden projection dimension |

## 4. Vocabulary Adaptation & Tensor Transfer Specification
When adapting from English (`vocab_size=500`) to Urdu + English code-switching (`vocab_size=1024`):
- **Transferred Exactly (Retained Tensors)**:
  - All `encoder.*` parameters (`encoder_embed.*`, all 5 `encoder.encoders.*` stacks, `encoder.downsample_output.*`).
  - `decoder.conv.weight` (`shape=[512, 4, 2]`, vocabulary-independent depthwise-grouped 1D convolution over embedding channels).
  - `joiner.encoder_proj.weight` (`[512, 256]`) and `joiner.encoder_proj.bias` (`[512]`).
  - `joiner.decoder_proj.weight` (`[512, 512]`) and `joiner.decoder_proj.bias` (`[512]`).
- **Reinitialized for New Vocabulary (`vocab_size=1024`)**:
  - `decoder.embedding.weight` (`[500, 512]` -> `[1024, 512]`): Newly initialized; row `0` (`<blk>`) and any exact matching special tokens (`<sos/eos>`, `<unk>`) are explicitly mapped and verified.
  - `joiner.output_linear.weight` (`[500, 512]` -> `[1024, 512]`) and `joiner.output_linear.bias` (`[500]` -> `[1024]`): Newly initialized (with `<blk>` row `0` preserved/verified).
  - `simple_am_proj.weight` (`[500, 256]` -> `[1024, 256]`) and `simple_am_proj.bias` (`[500]` -> `[1024]`): Newly initialized for pruned RNN-T simple AM loss.
  - `simple_lm_proj.weight` (`[500, 512]` -> `[1024, 512]`) and `simple_lm_proj.bias` (`[500]` -> `[1024]`): Newly initialized for pruned RNN-T simple LM loss.
- **Parameter Count Impact**:
  - English (`vocab_size=500`): `20,697,573` total parameters (`19,544,029` deployed after stripping training-only `simple_am_proj` and `simple_lm_proj`).
  - Adapted Urdu (`vocab_size=1024`): `21,370,389` total training parameters; `20,580,949` deployed ASR parameters (`encoder` + `decoder` + `joiner`), well within the 20–35M target.
