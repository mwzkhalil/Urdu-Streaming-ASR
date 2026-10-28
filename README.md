# Urdu Streaming ASR

### Compact Urdu Speech Recognition with Zipformer

An open-source Urdu automatic speech recognition project
built around a streaming Zipformer transducer.

The repository provides training code, tokenizer assets,
model configurations, evaluation evidence, and both
file-based and streaming inference examples.

| Specification | Details |
|---|---|
| Language | Urdu |
| Architecture | Zipformer Transducer |
| Audio input | 16 kHz mono |
| Inference | ONNX FP32 / INT8 |
| Decoding | Greedy search and modified beam search |

## Model Weights

The trained model binaries are hosted on Hugging Face.

**[Download model weights](https://huggingface.co/mahwizzzz/Urdu-Streaming-Zipformer-20M/tree/v1.2.0-rc1)**

Verified release: `v1.2.0-rc1`

Available artifacts:

| Component | Format |
|:--|:--|
| Selected checkpoint | PyTorch `.pt` |
| Encoder, decoder, joiner | ONNX FP32 |
| Quantized encoder, decoder, joiner | ONNX INT8 |

The source repository maintains the tokenizer and
inference implementation separately from model weights.

## Installation

Python and compatible PyTorch, sherpa-onnx, and related
dependencies are required.

```bash
git clone https://github.com/mwzkhalil/Urdu-Streaming-ASR.git
cd Urdu-Streaming-ASR

python -m venv .venv
source .venv/bin/activate

python -m pip install -r requirements/inference.txt
python -m pip install -e .
python -m pip install huggingface_hub
```


## Download the Model

```bash
hf download mahwizzzz/Urdu-Streaming-Zipformer-20M \
  --revision v1.2.0-rc1 \
  --include "onnx/*" \
  --local-dir models

cp -a tokenizer models/tokenizer
```

The PyTorch checkpoint can be obtained separately using
`--include "weights/*"` with the same pinned revision.

## File Transcription

INT8 quality decoding:

```bash
python examples/transcribe_file.py \
  examples/sample_urdu.wav \
  --model models \
  --mode quality
```

FP32 decoding:

```bash
python examples/transcribe_file.py \
  examples/sample_urdu.wav \
  --model models \
  --mode quality \
  --fp32
```

Fast greedy decoding:

```bash
python examples/transcribe_file.py \
  examples/sample_urdu.wav \
  --model models \
  --mode fast
```

## Streaming Recognition

Chunk-based streaming from an audio file:

```bash
python examples/stream_paced_or_mic.py \
  --audio examples/sample_urdu.wav \
  --model models \
  --mode quality
```

To simulate real-time chunk arrival, add `--real-time`.

Microphone input is supported through the
`--microphone` CLI option, subject to an available audio
device and supported runtime.

## Evaluation Results

The following results are for the selected released
model using INT8 ONNX, modified beam search with four
active paths, blank penalty 0.25, and normalized
WER/CER scoring.

| Benchmark | WER ↓ | CER ↓ |
|:--|--:|--:|
| CSALT Voice | 27.82% | 18.33% |
| FLEURS Urdu — test subset | 27.95% | 12.10% |
| IndicVoices Urdu | 17.42% | 9.51% |
| KathBath Urdu | 21.48% | 6.67% |
| RASA Urdu | 12.83% | 4.84% |
| UrduSpeech — code-switch | 33.48% | 21.80% |
| FLEURS Urdu — 299 utterances | 28.03% | 11.96% |

Lower is better.

The two FLEURS rows describe different evaluation
subsets and must not be treated as independent
combined benchmarks.

The authoritative evidence is retained in
[reports/evaluation.json](reports/evaluation.json).
Historical experiment identifiers are preserved
there for auditability.

## Streaming Performance

A recorded single-sample streaming run produced:

| Measurement | Observed value |
|:--|--:|
| Audio duration | 4.74 s |
| Processed chunks | 15 |
| Mean chunk compute time | 10.66 ms |
| Total compute real-time factor | 0.03837 |

These figures describe one sample run. They are
not population-level latency guarantees or full
live-microphone end-to-end measurements.

## Training

Training code is provided in `training/`,
`scripts/`, `configs/`, and the bundled Icefall recipe.

Install the compatible training dependencies using
`requirements/train.txt`, following the project's
environment documentation.

Inspect the available training options:

```bash
python training/train.py --help
```

The training entry point has been import-tested.
Complete training-from-scratch reproduction has
not yet been independently validated from this
cleaned source distribution.

Training requires appropriately prepared dataset
manifests and the compatible Icefall/k2 environment.

## Project Structure

| Directory | Purpose |
|:--|:--|
| `src/urdu_streaming_asr/` | Reusable application code |
| `training/` | Training entry point |
| `scripts/` | Preparation, training and evaluation |
| `configs/` | Model and experiment configuration |
| `tokenizer/` | BPE and token metadata |
| `third_party/icefall_recipe/` | Required upstream model components |
| `examples/` | File and streaming inference |
| `tests/` | Automated tests |
| `reports/` | Authoritative evaluation evidence |
| `docs/` | Technical methodology and provenance |

## Evaluation and Reproducibility

The project documents corpus provenance, data
partitioning, normalization, decoding settings,
checkpoint selection, and evaluation evidence.

Comparisons must use matched evaluation conditions.
Historical baseline discrepancies are retained
in the audit record rather than silently replaced.

See `docs/` and `reports/evaluation.json`.

## Limitations

Recognition performance varies with acoustic
conditions, speech style, dialect, recording
quality, and code-switching.

The available benchmark results should not be
generalized to unsupported languages or domains.

## Acknowledgements

The training architecture builds on components
from the open-source Icefall project. Applicable
third-party copyright and license notices are
retained alongside bundled source files.

## License

See [LICENSE](LICENSE) and the upstream license
included in `third_party/icefall_recipe/`.

## Citation

Please cite the associated project and upstream
architecture according to the available research
and license documentation.
