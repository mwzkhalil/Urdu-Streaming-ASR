# Hardware & Software Environment Specification

## 1. Remote GPU Host Preflight Summary (Redacted)
- **Hostname / OS**: `proxima` (Ubuntu 24.04 LTS, Linux x86_64)
- **CPU**: Intel(R) Core(TM) i5-8600K CPU @ 3.60GHz (6 cores, 6 threads)
- **System RAM**: 31 GiB total (~23 GiB available at preflight)
- **GPU**: 1x NVIDIA GeForce RTX 3060 (12,288 MiB VRAM), Driver `580.178.04`, idle at preflight (0 active GPU processes)
- **System CUDA Toolkit**: `/usr/bin/nvcc` (`12.8.93`)
- **System Audio Tools**: `ffmpeg` (`7:6.1.1-3ubuntu5`), `libsndfile1` (`1.2.2-1ubuntu5.24.04.1`), `libsoxr0` (`0.1.3-4build3`)
- **Process Supervisor**: `/usr/bin/tmux`
- **Storage (`/dev/nvme0n1p2` mounted on `/`)**: 468 GiB total, 295 GiB used, **149 GiB available** at preflight. Hard free-space floor enforced: **40 GiB**.
- **Hugging Face Authentication**: Existing authenticated token present on host (`~/.cache/huggingface/token`, verified HTTP 200 via `api/whoami-v2`, token never printed or copied).

## 2. Isolated Conda Environment (`urdu_asr_stream`)
To avoid modifying `base` or any other project's environment (`invoicepilot`, `junaid`, `mahwiz`, `tts`), this project creates and uses a dedicated environment:
- **Environment Name**: `urdu_asr_stream` (`/home/proxima/miniconda3/envs/urdu_asr_stream`)
- **Python**: `3.11.15`
- **PyTorch / Torchaudio**: `torch==2.11.0+cu130`, `torchaudio==2.11.0+cu130`
- **k2**: `k2==1.24.4.dev20260625+cuda13.0.torch2.11.0` (prebuilt wheel from `https://k2-fsa.github.io/k2/cuda.html` matching CUDA 13.0 and PyTorch 2.11.0 on Python 3.11)
- **Lhotse**: `lhotse` (installed via pip)
- **Icefall**: Cloned at pinned commit `bfde1030136ffc6770857d3e92bd6762497a8f54` in `~/PROXIMA/MAHWIZ/Urdu-Streaming-ASR/third_party/icefall` and installed in editable/PYTHONPATH mode
- **Tokenizer & ONNX / Deployment**: `sentencepiece`, `onnx`, `onnxruntime`, `sherpa-onnx`, `soundfile`, `pyarrow`, `jiwer`, `kaldi-native-fbank`

## 3. Mixed Precision & Memory Policy
- **AMP Mode**: FP16 (`--use-fp16 True`) with PyTorch `GradScaler` and `icefall` `ScaledAdam` + `Eden` scheduler.
- **Batching**: `DynamicBucketingSampler` (`num_buckets=20`, conservative `max_duration` calibrated via actual CUDA forward/backward peak memory measurement on the 12 GiB RTX 3060).
