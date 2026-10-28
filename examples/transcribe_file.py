#!/usr/bin/env python3
"""Minimal offline-file streaming transcription example using sherpa-onnx.

Downloads (or loads locally) the exported ONNX transducer files from
`mahwizzzz/Urdu-Streaming-Zipformer-20M` and transcribes a 16 kHz WAV/FLAC file.
Supports both Quality mode (default: modified_beam_search, max_active_paths=4, blank_penalty=0.25)
and Fast mode (greedy_search, blank_penalty=0.0).
"""

import argparse
from pathlib import Path

import numpy as np
import sherpa_onnx
import soundfile as sf

DECODING_MODES = {
    "quality": {"decoding_method": "modified_beam_search", "max_active_paths": 4, "blank_penalty": 0.25},
    "fast": {"decoding_method": "greedy_search", "max_active_paths": 4, "blank_penalty": 0.0},
}


def resolve_model_paths(repo_or_dir: str, use_int8: bool = True, revision: str | None = None):
    local_p = Path(repo_or_dir)
    suffix = ".int8.onnx" if use_int8 else ".onnx"
    if local_p.exists() and local_p.is_dir():
        onnx_dir = local_p / "onnx" if (local_p / "onnx").exists() else local_p
        tok_path = (
            local_p / "tokenizer" / "tokens.txt"
            if (local_p / "tokenizer" / "tokens.txt").exists()
            else onnx_dir / "tokens.txt"
        )
        return (
            str(onnx_dir / f"encoder{suffix}"),
            str(onnx_dir / f"decoder{suffix}"),
            str(onnx_dir / f"joiner{suffix}"),
            str(tok_path),
        )

    from huggingface_hub import hf_hub_download

    enc = hf_hub_download(repo_id=repo_or_dir, filename=f"onnx/encoder{suffix}", revision=revision)
    dec = hf_hub_download(repo_id=repo_or_dir, filename=f"onnx/decoder{suffix}", revision=revision)
    joiner = hf_hub_download(repo_id=repo_or_dir, filename=f"onnx/joiner{suffix}", revision=revision)
    tokens = hf_hub_download(repo_id=repo_or_dir, filename="tokenizer/tokens.txt", revision=revision)
    return enc, dec, joiner, tokens


def transcribe_file(
    audio_path: str,
    repo_or_dir: str = "mahwizzzz/Urdu-Streaming-Zipformer-20M",
    use_int8: bool = True,
    num_threads: int = 1,
    mode: str = "quality",
    revision: str | None = None,
) -> str:
    mode_cfg = DECODING_MODES.get(mode.lower(), DECODING_MODES["quality"])
    enc_p, dec_p, join_p, tok_p = resolve_model_paths(repo_or_dir, use_int8=use_int8, revision=revision)
    recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
        tokens=tok_p,
        encoder=enc_p,
        decoder=dec_p,
        joiner=join_p,
        num_threads=num_threads,
        sample_rate=16000,
        feature_dim=80,
        decoding_method=mode_cfg["decoding_method"],
        max_active_paths=mode_cfg["max_active_paths"],
        blank_penalty=mode_cfg["blank_penalty"],
    )

    samples, sr = sf.read(audio_path, dtype="float32", always_2d=True)
    mono = samples.mean(axis=1, dtype=np.float32) if samples.shape[1] > 1 else samples[:, 0]
    if sr != 16000:
        duration = len(mono) / float(sr)
        target_len = max(1, int(round(duration * 16000)))
        x_old = np.linspace(0.0, 1.0, num=len(mono), endpoint=False, dtype=np.float64)
        x_new = np.linspace(0.0, 1.0, num=target_len, endpoint=False, dtype=np.float64)
        mono = np.interp(x_new, x_old, mono).astype(np.float32)

    stream = recognizer.create_stream()
    stream.accept_waveform(16000, mono)
    stream.accept_waveform(16000, np.zeros(int(0.66 * 16000), dtype=np.float32))
    stream.input_finished()
    while recognizer.is_ready(stream):
        recognizer.decode_stream(stream)
    return recognizer.get_result(stream).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Transcribe an Urdu audio file with sherpa-onnx")
    parser.add_argument("audio", type=str, help="Path to 16 kHz mono WAV/FLAC file")
    parser.add_argument(
        "--model",
        type=str,
        default="mahwizzzz/Urdu-Streaming-Zipformer-20M",
        help="Local repository directory or Hugging Face repo_id",
    )
    parser.add_argument("--revision", type=str, default=None, help="Optional HF revision or tag")
    parser.add_argument("--mode", choices=["quality", "fast"], default="quality", help="Decoding mode: quality (beam-4, bp=0.25) or fast (greedy, bp=0.0)")
    parser.add_argument("--fp32", action="store_true", help="Use FP32 ONNX weights instead of INT8")
    parser.add_argument("--threads", type=int, default=1, help="Number of CPU threads")
    args = parser.parse_args()

    text = transcribe_file(
        args.audio,
        repo_or_dir=args.model,
        use_int8=not args.fp32,
        num_threads=args.threads,
        mode=args.mode,
        revision=args.revision,
    )
    print(f"Transcript ({'FP32' if args.fp32 else 'INT8'}, mode={args.mode}): {text}")


if __name__ == "__main__":
    main()
