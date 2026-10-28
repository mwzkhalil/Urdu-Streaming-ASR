#!/usr/bin/env python3
"""Genuine chunk-paced (320 ms hop) streaming and live microphone transcription example.

Supports:
1. Chunk-by-chunk streaming simulation over an audio file (`--audio path/to/file.wav`),
   optionally paced at real-time wall-clock speed (`--real-time`).
2. Live microphone streaming (`--microphone`) if `sounddevice` is installed.
3. Quality mode (default: modified_beam_search, max_active_paths=4, blank_penalty=0.25)
   and Fast mode (greedy_search, blank_penalty=0.0).
"""

import argparse
import sys
import time
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


def build_recognizer(
    repo_or_dir: str,
    use_int8: bool = True,
    num_threads: int = 1,
    mode: str = "quality",
    revision: str | None = None,
) -> sherpa_onnx.OnlineRecognizer:
    mode_cfg = DECODING_MODES.get(mode.lower(), DECODING_MODES["quality"])
    enc_p, dec_p, join_p, tok_p = resolve_model_paths(repo_or_dir, use_int8=use_int8, revision=revision)
    return sherpa_onnx.OnlineRecognizer.from_transducer(
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
        enable_endpoint_detection=True,
        rule1_min_trailing_silence=2.4,
        rule2_min_trailing_silence=1.2,
        rule3_min_utterance_length=20.0,
    )


def stream_file_chunks(
    recognizer: sherpa_onnx.OnlineRecognizer,
    audio_path: str,
    chunk_ms: int = 320,
    real_time_paced: bool = False,
) -> None:
    samples, sr = sf.read(audio_path, dtype="float32", always_2d=True)
    mono = samples.mean(axis=1, dtype=np.float32) if samples.shape[1] > 1 else samples[:, 0]
    if sr != 16000:
        duration = len(mono) / float(sr)
        target_len = max(1, int(round(duration * 16000)))
        x_old = np.linspace(0.0, 1.0, num=len(mono), endpoint=False, dtype=np.float64)
        x_new = np.linspace(0.0, 1.0, num=target_len, endpoint=False, dtype=np.float64)
        mono = np.interp(x_new, x_old, mono).astype(np.float32)
        sr = 16000

    chunk_samples = int((chunk_ms / 1000.0) * sr)
    num_chunks = (len(mono) + chunk_samples - 1) // chunk_samples
    stream = recognizer.create_stream()

    last_text = ""
    chunk_compute_ms = []
    for idx in range(num_chunks):
        chunk = mono[idx * chunk_samples : (idx + 1) * chunk_samples]
        if real_time_paced:
            time.sleep(len(chunk) / float(sr))
        t0 = time.perf_counter()
        stream.accept_waveform(sr, chunk)
        while recognizer.is_ready(stream):
            recognizer.decode_stream(stream)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        chunk_compute_ms.append(dt_ms)
        current = recognizer.get_result(stream).strip()
        audio_sec = min(len(mono), (idx + 1) * chunk_samples) / float(sr)
        if current != last_text:
            print(f"[PARTIAL t={audio_sec:.2f}s | chunk_compute={dt_ms:.2f}ms] {current}")
            last_text = current

    t_flush = time.perf_counter()
    stream.accept_waveform(sr, np.zeros(int(0.66 * sr), dtype=np.float32))
    stream.input_finished()
    while recognizer.is_ready(stream):
        recognizer.decode_stream(stream)
    flush_ms = (time.perf_counter() - t_flush) * 1000.0
    final_text = recognizer.get_result(stream).strip()
    audio_dur = len(mono) / float(sr)
    print(f"[FINAL   t={audio_dur:.2f}s | finalize_compute={flush_ms:.2f}ms] {final_text}")
    print(
        f"Summary: audio_duration={audio_dur:.2f}s, chunks={num_chunks}, "
        f"mean_chunk_compute={np.mean(chunk_compute_ms):.2f}ms, "
        f"total_compute_rtf={(sum(chunk_compute_ms) + flush_ms) / (audio_dur * 1000.0):.5f}"
    )


def stream_microphone(recognizer: sherpa_onnx.OnlineRecognizer, chunk_ms: int = 320) -> None:
    try:
        import sounddevice as sd
    except ImportError:
        print("Install `sounddevice` (`pip install sounddevice`) for live microphone capture.", file=sys.stderr)
        sys.exit(1)

    sr = 16000
    chunk_samples = int((chunk_ms / 1000.0) * sr)
    stream = recognizer.create_stream()
    last_text = ""
    utt_idx = 1
    print("Listening on default microphone (16 kHz mono, 320 ms chunks). Press Ctrl+C to stop...")
    with sd.InputStream(samplerate=sr, channels=1, dtype="float32", blocksize=chunk_samples) as mic:
        try:
            while True:
                data, _ = mic.read(chunk_samples)
                stream.accept_waveform(sr, data[:, 0])
                while recognizer.is_ready(stream):
                    recognizer.decode_stream(stream)
                current = recognizer.get_result(stream).strip()
                if current and current != last_text:
                    print(f"\r[Utterance #{utt_idx} PARTIAL] {current}", end="", flush=True)
                    last_text = current
                if recognizer.is_endpoint(stream):
                    if current:
                        print(f"\r[Utterance #{utt_idx} FINAL  ] {current}")
                        utt_idx += 1
                    recognizer.reset(stream)
                    last_text = ""
        except KeyboardInterrupt:
            print("\nStopped microphone streaming.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Chunk-paced streaming or microphone Urdu ASR with sherpa-onnx")
    parser.add_argument("--audio", type=str, help="Path to 16 kHz mono WAV/FLAC file for chunk-stream simulation")
    parser.add_argument("--microphone", action="store_true", help="Stream live audio from the default microphone")
    parser.add_argument(
        "--model",
        type=str,
        default="mahwizzzz/Urdu-Streaming-Zipformer-20M",
        help="Local repository directory or Hugging Face repo_id",
    )
    parser.add_argument("--revision", type=str, default=None, help="Optional HF revision or tag")
    parser.add_argument("--mode", choices=["quality", "fast"], default="quality", help="Decoding mode: quality (beam-4, bp=0.25) or fast (greedy, bp=0.0)")
    parser.add_argument("--fp32", action="store_true", help="Use FP32 ONNX weights instead of INT8")
    parser.add_argument("--real-time", action="store_true", help="Sleep 320 ms between chunks to simulate live arrival")
    parser.add_argument("--threads", type=int, default=1, help="Number of CPU threads")
    args = parser.parse_args()

    recognizer = build_recognizer(
        args.model,
        use_int8=not args.fp32,
        num_threads=args.threads,
        mode=args.mode,
        revision=args.revision,
    )
    if args.microphone:
        stream_microphone(recognizer)
    elif args.audio:
        stream_file_chunks(recognizer, args.audio, chunk_ms=320, real_time_paced=args.real_time)
    else:
        parser.error("Provide either `--audio <path>` or `--microphone`.")


if __name__ == "__main__":
    main()
