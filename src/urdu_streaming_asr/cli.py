import argparse
import json
from pathlib import Path
import sys

import numpy as np

from .inference import UrduStreamingRecognizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Urdu Streaming ASR CLI (file and microphone transcription)")
    parser.add_argument("--model-dir", type=Path, required=True, help="Directory containing exported ONNX files and tokens.txt")
    parser.add_argument("--use-int8", action="store_true", help="Use INT8 quantized ONNX models")
    parser.add_argument("--num-threads", type=int, default=1, help="CPU threads for ONNX Runtime")
    parser.add_argument("--chunk-ms", type=int, default=320, help="Streaming chunk size in milliseconds")
    parser.add_argument("--audio", type=Path, default=None, help="Path to audio file for streaming transcription")
    parser.add_argument("--mic", action="store_true", help="Run live local microphone transcription")
    parser.add_argument("--mic-seconds", type=float, default=15.0, help="Duration to capture from microphone")
    parser.add_argument("--json-out", type=Path, default=None, help="Optional JSON file for output and metrics")
    return parser.parse_args()


def resolve_onnx_files(model_dir: Path, use_int8: bool):
    suffix = ".int8.onnx" if use_int8 else ".onnx"
    enc = sorted(model_dir.glob(f"encoder*{suffix}"))
    dec = sorted(model_dir.glob(f"decoder*{suffix}"))
    joi = sorted(model_dir.glob(f"joiner*{suffix}"))
    if not use_int8:
        enc = [p for p in enc if not p.name.endswith(".int8.onnx")]
        dec = [p for p in dec if not p.name.endswith(".int8.onnx")]
        joi = [p for p in joi if not p.name.endswith(".int8.onnx")]
    tokens = model_dir / "tokens.txt"
    if not (enc and dec and joi and tokens.exists()):
        raise FileNotFoundError(f"Could not find complete ONNX model set in {model_dir} (use_int8={use_int8})")
    return enc[0], dec[0], joi[0], tokens


def main() -> None:
    args = parse_args()
    enc, dec, joi, tokens = resolve_onnx_files(args.model_dir, args.use_int8)
    recognizer = UrduStreamingRecognizer(
        encoder_path=enc,
        decoder_path=dec,
        joiner_path=joi,
        tokens_path=tokens,
        num_threads=args.num_threads,
    )

    if args.audio is not None:
        final_text, events, metrics = recognizer.transcribe_file_streaming(args.audio, chunk_ms=args.chunk_ms)
        for ev in events:
            tag = "FINAL" if ev.is_final else "PARTIAL"
            print(f"[{tag} t={ev.audio_time_sec:.2f}s wall={ev.wall_time_sec:.3f}s] {ev.text}")
        print(f"\nMETRICS: {json.dumps(metrics, ensure_ascii=False)}")
        if args.json_out:
            args.json_out.parent.mkdir(parents=True, exist_ok=True)
            args.json_out.write_text(
                json.dumps(
                    {
                        "audio": str(args.audio),
                        "final_text": final_text,
                        "events": [ev.__dict__ for ev in events],
                        "metrics": metrics,
                    },
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
        return

    if args.mic:
        try:
            import sounddevice as sd
        except ImportError:
            print("Install 'sounddevice' to use --mic live capture.", file=sys.stderr)
            sys.exit(1)
        sr = 16000
        block = int(sr * (args.chunk_ms / 1000.0))
        print(f"Listening on default microphone for {args.mic_seconds}s (chunk={args.chunk_ms}ms)...")
        last_partial = ""

        def callback(indata, frames, time_info, status):
            nonlocal last_partial
            chunk = indata[:, 0].astype(np.float32)
            cur = recognizer.accept_chunk(chunk, sr)
            if cur != last_partial:
                print(f"[PARTIAL] {cur}", flush=True)
                last_partial = cur
            if recognizer.is_endpoint() and cur:
                fin = recognizer.finalize()
                print(f"[ENDPOINT FINAL] {fin}", flush=True)
                recognizer.reset()
                last_partial = ""

        with sd.InputStream(samplerate=sr, channels=1, dtype="float32", blocksize=block, callback=callback):
            sd.sleep(int(args.mic_seconds * 1000))
        final_text = recognizer.finalize()
        if final_text:
            print(f"[FINAL] {final_text}")
        return

    raise SystemExit("Specify either --audio <file> or --mic")


if __name__ == "__main__":
    main()
