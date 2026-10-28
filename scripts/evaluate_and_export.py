import argparse
from collections import defaultdict
import json
import math
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import time
from typing import Any, Dict, List, Tuple

import jiwer
from lhotse import CutSet
from lhotse.features import Fbank, FbankConfig
import numpy as np
import sentencepiece as spm
import soundfile as sf
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

ROOT = Path(__file__).resolve().parents[1]
ICEFALL_ROOT = ROOT / "third_party" / "icefall"
RECIPE_STREAM = ICEFALL_ROOT / "egs" / "librispeech" / "ASR" / "pruned_transducer_stateless7_streaming"
RECIPE_BASE = ICEFALL_ROOT / "egs" / "librispeech" / "ASR" / "pruned_transducer_stateless7"

for p in (RECIPE_BASE, ICEFALL_ROOT, RECIPE_STREAM, ROOT / "scripts", ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from train_transducer import build_transducer, streaming_greedy_decode_waveform
from urdu_streaming_asr.inference import UrduStreamingRecognizer
from urdu_streaming_asr.normalizer import classify_language_mix, normalize_urdu_text
from zipformer import stack_states


def compute_corpus_metrics(refs_raw: List[str], hyps_raw: List[str]) -> Dict[str, Any]:
    refs_norm = [normalize_urdu_text(r) for r in refs_raw]
    hyps_norm = [normalize_urdu_text(h) for h in hyps_raw]

    valid_raw = [(r, h) for r, h in zip(refs_raw, hyps_raw) if r.strip()]
    valid_norm = [(r, h) for r, h in zip(refs_norm, hyps_norm) if r.strip()]

    raw_wer = jiwer.wer([x[0] for x in valid_raw], [x[1] for x in valid_raw]) if valid_raw else 1.0
    raw_cer = jiwer.cer([x[0] for x in valid_raw], [x[1] for x in valid_raw]) if valid_raw else 1.0

    if valid_norm:
        out_w = jiwer.process_words([x[0] for x in valid_norm], [x[1] for x in valid_norm])
        norm_wer = out_w.wer
        norm_cer = jiwer.cer([x[0] for x in valid_norm], [x[1] for x in valid_norm])
        subs, dels, ins = out_w.substitutions, out_w.deletions, out_w.insertions
    else:
        norm_wer, norm_cer, subs, dels, ins = 1.0, 1.0, 0, 0, 0

    return {
        "utterances": len(valid_norm),
        "raw_wer": round(float(raw_wer) * 100.0, 2),
        "raw_cer": round(float(raw_cer) * 100.0, 2),
        "normalized_wer": round(float(norm_wer) * 100.0, 2),
        "normalized_cer": round(float(norm_cer) * 100.0, 2),
        "substitutions": int(subs),
        "deletions": int(dels),
        "insertions": int(ins),
    }


def export_onnx_via_upstream(
    exp_dir: Path,
    tokens_path: Path,
    epoch: int,
    avg: int,
    use_averaged_model: bool,
    release_onnx_dir: Path,
    blank_bias_adjustment: float = 0.0,
) -> Dict[str, Any]:
    from icefall.checkpoint import average_checkpoints

    if avg > 1:
        ckpts = [exp_dir / f"epoch-{e}.pt" for e in range(epoch - avg + 1, epoch + 1)]
        avg_state = average_checkpoints(ckpts, device=torch.device("cpu"))
    else:
        raw = torch.load(str(exp_dir / f"epoch-{epoch}.pt"), map_location="cpu", weights_only=False)
        avg_state = raw["model"]

    if blank_bias_adjustment != 0.0 and "joiner.output_linear.bias" in avg_state:
        avg_state["joiner.output_linear.bias"] = avg_state["joiner.output_linear.bias"].clone()
        avg_state["joiner.output_linear.bias"][0] += blank_bias_adjustment

    export_epoch = 99
    export_pt = exp_dir / f"epoch-{export_epoch}.pt"
    torch.save({"model": avg_state}, str(export_pt))

    export_script = RECIPE_STREAM / "export-onnx.py"
    cmd = [
        sys.executable,
        str(export_script),
        "--tokens",
        str(tokens_path),
        "--exp-dir",
        str(exp_dir),
        "--use-averaged-model",
        "False",
        "--epoch",
        str(export_epoch),
        "--avg",
        "1",
        "--decode-chunk-len",
        "32",
        "--num-encoder-layers",
        "2,2,2,2,2",
        "--feedforward-dims",
        "768,768,768,768,768",
        "--nhead",
        "8,8,8,8,8",
        "--encoder-dims",
        "256,256,256,256,256",
        "--attention-dims",
        "192,192,192,192,192",
        "--encoder-unmasked-dims",
        "192,192,192,192,192",
        "--zipformer-downsampling-factors",
        "1,2,4,8,2",
        "--cnn-module-kernels",
        "31,31,31,31,31",
        "--decoder-dim",
        "512",
        "--joiner-dim",
        "512",
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["PYTHONPATH"] = f"{RECIPE_BASE}:{ICEFALL_ROOT}:{RECIPE_STREAM}:" + env.get("PYTHONPATH", "")
    subprocess.run(cmd, check=True, env=env)

    release_onnx_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"epoch-{export_epoch}-avg-1"
    exported_files = {}
    for comp in ("encoder", "decoder", "joiner"):
        for ext in (".onnx", ".int8.onnx"):
            src = exp_dir / f"{comp}-{suffix}{ext}"
            dst = release_onnx_dir / f"{comp}{ext}"
            if src.exists():
                shutil.copyfile(src, dst)
                exported_files[f"{comp}{ext}"] = {
                    "path": str(dst),
                    "size_mb": round(dst.stat().st_size / (1024 ** 2), 2),
                }
    shutil.copyfile(tokens_path, release_onnx_dir / "tokens.txt")
    bpe_src = tokens_path.parent / "bpe.model"
    if bpe_src.exists():
        shutil.copyfile(bpe_src, release_onnx_dir / "bpe.model")
    return exported_files


@torch.no_grad()
def evaluate_partition_batched_streaming(
    model: torch.nn.Module,
    sp: spm.SentencePieceProcessor,
    cuts: CutSet,
    device: torch.device,
    decode_chunk_len: int = 32,
    batch_size: int = 40,
    blank_penalty: float = 0.0,
) -> Tuple[Dict[str, Any], List[str], List[str]]:
    model.eval()
    extractor = Fbank(FbankConfig(num_mel_bins=80))
    pad_length = 7
    log_eps = math.log(1e-10)
    blank_id = model.decoder.blank_id
    context_size = model.decoder.context_size

    cut_list = list(cuts)
    refs_raw: List[str] = []
    hyps_raw: List[str] = []
    by_mix_refs: Dict[str, List[str]] = defaultdict(list)
    by_mix_hyps: Dict[str, List[str]] = defaultdict(list)
    total_audio_dur = 0.0

    for b_start in range(0, len(cut_list), batch_size):
        b_cuts = cut_list[b_start : b_start + batch_size]
        N = len(b_cuts)
        feats_list = []
        for c in b_cuts:
            try:
                wav = c.load_audio()[0]
            except Exception:
                try:
                    wav_p = c.recording.sources[0].source
                    data, _ = sf.read(str(wav_p), dtype="float32", always_2d=True)
                    wav = data[:, 0]
                except Exception:
                    wav = np.zeros(max(1600, int(round(float(c.duration) * 16000))), dtype=np.float32)
            total_audio_dur += float(len(wav)) / 16000.0
            f_np = extractor.extract(wav, sampling_rate=16000)
            f_t = torch.from_numpy(f_np).to(device=device, dtype=torch.float32)
            f_pad = F.pad(f_t, (0, 0, 0, pad_length + decode_chunk_len), mode="constant", value=log_eps)
            feats_list.append(f_pad)

        max_frames = max(f.size(0) for f in feats_list)
        feats_batch = pad_sequence(feats_list, batch_first=True, padding_value=log_eps)
        orig_frames = [f.size(0) for f in feats_list]

        states = stack_states([model.encoder.get_init_state(device=device) for _ in range(N)])
        hyps = [[-1] * (context_size - 1) + [blank_id] for _ in range(N)]
        dec_in = torch.tensor([h[-context_size:] for h in hyps], device=device, dtype=torch.int64)
        dec_out = model.joiner.decoder_proj(model.decoder(dec_in, need_pad=False))

        pos = 0
        while pos < max_frames:
            chunk_len = min(max_frames - pos, decode_chunk_len + pad_length)
            chunk_x = feats_batch[:, pos : pos + chunk_len, :]
            if chunk_x.size(1) < 23:
                pad_more = 23 - chunk_x.size(1)
                chunk_x = F.pad(chunk_x, (0, 0, 0, pad_more), mode="constant", value=log_eps)
                chunk_len = 23
            x_lens = torch.tensor(
                [max(23, min(max(0, orig_frames[i] - pos), chunk_len)) for i in range(N)],
                device=device,
                dtype=torch.int64,
            )
            active_mask = [pos < orig_frames[i] for i in range(N)]
            pos += decode_chunk_len

            enc_out, enc_out_lens, states = model.encoder.streaming_forward(x=chunk_x, x_lens=x_lens, states=states)
            enc_out = model.joiner.encoder_proj(enc_out)
            T = enc_out.size(1)
            for t in range(T):
                cur_enc = enc_out[:, t : t + 1, :]
                logits = model.joiner(
                    cur_enc.unsqueeze(2),
                    dec_out.unsqueeze(1),
                    project_input=False,
                ).squeeze(1).squeeze(1)
                if blank_penalty != 0.0:
                    logits[:, 0] -= blank_penalty
                y = logits.argmax(dim=-1).tolist()
                emitted = False
                for i, v in enumerate(y):
                    if active_mask[i] and t < int(enc_out_lens[i].item()) and v != blank_id:
                        hyps[i].append(v)
                        emitted = True
                if emitted:
                    dec_in = torch.tensor([h[-context_size:] for h in hyps], device=device, dtype=torch.int64)
                    dec_out = model.joiner.decoder_proj(model.decoder(dec_in, need_pad=False))

        for c, h in zip(b_cuts, hyps):
            toks = h[context_size:]
            hyp_t = sp.decode(toks) if toks else ""
            ref_t = c.supervisions[0].custom.get("text_raw") or c.supervisions[0].text
            refs_raw.append(ref_t)
            hyps_raw.append(hyp_t)
            mix = classify_language_mix(ref_t)
            by_mix_refs[mix].append(ref_t)
            by_mix_hyps[mix].append(hyp_t)

    overall = compute_corpus_metrics(refs_raw, hyps_raw)
    mix_breakdown = {
        k: compute_corpus_metrics(by_mix_refs[k], by_mix_hyps[k])
        for k in sorted(by_mix_refs.keys())
    }
    return {
        "evaluated_utterances": len(cut_list),
        "evaluated_audio_hours": round(total_audio_dur / 3600.0, 4),
        "accuracy": overall,
        "language_mix_breakdown": mix_breakdown,
    }, refs_raw, hyps_raw


def benchmark_sherpa_onnx_in_fresh_process(
    onnx_dir: Path,
    cuts_jsonl_gz: Path,
    use_int8: bool,
    num_threads: int,
    max_cuts: int = 80,
) -> Dict[str, Any]:
    subset_cuts = list(CutSet.from_jsonl(str(cuts_jsonl_gz)))[:max_cuts]
    items_payload = [
        {
            "audio_path": str(c.recording.sources[0].source),
            "ref_text": c.supervisions[0].custom.get("text_raw") or c.supervisions[0].text,
        }
        for c in subset_cuts
    ]
    worker_code = f"""
import json, os, resource, time
from pathlib import Path
import jiwer
import numpy as np
import soundfile as sf
import sys
sys.path.insert(0, {str(ROOT / 'src')!r})
from urdu_streaming_asr.inference import UrduStreamingRecognizer
from urdu_streaming_asr.normalizer import normalize_urdu_text

def get_proc_mem_mib():
    rss, hwm = 0.0, 0.0
    try:
        with open("/proc/self/status", "r") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    rss = int(line.split()[1]) / 1024.0
                elif line.startswith("VmHWM:"):
                    hwm = int(line.split()[1]) / 1024.0
    except Exception:
        pass
    return round(rss, 2), round(hwm, 2)

rss_before_load, _ = get_proc_mem_mib()
onnx_dir = Path({str(onnx_dir)!r})
suffix = ".int8.onnx" if {use_int8!r} else ".onnx"
rec = UrduStreamingRecognizer(
    encoder_path=onnx_dir / f"encoder{{suffix}}",
    decoder_path=onnx_dir / f"decoder{{suffix}}",
    joiner_path=onnx_dir / f"joiner{{suffix}}",
    tokens_path=onnx_dir / "tokens.txt",
    num_threads={num_threads},
)
rss_after_load, _ = get_proc_mem_mib()
items = {items_payload!r}
first_audio = items[0]["audio_path"]
_, _, cold_metrics = rec.transcribe_file_streaming(first_audio, chunk_ms=320)

refs_raw, hyps_raw = [], []
total_audio_dur = 0.0
total_compute_wall = 0.0
compute_first_partials = []
paced_first_partials = []
finalize_delays = []
chunk_compute_times_ms = []

for item in items:
    audio_p = item["audio_path"]
    ref_t = item["ref_text"]
    rec.reset()
    data, sr = sf.read(str(audio_p), dtype="float32", always_2d=True)
    mono = data[:, 0]
    chunk_samples = int(0.320 * sr)
    n_chunks = (len(mono) + chunk_samples - 1) // chunk_samples
    t0 = time.perf_counter()
    first_compute_sec = None
    first_paced_sec = None
    for idx in range(n_chunks):
        chunk = mono[idx * chunk_samples : (idx + 1) * chunk_samples]
        tc0 = time.perf_counter()
        cur = rec.accept_chunk(chunk, sr)
        tc1 = time.perf_counter()
        chunk_compute_times_ms.append((tc1 - tc0) * 1000.0)
        if cur and first_compute_sec is None:
            first_compute_sec = tc1 - t0
            first_paced_sec = ((idx + 1) * 0.320) + (tc1 - tc0)
    tf0 = time.perf_counter()
    final_text = rec.finalize()
    tf1 = time.perf_counter()
    wall = tf1 - t0
    dur = len(mono) / float(sr)
    total_audio_dur += dur
    total_compute_wall += wall
    if first_compute_sec is not None:
        compute_first_partials.append(first_compute_sec * 1000.0)
        paced_first_partials.append(first_paced_sec * 1000.0)
    finalize_delays.append((tf1 - tf0) * 1000.0)
    refs_raw.append(ref_t)
    hyps_raw.append(final_text)

rss_after_decode, hwm_after_decode = get_proc_mem_mib()
ru_rss_mb = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 2)
peak_rss_mb = hwm_after_decode if hwm_after_decode > 0 else ru_rss_mb

refs_norm = [normalize_urdu_text(r) for r in refs_raw]
hyps_norm = [normalize_urdu_text(h) for h in hyps_raw]
valid_raw = [(r, h) for r, h in zip(refs_raw, hyps_raw) if r.strip()]
valid_norm = [(r, h) for r, h in zip(refs_norm, hyps_norm) if r.strip()]
out_w = jiwer.process_words([x[0] for x in valid_norm], [x[1] for x in valid_norm])
acc = {{
    "utterances": len(valid_norm),
    "raw_wer": round(float(jiwer.wer([x[0] for x in valid_raw], [x[1] for x in valid_raw])) * 100.0, 2),
    "raw_cer": round(float(jiwer.cer([x[0] for x in valid_raw], [x[1] for x in valid_raw])) * 100.0, 2),
    "normalized_wer": round(float(out_w.wer) * 100.0, 2),
    "normalized_cer": round(float(jiwer.cer([x[0] for x in valid_norm], [x[1] for x in valid_norm])) * 100.0, 2),
    "substitutions": int(out_w.substitutions),
    "deletions": int(out_w.deletions),
    "insertions": int(out_w.insertions),
}}

out = {{
    "precision": "int8" if {use_int8!r} else "fp32",
    "num_threads": {num_threads},
    "evaluated_utterances": len(items),
    "evaluated_audio_hours": round(total_audio_dur / 3600.0, 4),
    "cold_start_rtf": cold_metrics["rtf"],
    "warm_average_rtf": round(total_compute_wall / max(1e-6, total_audio_dur), 5),
    "mean_chunk_compute_ms": round(float(np.mean(chunk_compute_times_ms)), 2),
    "p95_chunk_compute_ms": round(float(np.percentile(chunk_compute_times_ms, 95)), 2),
    "mean_decoder_exec_to_first_partial_ms": round(float(np.mean(compute_first_partials)), 2) if compute_first_partials else None,
    "mean_paced_stream_time_to_first_nonempty_partial_ms": round(float(np.mean(paced_first_partials)), 2) if paced_first_partials else None,
    "p95_paced_stream_time_to_first_nonempty_partial_ms": round(float(np.percentile(paced_first_partials, 95)), 2) if paced_first_partials else None,
    "mean_finalization_delay_ms": round(float(np.mean(finalize_delays)), 2),
    "algorithmic_chunk_accumulation_ms": 320.0,
    "algorithmic_lookahead_ms": 40.0,
    "rss_before_load_mib": rss_before_load,
    "rss_after_load_mib": rss_after_load,
    "rss_after_decode_mib": rss_after_decode,
    "net_model_and_runtime_rss_mib": round(max(0.0, peak_rss_mb - rss_before_load), 2),
    "fresh_process_peak_rss_mib": peak_rss_mb,
    "accuracy": acc,
}}
print("BENCHMARK_JSON:" + json.dumps(out, ensure_ascii=False), flush=True)
os._exit(0)
"""
    out_str = subprocess.check_output(
        [sys.executable, "-c", worker_code],
        text=True,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
    )
    for line in out_str.splitlines():
        if line.startswith("BENCHMARK_JSON:"):
            return json.loads(line[len("BENCHMARK_JSON:") :])
    raise RuntimeError(f"Failed to parse benchmark output: {out_str}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate trained Urdu Streaming ASR and export sherpa-onnx package")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--exp-dir", type=Path, default=ROOT / "exp" / "urdu_zipformer_20m_v2")
    parser.add_argument("--reports-dir", type=Path, default=ROOT / "reports")
    parser.add_argument("--release-dir", type=Path, default=ROOT / "release")
    parser.add_argument("--epoch", type=int, default=14)
    parser.add_argument("--avg", type=int, default=4)
    parser.add_argument("--use-averaged-model", type=int, default=0)
    args = parser.parse_args()

    onnx_dir = args.release_dir / "onnx"
    weights_dir = args.release_dir / "weights"
    onnx_dir.mkdir(parents=True, exist_ok=True)
    weights_dir.mkdir(parents=True, exist_ok=True)
    args.reports_dir.mkdir(parents=True, exist_ok=True)

    tokens_path = args.data_dir / "lang_bpe_1024" / "tokens.txt"
    bpe_path = args.data_dir / "lang_bpe_1024" / "bpe.model"

    print(f"[1/4] Exporting checkpoint (epoch={args.epoch}, avg={args.avg}, blank_bias_adjustment=0.0) to ONNX FP32 and INT8...")
    exported_files = export_onnx_via_upstream(
        exp_dir=args.exp_dir,
        tokens_path=tokens_path,
        epoch=args.epoch,
        avg=args.avg,
        use_averaged_model=bool(args.use_averaged_model),
        release_onnx_dir=onnx_dir,
        blank_bias_adjustment=0.0,
    )

    best_pt = args.exp_dir / "best-valid-loss.pt"
    best_cer_pt = args.exp_dir / "best-dev-cer.pt"
    epoch_pt = args.exp_dir / "epoch-99.pt"
    if best_pt.exists():
        shutil.copyfile(best_pt, weights_dir / "best-valid-loss.pt")
    if best_cer_pt.exists():
        shutil.copyfile(best_cer_pt, weights_dir / "best-dev-cer.pt")
    if epoch_pt.exists():
        shutil.copyfile(epoch_pt, weights_dir / f"epoch-{args.epoch}-avg-{args.avg}.pt")

    print("[2/4] Checking per-clip PyTorch vs ONNX FP32 vs ONNX INT8 decoding parity on 25 validation cuts...")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    sp = spm.SentencePieceProcessor()
    sp.load(str(bpe_path))
    pt_model = build_transducer(vocab_size=sp.get_piece_size(), blank_id=0, context_size=2).to(device)
    ckpt_to_load = epoch_pt if epoch_pt.exists() else best_pt
    raw_ckpt = torch.load(str(ckpt_to_load), map_location="cpu", weights_only=False)
    pt_model.load_state_dict(raw_ckpt["model"], strict=True)

    dev_cuts = CutSet.from_jsonl(str(args.data_dir / "manifests" / "cuts_dev.jsonl.gz"))
    parity_cuts = list(dev_cuts)[:25]
    onnx_rec_fp32 = UrduStreamingRecognizer(
        encoder_path=onnx_dir / "encoder.onnx",
        decoder_path=onnx_dir / "decoder.onnx",
        joiner_path=onnx_dir / "joiner.onnx",
        tokens_path=onnx_dir / "tokens.txt",
        num_threads=2,
    )
    onnx_rec_int8 = UrduStreamingRecognizer(
        encoder_path=onnx_dir / "encoder.int8.onnx",
        decoder_path=onnx_dir / "decoder.int8.onnx",
        joiner_path=onnx_dir / "joiner.int8.onnx",
        tokens_path=onnx_dir / "tokens.txt",
        num_threads=2,
    )
    pt_hyps, ox32_hyps, ox8_hyps, parity_refs = [], [], [], []
    per_clip_parity = []
    exact_pt_ox32_matches = 0
    for c in parity_cuts:
        wav_p = c.recording.sources[0].source
        wav, _ = sf.read(str(wav_p), dtype="float32")
        pt_text_raw, pt_toks = streaming_greedy_decode_waveform(pt_model, sp, wav, device)
        ox32_text, _, _ = onnx_rec_fp32.transcribe_file_streaming(wav_p)
        ox8_text, _, _ = onnx_rec_int8.transcribe_file_streaming(wav_p)
        ref_norm = normalize_urdu_text(c.supervisions[0].text)
        pt_norm = normalize_urdu_text(pt_text_raw)
        ox32_norm = normalize_urdu_text(ox32_text)
        ox8_norm = normalize_urdu_text(ox8_text)
        if pt_norm == ox32_norm:
            exact_pt_ox32_matches += 1
        parity_refs.append(ref_norm)
        pt_hyps.append(pt_norm)
        ox32_hyps.append(ox32_norm)
        ox8_hyps.append(ox8_norm)
        per_clip_parity.append(
            {
                "id": c.id,
                "duration_sec": round(float(c.duration), 2),
                "ref": ref_norm,
                "pytorch_streaming": pt_norm,
                "onnx_fp32_streaming": ox32_norm,
                "onnx_int8_streaming": ox8_norm,
                "pt_token_ids": pt_toks,
                "exact_match_pt_vs_onnx_fp32": pt_norm == ox32_norm,
            }
        )

    print("[3/4] Evaluating across FULL development (769 cuts) and FULL held-out test (938 cuts) partitions...")
    manifests_dir = args.data_dir / "manifests"
    split_results: Dict[str, Any] = {}
    all_dev_refs, all_dev_hyps = [], []
    all_test_refs, all_test_hyps = [], []

    for cut_file in sorted(manifests_dir.glob("cuts_*.jsonl.gz")):
        name = cut_file.name.replace("cuts_", "").replace(".jsonl.gz", "")
        if name in ("train", "dev", "test"):
            continue
        cs = CutSet.from_jsonl(str(cut_file))
        res, r_raw, h_raw = evaluate_partition_batched_streaming(
            model=pt_model,
            sp=sp,
            cuts=cs,
            device=device,
            decode_chunk_len=32,
            batch_size=40,
            blank_penalty=0.0,
        )
        split_results[name] = res
        if name.startswith("dev_"):
            all_dev_refs.extend(r_raw)
            all_dev_hyps.extend(h_raw)
        elif name.startswith("test_"):
            all_test_refs.extend(r_raw)
            all_test_hyps.extend(h_raw)
        print(
            f"  {name} (n={res['evaluated_utterances']}): "
            f"Norm WER={res['accuracy']['normalized_wer']}% CER={res['accuracy']['normalized_cer']}%"
        )

    pooled_dev_metrics = compute_corpus_metrics(all_dev_refs, all_dev_hyps)
    pooled_test_metrics = compute_corpus_metrics(all_test_refs, all_test_hyps)

    print("[4/4] Benchmarking CPU RTF, Fresh-Process Peak RSS, and Paced Streaming Latency (FP32 vs INT8, 1 vs 4 threads)...")
    test_cuts_path = manifests_dir / "cuts_test.jsonl.gz"
    cpu_benchmarks = []
    for use_int8 in (False, True):
        for n_threads in (1, 4):
            bm = benchmark_sherpa_onnx_in_fresh_process(
                onnx_dir=onnx_dir,
                cuts_jsonl_gz=test_cuts_path,
                use_int8=use_int8,
                num_threads=n_threads,
                max_cuts=80,
            )
            cpu_benchmarks.append(bm)
            print(
                f"  {bm['precision']} threads={n_threads}: RTF={bm['warm_average_rtf']} "
                f"chunk_compute={bm['mean_chunk_compute_ms']}ms "
                f"paced_first_partial={bm['mean_paced_stream_time_to_first_nonempty_partial_ms']}ms "
                f"RSS={bm['fresh_process_peak_rss_mib']}MiB "
                f"WER={bm['accuracy']['normalized_wer']}% CER={bm['accuracy']['normalized_cer']}%"
            )

    dev_wers = [v["accuracy"]["normalized_wer"] for k, v in split_results.items() if k.startswith("dev_")]
    dev_cers = [v["accuracy"]["normalized_cer"] for k, v in split_results.items() if k.startswith("dev_")]
    test_wers = [v["accuracy"]["normalized_wer"] for k, v in split_results.items() if k.startswith("test_")]
    test_cers = [v["accuracy"]["normalized_cer"] for k, v in split_results.items() if k.startswith("test_")]

    eval_summary = {
        "evaluated_checkpoint": str(weights_dir / f"epoch-{args.epoch}-avg-{args.avg}.pt"),
        "epoch": args.epoch,
        "avg": args.avg,
        "blank_bias_adjustment": 0.0,
        "evaluation_disclosure": (
            "Full official splits evaluated (769 dev utterances across 3 sources; 938 test utterances across 6 sources). "
            "Note: the first 50 utterances of each test partition were previously inspected during the initial v1 failure diagnosis."
        ),
        "exported_onnx_files": exported_files,
        "pytorch_vs_onnx_parity_25_cuts": {
            "pytorch_norm_wer": round(jiwer.wer(parity_refs, pt_hyps) * 100.0, 2),
            "onnx_fp32_norm_wer": round(jiwer.wer(parity_refs, ox32_hyps) * 100.0, 2),
            "onnx_int8_norm_wer": round(jiwer.wer(parity_refs, ox8_hyps) * 100.0, 2),
            "pytorch_norm_cer": round(jiwer.cer(parity_refs, pt_hyps) * 100.0, 2),
            "onnx_fp32_norm_cer": round(jiwer.cer(parity_refs, ox32_hyps) * 100.0, 2),
            "onnx_int8_norm_cer": round(jiwer.cer(parity_refs, ox8_hyps) * 100.0, 2),
            "exact_clip_matches_pt_vs_onnx_fp32": f"{exact_pt_ox32_matches}/25",
            "sample_clips": per_clip_parity[:10],
        },
        "source_balanced_macro_dev_wer": round(float(np.mean(dev_wers)), 2),
        "source_balanced_macro_dev_cer": round(float(np.mean(dev_cers)), 2),
        "pooled_dev_accuracy": pooled_dev_metrics,
        "source_balanced_macro_test_wer": round(float(np.mean(test_wers)), 2),
        "source_balanced_macro_test_cer": round(float(np.mean(test_cers)), 2),
        "pooled_test_accuracy": pooled_test_metrics,
        "before_repair_v1_comparison": {
            "v1_checkpoint": "exp/urdu_zipformer_20m/epoch-25-avg-5.pt",
            "v1_macro_dev_wer": 97.22,
            "v1_macro_dev_cer": 86.09,
            "v1_macro_test_wer": 100.96,
            "v1_macro_test_cer": 90.02,
        },
        "per_partition_results": split_results,
        "cpu_deployment_benchmarks": cpu_benchmarks,
    }

    old_eval_path = args.reports_dir / "evaluation_report.json"
    if old_eval_path.exists():
        try:
            old_data = json.loads(old_eval_path.read_text(encoding="utf-8"))
            if "external_baseline_comparison" in old_data:
                eval_summary["external_baseline_comparison"] = old_data["external_baseline_comparison"]
        except Exception:
            pass

    old_eval_path.write_text(
        json.dumps(eval_summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("EVAL_AND_EXPORT_V2_COMPLETE")


if __name__ == "__main__":
    main()
