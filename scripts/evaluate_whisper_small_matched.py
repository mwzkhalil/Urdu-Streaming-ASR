import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

import jiwer
import torch
from lhotse import CutSet
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from urdu_streaming_asr.normalizer import normalize_urdu_text


def compute_metrics(refs_raw: List[str], hyps_raw: List[str]) -> Dict[str, float | int]:
    refs_norm = [normalize_urdu_text(r) for r in refs_raw]
    hyps_norm = [normalize_urdu_text(h) for h in hyps_raw]
    valid_raw = [(r, h) for r, h in zip(refs_raw, hyps_raw) if r.strip()]
    valid_norm = [(r, h) for r, h in zip(refs_norm, hyps_norm) if r.strip()]

    raw_wer = jiwer.wer([x[0] for x in valid_raw], [x[1] for x in valid_raw]) if valid_raw else 1.0
    raw_cer = jiwer.cer([x[0] for x in valid_raw], [x[1] for x in valid_raw]) if valid_raw else 1.0

    out_w = jiwer.process_words([x[0] for x in valid_norm], [x[1] for x in valid_norm])
    norm_cer = jiwer.cer([x[0] for x in valid_norm], [x[1] for x in valid_norm])
    return {
        "utterances": len(valid_norm),
        "raw_wer": round(float(raw_wer) * 100.0, 2),
        "raw_cer": round(float(raw_cer) * 100.0, 2),
        "normalized_wer": round(float(out_w.wer) * 100.0, 2),
        "normalized_cer": round(float(norm_cer) * 100.0, 2),
        "substitutions": int(out_w.substitutions),
        "deletions": int(out_w.deletions),
        "insertions": int(out_w.insertions),
    }


def main() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    snap = next((ROOT / "data/whisper_small_local/models--openai--whisper-small/snapshots").glob("*"))
    processor = AutoProcessor.from_pretrained(str(snap), local_files_only=True)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(str(snap), dtype=torch.float16, local_files_only=True).to(device)
    model.eval()

    v4_man = ROOT / "data/manifests_v4_baseline"
    v5_eval = ROOT / "data/manifests_v5_eval"

    splits = [
        ("dev_fleurs_ur_pk", v4_man / "cuts_dev_fleurs_ur_pk.jsonl.gz"),
        ("dev_indicvoices_ur", v4_man / "cuts_dev_indicvoices_ur.jsonl.gz"),
        ("dev_kathbath_ur", v4_man / "cuts_dev_kathbath_ur.jsonl.gz"),
        ("test_csalt_voice", v4_man / "cuts_test_csalt_voice.jsonl.gz"),
        ("test_fleurs_ur_pk", v4_man / "cuts_test_fleurs_ur_pk.jsonl.gz"),
        ("test_indicvoices_ur", v4_man / "cuts_test_indicvoices_ur.jsonl.gz"),
        ("test_kathbath_ur", v4_man / "cuts_test_kathbath_ur.jsonl.gz"),
        ("test_rasa_ur", v4_man / "cuts_test_rasa_ur.jsonl.gz"),
        ("test_urduspeech_benchmark_cs", v4_man / "cuts_test_urduspeech_benchmark_cs.jsonl.gz"),
        ("v5_reserved_dev", v5_eval / "cuts_v5_reserved_dev.jsonl.gz"),
        ("v5_reserved_test", v5_eval / "cuts_v5_reserved_test.jsonl.gz"),
    ]

    split_results = {}
    all_dev_refs, all_dev_hyps = [], []
    all_test_refs, all_test_hyps = [], []

    for sname, spath in splits:
        cuts = list(CutSet.from_jsonl(str(spath)))
        refs_r, hyps_r = [], []
        bs = 24
        for i in range(0, len(cuts), bs):
            batch_c = cuts[i : i + bs]
            wavs = [c.load_audio()[0] for c in batch_c]
            inputs = processor(wavs, sampling_rate=16000, return_tensors="pt")
            feats = inputs.input_features.to(device=device, dtype=torch.float16)
            with torch.no_grad():
                gen_ids = model.generate(feats, language="ur", task="transcribe", max_new_tokens=128)
            txts = processor.batch_decode(gen_ids, skip_special_tokens=True)
            for c, t in zip(batch_c, txts):
                ref = c.supervisions[0].custom.get("text_raw") or c.supervisions[0].text
                refs_r.append(ref)
                hyps_r.append(t)

        m = compute_metrics(refs_r, hyps_r)
        split_results[sname] = m
        if sname.startswith("dev_"):
            all_dev_refs.extend(refs_r)
            all_dev_hyps.extend(hyps_r)
        elif sname.startswith("test_"):
            all_test_refs.extend(refs_r)
            all_test_hyps.extend(hyps_r)
        print(f"  [whisper-small] {sname:30s} (n={m['utterances']}): Norm WER={m['normalized_wer']}% CER={m['normalized_cer']}%", flush=True)

    dev_keys = ["dev_fleurs_ur_pk", "dev_indicvoices_ur", "dev_kathbath_ur"]
    test_keys = ["test_csalt_voice", "test_fleurs_ur_pk", "test_indicvoices_ur", "test_kathbath_ur", "test_rasa_ur", "test_urduspeech_benchmark_cs"]
    import numpy as np
    payload = {
        "model": "openai/whisper-small (244M parameters, offline non-streaming encoder-decoder)",
        "inference_settings": {
            "language": "ur",
            "task": "transcribe",
            "dtype": "torch.float16",
            "max_new_tokens": 128,
            "normalizer": "urdu_norm_v1.0",
        },
        "dev_769_summary": {
            "macro_normalized_wer": round(float(np.mean([split_results[k]["normalized_wer"] for k in dev_keys])), 2),
            "macro_normalized_cer": round(float(np.mean([split_results[k]["normalized_cer"] for k in dev_keys])), 2),
            "pooled": compute_metrics(all_dev_refs, all_dev_hyps),
        },
        "test_938_summary": {
            "macro_normalized_wer": round(float(np.mean([split_results[k]["normalized_wer"] for k in test_keys])), 2),
            "macro_normalized_cer": round(float(np.mean([split_results[k]["normalized_cer"] for k in test_keys])), 2),
            "pooled": compute_metrics(all_test_refs, all_test_hyps),
        },
        "per_partition_results": split_results,
    }
    (ROOT / "reports/whisper_small_matched_eval.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print("WHISPER_SMALL_MATCHED_EVAL_COMPLETE:", json.dumps(payload["test_938_summary"], indent=2))


if __name__ == "__main__":
    main()
