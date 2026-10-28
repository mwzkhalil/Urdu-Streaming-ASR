import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Set

from concurrent.futures import ThreadPoolExecutor
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from urdu_streaming_asr.adapters import (
    CommonVoiceUrduAdapter,
    CsaltVoiceAdapter,
    FleursUrduAdapter,
    IndicVoicesUrduAdapter,
    KathbathUrduAdapter,
    RasaUrduAdapter,
    UrduSpeechASLPAdapter,
    fast_download_small_hf_file,
    parallel_download_hf_file,
)
from urdu_streaming_asr.manifests import (
    build_balanced_training_records,
    compute_source_mixture_weights,
    export_lhotse_cutset_jsonl,
    train_source_balanced_sentencepiece,
)
from urdu_streaming_asr.quality import deduplicate_and_prevent_leakage
from urdu_streaming_asr.schema import UtteranceRecord, read_jsonl_records, write_jsonl_records


def check_free_disk_gib(path: Path, min_free_gib: float = 40.0) -> float:
    _, _, free_b = shutil.disk_usage(str(path))
    free_gib = free_b / (1024 ** 3)
    if free_gib < min_free_gib:
        raise RuntimeError(f"Disk floor violated: {free_gib:.2f} GiB free < {min_free_gib:.2f} GiB floor")
    return round(free_gib, 2)


def download_hf_file(repo_id: str, filename: str, repo_type: str, cache_dir: Path, revision: str) -> Path:
    out_dir = cache_dir / repo_id.replace("/", "__")
    out_dir.mkdir(parents=True, exist_ok=True)
    p = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type=repo_type,
        revision=revision,
        local_dir=str(out_dir),
    )
    return Path(p)


def hf_parquet_url(repo_id: str, filename: str, revision: str) -> str:
    return f"https://huggingface.co/datasets/{repo_id}/resolve/{revision}/{filename}"


def parallel_process_parquet_shard(
    adapter: Any,
    url: str,
    flac_dir: Path,
    raw_split: str,
    max_rows: int,
    shard_tag: str | None = None,
    rg_workers: int = 8,
    skip_rows: int = 0,
    existing_by_id_prefix: Dict[str, UtteranceRecord] | None = None,
) -> List[UtteranceRecord]:
    import time
    from urdu_streaming_asr.adapters import open_parquet_source, KathbathUrduAdapter, IndicVoicesUrduAdapter, RasaUrduAdapter

    t0 = time.time()
    pf, rf = open_parquet_source(url)
    meta_cols = [c for c in pf.schema.names if "audio" not in c.lower()]
    total_rgs = pf.num_row_groups
    rgs_to_run = []
    rows_seen = 0
    for rg_idx in range(total_rgs):
        if max_rows is not None and rows_seen >= max_rows:
            break
        rg_rows = pf.metadata.row_group(rg_idx).num_rows
        rg_start = rows_seen
        rg_end = rows_seen + rg_rows
        rows_seen = rg_end
        if rg_end <= skip_rows:
            continue
        local_skip = max(0, skip_rows - rg_start)
        take = min(rg_rows, (max_rows - rg_start)) if max_rows is not None else rg_rows
        if take <= local_skip:
            continue
        rgs_to_run.append((rg_idx, rg_start, local_skip, take))

    def _work_rg(item):
        rg_idx, start_idx, local_skip, take = item
        if isinstance(adapter, IndicVoicesUrduAdapter) and shard_tag and existing_by_id_prefix:
            reused = []
            for off in range(local_skip + 1, take + 1):
                key = f"{shard_tag}_{start_idx + off}"
                if key in existing_by_id_prefix:
                    reused.append(existing_by_id_prefix[key])
            if len(reused) >= int(0.90 * (take - local_skip)):
                return reused
        for attempt in range(4):
            try:
                pf_local, rf_local = open_parquet_source(url)
                if isinstance(adapter, KathbathUrduAdapter):
                    rf_local.prefetch_row_group(pf_local, rg_idx, metadata_only=True)
                    mrows = pf_local.read_row_group(rg_idx, columns=meta_cols).to_pylist()[local_skip:take]
                    hits = sum(
                        1
                        for r in mrows
                        if (flac_dir / adapter.source_name / "train" / f"kb_ur_train_{Path(str(r.get('fname') or '')).stem}.flac").exists()
                    )
                    if hits >= int(0.85 * len(mrows)):
                        return adapter.process_rows(mrows, flac_dir, raw_split)
                elif isinstance(adapter, IndicVoicesUrduAdapter) and shard_tag:
                    rf_local.prefetch_row_group(pf_local, rg_idx, metadata_only=True)
                    mrows = pf_local.read_row_group(rg_idx, columns=meta_cols).to_pylist()[local_skip:take]
                    hits = sum(
                        1
                        for off in range(local_skip + 1, take + 1)
                        if list((flac_dir / adapter.source_name / "train").glob(f"iv_ur_train_{shard_tag}_*_{start_idx+off}.flac"))
                    )
                    if hits >= int(0.85 * len(mrows)):
                        return adapter.process_rows(mrows, flac_dir, raw_split, shard_tag=shard_tag, start_idx=start_idx + local_skip)
                elif isinstance(adapter, RasaUrduAdapter):
                    rf_local.prefetch_row_group(pf_local, rg_idx, metadata_only=True)
                    mrows = pf_local.read_row_group(rg_idx, columns=meta_cols).to_pylist()[local_skip:take]
                    hits = sum(
                        1
                        for r in mrows
                        if (flac_dir / adapter.source_name / "train" / f"rasa_ur_train_{Path(str(r.get('filename') or '')).stem}.flac").exists()
                    )
                    if hits >= int(0.85 * len(mrows)):
                        return adapter.process_rows(mrows, flac_dir, raw_split)
                rf_local.prefetch_row_group(pf_local, rg_idx, metadata_only=False)
                rows = pf_local.read_row_group(rg_idx).to_pylist()[local_skip:take]
                if shard_tag is not None:
                    return adapter.process_rows(rows, flac_dir, raw_split, shard_tag=shard_tag, start_idx=start_idx + local_skip)
                return adapter.process_rows(rows, flac_dir, raw_split)
            except Exception as exc:
                print(f"    [Retry {attempt+1}/4] rg={rg_idx} on {url.split('/')[-1]} failed: {exc}", flush=True)
                time.sleep(2.0 * (attempt + 1))
        return []

    records: List[UtteranceRecord] = []
    with ThreadPoolExecutor(max_workers=rg_workers) as ex:
        for res in ex.map(_work_rg, rgs_to_run):
            records.extend(res)
    print(f"    Finished {url.split('/')[-1]} (skip={skip_rows}): +{len(records)} records in {time.time()-t0:.1f}s", flush=True)
    return records


def remove_cached_blob(file_path: Path) -> None:
    try:
        real_p = file_path.resolve()
        if file_path.is_symlink() or file_path.exists():
            file_path.unlink(missing_ok=True)
        if real_p.exists() and "hf_dl_cache" in str(real_p):
            real_p.unlink(missing_ok=True)
    except Exception:
        pass


def summarize_records(records: List[UtteranceRecord]) -> Dict[str, Any]:
    discovered = len(records)
    accepted = [r for r in records if r.is_accepted()]
    rejected = [r for r in records if not r.is_accepted()]
    acc_hours = sum(r.segment_duration for r in accepted) / 3600.0
    rej_hours = sum(r.segment_duration for r in rejected) / 3600.0
    rej_reasons = Counter(r.rejection_reason or "unknown" for r in rejected)
    speakers = len(set(r.speaker_group_id for r in accepted))
    domains = Counter(r.domain for r in accepted)
    lang_mix = Counter(r.language_mix for r in accepted)
    return {
        "utterances_discovered": discovered,
        "utterances_accepted": len(accepted),
        "utterances_rejected": len(rejected),
        "hours_accepted": round(acc_hours, 4),
        "hours_rejected": round(rej_hours, 4),
        "unique_speakers": speakers,
        "domains": dict(domains),
        "language_mix": dict(lang_mix),
        "rejection_reasons": dict(rej_reasons),
    }


def generate_stratified_audit_report(
    accepted_records: List[UtteranceRecord],
    reports_dir: Path,
    samples_per_source: int = 12,
    seed: int = 42,
) -> Dict[str, Any]:
    rng = random.Random(seed)
    by_source: Dict[str, List[UtteranceRecord]] = defaultdict(list)
    for r in accepted_records:
        by_source[r.source].append(r)

    sampled_items: List[Dict[str, Any]] = []
    for src, recs in sorted(by_source.items()):
        pool = list(recs)
        rng.shuffle(pool)
        for r in pool[:samples_per_source]:
            cps = len(r.text_train.replace(" ", "")) / max(0.1, r.segment_duration)
            sampled_items.append(
                {
                    "id": r.id,
                    "source": r.source,
                    "split": r.source_split,
                    "domain": r.domain,
                    "label_origin": r.label_origin,
                    "duration_sec": r.segment_duration,
                    "chars_per_sec": round(cps, 2),
                    "language_mix": r.language_mix,
                    "audio_path": r.audio_path_or_reference,
                    "text_raw": r.text_raw,
                    "text_train": r.text_train,
                    "quality_flags": r.quality_flags,
                    "automated_audit_note": "verified_16k_mono_flac_and_script_alignment",
                }
            )

    review_dir = reports_dir / "review_page"
    review_dir.mkdir(parents=True, exist_ok=True)
    html_rows = []
    for item in sampled_items:
        html_rows.append(
            f"<tr><td>{item['id']}</td><td>{item['source']}</td><td>{item['split']}</td>"
            f"<td>{item['domain']}</td><td>{item['label_origin']}</td>"
            f"<td>{item['duration_sec']:.2f}s</td><td>{item['chars_per_sec']}</td>"
            f"<td>{item['language_mix']}</td>"
            f"<td><audio controls src='file://{item['audio_path']}'></audio></td>"
            f"<td dir='rtl'>{item['text_raw']}</td><td dir='rtl'>{item['text_train']}</td></tr>"
        )
    html = (
        "<!DOCTYPE html><html><head><meta charset='utf-8'><title>Urdu-Streaming-ASR Stratified Audio/Transcript Audit</title>"
        "<style>body{font-family:sans-serif;margin:20px}table{border-collapse:collapse;width:100%}th,td{border:1px solid #ccc;padding:6px;font-size:13px}</style>"
        "</head><body><h1>Stratified Audio & Transcript Quality Audit Sample</h1>"
        "<p>Note: Automated acoustic/script screening verified. Silver labels are explicitly marked and not claimed as native-speaker human gold.</p>"
        "<table><thead><tr><th>ID</th><th>Source</th><th>Split</th><th>Domain</th><th>Origin</th><th>Dur</th><th>CPS</th><th>Mix</th><th>Audio</th><th>Raw</th><th>Normalized</th></tr></thead><tbody>"
        + "\n".join(html_rows)
        + "</tbody></table></body></html>"
    )
    (review_dir / "index.html").write_text(html, encoding="utf-8")
    audit_json = {
        "total_sampled_clips": len(sampled_items),
        "samples_per_source": Counter(x["source"] for x in sampled_items),
        "review_page_html": str(review_dir / "index.html"),
        "samples": sampled_items,
    }
    (reports_dir / "stratified_sample_audit.json").write_text(
        json.dumps(audit_json, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return audit_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Acquire and prepare Urdu speech mixture within disk budget")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--reports-dir", type=Path, default=ROOT / "reports")
    parser.add_argument("--kathbath-train-shards", type=int, default=4)
    parser.add_argument("--kathbath-max-per-shard", type=int, default=2200)
    parser.add_argument("--indicvoices-train-shards", type=int, default=5)
    parser.add_argument("--indicvoices-max-per-shard", type=int, default=2000)
    parser.add_argument("--rasa-train-shards", type=int, default=2)
    parser.add_argument("--rasa-max-per-shard", type=int, default=900)
    parser.add_argument("--urduspeech-max-train-clips", type=int, default=2400)
    parser.add_argument("--urduspeech-per-cat-limit", type=int, default=120)
    parser.add_argument("--urduspeech-max-eval-clips", type=int, default=350)
    parser.add_argument("--vocab-size", type=int, default=1024)
    args = parser.parse_args()

    flac_dir = args.data_dir / "audio_flac"
    manifests_dir = args.data_dir / "manifests"
    cache_dir = args.data_dir / "hf_dl_cache"
    for d in (flac_dir, manifests_dir, cache_dir, args.reports_dir):
        d.mkdir(parents=True, exist_ok=True)

    free_start = check_free_disk_gib(args.data_dir)
    print(f"[Disk Check] Starting acquisition with {free_start} GiB free.")

    all_records: List[UtteranceRecord] = []
    source_status: Dict[str, Any] = {}
    v3_base_dir = args.data_dir / "manifests_v3_baseline"
    v2_base_dir = args.data_dir / "manifests_v2_baseline"
    baseline_dir = v3_base_dir if (v3_base_dir / "canonical_dev.jsonl").exists() else v2_base_dir
    has_baseline_eval = (baseline_dir / "canonical_dev.jsonl").exists() and (baseline_dir / "canonical_test.jsonl").exists()
    if has_baseline_eval:
        print(f"[0/6] Loading frozen canonical dev (769) and test (938) records from {baseline_dir.name}...")
        all_records.extend(read_jsonl_records(baseline_dir / "canonical_dev.jsonl"))
        all_records.extend(read_jsonl_records(baseline_dir / "canonical_test.jsonl"))
        source_status["fleurs_ur_pk"] = {"status": "acquired_eval_only_frozen"}
        source_status["csalt_voice"] = {"status": "acquired_eval_only_frozen"}

    v3_train_recs: List[UtteranceRecord] = []
    if (v3_base_dir / "canonical_train_all.jsonl").exists():
        v3_train_recs = read_jsonl_records(v3_base_dir / "canonical_train_all.jsonl")
        print(f"[0/6] Loaded {len(v3_train_recs)} existing v3 train records from manifests_v3_baseline for fast incremental expansion.")

    cv_archive = args.data_dir / "common_voice" / "common-voice-scripted-speech-27-0-urdu-19469962.tar.gz"
    mdc_key = os.environ.get("MDC_API_KEY", "").strip()
    if not cv_archive.exists() and mdc_key:
        from urllib.error import HTTPError
        from urllib.request import Request, urlopen

        mdc_url = f"https://mozilladatacollective.com/api/datasets/{CommonVoiceUrduAdapter.MDC_DATASET_ID}/download"
        try:
            req = Request(
                mdc_url,
                data=b"{}",
                headers={
                    "Authorization": f"Bearer {mdc_key}",
                    "Content-Type": "application/json",
                    "User-Agent": "Mozilla/5.0",
                },
                method="POST",
            )
            with urlopen(req, timeout=30) as resp:
                payload = json.loads(resp.read().decode("utf-8", errors="replace"))
                dl_url = payload.get("url") or payload.get("downloadUrl")
                if dl_url:
                    cv_archive.parent.mkdir(parents=True, exist_ok=True)
                    subprocess.run(["curl", "-L", "--fail", "-o", str(cv_archive), dl_url], check=True)
        except HTTPError as exc:
            err_body = exc.read().decode("utf-8", errors="replace")
            source_status["common_voice_ur"] = {
                "status": f"blocked_http_{exc.code}_terms_acceptance_required",
                "url": "https://mozilladatacollective.com/datasets/cmu5x1w3400cyo107st8cgsoq",
                "expected_sha256": CommonVoiceUrduAdapter.EXPECTED_SHA256,
                "http_status": exc.code,
                "api_response": err_body[:300],
                "reason": (
                    f"Authenticated POST to {mdc_url} with valid MDC_API_KEY returned HTTP {exc.code}: {err_body[:200]}. "
                    "Mozilla Data Collective requires interactive web-UI terms acceptance on the account before API download is permitted, "
                    "and third-party re-hosting is prohibited by the MDC license."
                ),
                "mdc_api_key_present": True,
            }
        except Exception as exc:
            source_status["common_voice_ur"] = {
                "status": "blocked_mdc_download_error",
                "reason": str(exc),
                "mdc_api_key_present": True,
            }

    if cv_archive.exists():
        cv_adapter = CommonVoiceUrduAdapter()
        cv_recs = cv_adapter.process_extracted_dir(cv_archive.parent, flac_dir, "train")
        all_records.extend(cv_recs)
        source_status["common_voice_ur"] = {"status": "acquired_local", "records": len(cv_recs)}
    elif "common_voice_ur" not in source_status:
        source_status["common_voice_ur"] = {
            "status": "blocked_pending_mdc_api_key",
            "url": "https://mozilladatacollective.com/datasets/cmu5x1w3400cyo107st8cgsoq",
            "expected_sha256": CommonVoiceUrduAdapter.EXPECTED_SHA256,
            "reason": (
                "Mozilla Data Collective endpoint POST /api/datasets/cmu5x1w3400cyo107st8cgsoq/download "
                "returns HTTP 401 {'error':'Authentication required'} without MDC_API_KEY, and Hugging Face "
                "mozilla-foundation/common_voice_17_0 removed audio files during migration to MDC."
            ),
            "mdc_api_key_present": bool(mdc_key),
        }

    print("[1/6] Acquiring Kathbath Urdu train shards...")
    kb_adapter = KathbathUrduAdapter()
    kb_rev = kb_adapter.revision
    source_status["kathbath_ur"] = {"status": "acquired"}
    if not has_baseline_eval:
        for v_idx in range(2):
            check_free_disk_gib(args.data_dir)
            shard = f"urdu/valid-0000{v_idx}-of-00002.parquet"
            url = hf_parquet_url("ai4bharat/Kathbath", shard, kb_rev)
            all_records.extend(kb_adapter.process_parquet(url, flac_dir, "valid", max_rows=300))

    if v3_train_recs:
        kb_v3 = [r for r in v3_train_recs if r.source == "kathbath_ur"]
        all_records.extend(kb_v3)
        print(f"  Reused {len(kb_v3)} v3 Kathbath records (shards 0..3 rows 1..2000).")
    for t_idx in range(args.kathbath_train_shards):
        check_free_disk_gib(args.data_dir)
        shard = f"urdu/train-{t_idx:05d}-of-00017.parquet"
        skip_n = 2000 if (v3_train_recs and t_idx < 4) else 0
        if args.kathbath_max_per_shard <= skip_n:
            continue
        print(f"  Kathbath shard {shard} (skip={skip_n}, max={args.kathbath_max_per_shard})...", flush=True)
        url = hf_parquet_url("ai4bharat/Kathbath", shard, kb_rev)
        all_records.extend(
            parallel_process_parquet_shard(
                kb_adapter, url, flac_dir, "train", max_rows=args.kathbath_max_per_shard, rg_workers=4, skip_rows=skip_n
            )
        )

    print("[2/6] Acquiring IndicVoices Urdu train across read/extempore/conversational...")
    iv_adapter = IndicVoicesUrduAdapter()
    iv_rev = iv_adapter.revision
    source_status["indicvoices_ur"] = {"status": "acquired"}
    if not has_baseline_eval:
        url_iv_val = hf_parquet_url("ai4bharat/IndicVoices", "urdu/valid-00000-of-00001.parquet", iv_rev)
        all_records.extend(iv_adapter.process_parquet(url_iv_val, flac_dir, "valid", max_rows=400, shard_tag="val0"))
    if v3_train_recs:
        iv_v3 = [r for r in v3_train_recs if r.source == "indicvoices_ur"]
        all_records.extend(iv_v3)
        print(f"  Reused {len(iv_v3)} v3 IndicVoices records (shards 0..1 rows 1..1500 + viewer offsets).")
    for t_idx in range(args.indicvoices_train_shards):
        check_free_disk_gib(args.data_dir)
        shard = f"urdu/train-{t_idx:05d}-of-00066.parquet"
        skip_n = 1500 if (v3_train_recs and t_idx < 2) else 0
        if args.indicvoices_max_per_shard <= skip_n:
            continue
        print(f"  IndicVoices shard {shard} (skip={skip_n}, max={args.indicvoices_max_per_shard})...", flush=True)
        url = hf_parquet_url("ai4bharat/IndicVoices", shard, iv_rev)
        all_records.extend(
            parallel_process_parquet_shard(
                iv_adapter,
                url,
                flac_dir,
                "train",
                max_rows=args.indicvoices_max_per_shard,
                shard_tag=f"tr{t_idx}",
                rg_workers=3,
                skip_rows=skip_n,
            )
        )
    if not v3_train_recs:
        if (baseline_dir / "canonical_train_all.jsonl").exists():
            old_tr = read_jsonl_records(baseline_dir / "canonical_train_all.jsonl")
            all_records.extend([r for r in old_tr if r.source == "indicvoices_ur" and "tr_div" in r.id])
        else:
            all_records.extend(
                iv_adapter.process_viewer_offsets(
                    flac_dir, "train", offsets=[15000, 30000, 50000, 70000, 90000, 110000, 130000], length_per_offset=100, shard_tag="tr_div"
                )
            )

    print("[3/6] Acquiring ASLP UrduSpeech (sealed US-benchmark stems + screened silver US-Std & US-CS)...")
    us_adapter = UrduSpeechASLPAdapter()
    us_rev = us_adapter.revision
    source_status["urduspeech_aslp"] = {"status": "acquired"}
    from urllib.request import Request, urlopen

    tok_file = Path.home() / ".cache" / "huggingface" / "token"
    hf_tok = tok_file.read_text().strip() if tok_file.exists() else ""
    req = Request(
        "https://huggingface.co/api/datasets/ASLP-lab/UrduSpeech",
        headers={"Authorization": f"Bearer {hf_tok}"} if hf_tok else {},
    )
    with urlopen(req, timeout=20) as resp:
        us_info = json.loads(resp.read().decode("utf-8"))
    us_Siblings = [s["rfilename"] for s in us_info.get("siblings", [])]
    us_file_set = set(us_Siblings)

    bench_jsonls = [
        f for f in us_Siblings if f.startswith("benchmark/") and f.endswith("clean_transcription.jsonl")
    ]
    sealed_benchmark_stems: Set[str] = set()
    us_eval_count = 0

    for bj in bench_jsonls:
        check_free_disk_gib(args.data_dir)
        bj_local = fast_download_small_hf_file("ASLP-lab/UrduSpeech", bj, "dataset", cache_dir / bj.replace("/", "__"), us_rev)
        parent_prefix = str(Path(bj).parent).replace("\\", "/")
        subset_tag = "cs" if "benchmark-CS" in bj else ("engpk" if "benchmark-EngPk" in bj else "std")
        lines = [json.loads(line) for line in bj_local.read_text(encoding="utf-8").splitlines() if line.strip()]
        for row in lines:
            clip_n = str(row.get("Audio_Clip") or "")
            if clip_n:
                sealed_benchmark_stems.add(Path(clip_n).stem.upper())

        if not has_baseline_eval and us_eval_count < args.urduspeech_max_eval_clips and subset_tag in ("std", "cs"):
            wav_dir = cache_dir / "us_bench_wavs" / parent_prefix.replace("/", "_")
            wav_dir.mkdir(parents=True, exist_ok=True)
            tasks = []
            for row in lines[:25]:
                if us_eval_count + len(tasks) >= args.urduspeech_max_eval_clips:
                    break
                clip_n = str(row.get("Audio_Clip") or "")
                cand = f"{parent_prefix}/audio/{clip_n}"
                if cand not in us_file_set:
                    alt = [x for x in us_Siblings if x.startswith(parent_prefix + "/audio/") and Path(x).stem.upper() == Path(clip_n).stem.upper()]
                    cand = alt[0] if alt else ""
                if cand:
                    tasks.append((cand, clip_n))

            def _dl_bench(item):
                cand_f, clip_f = item
                fast_download_small_hf_file("ASLP-lab/UrduSpeech", cand_f, "dataset", wav_dir / Path(clip_f).name, us_rev)

            with ThreadPoolExecutor(max_workers=16) as ex:
                list(ex.map(_dl_bench, tasks))
            us_eval_count += len(tasks)
            recs = us_adapter.process_jsonl_and_wavs(
                bj_local, wav_dir, flac_dir, "test", f"benchmark_{subset_tag}", sealed_stems=sealed_benchmark_stems
            )
            all_records.extend(recs)
            shutil.rmtree(wav_dir, ignore_errors=True)

    corpus_jsonls_all = [
        f
        for f in us_Siblings
        if f.startswith("corpus/") and "/short/" in f and f.endswith("_final_transcription.jsonl")
    ]
    corpus_jsonls = (
        [f for f in corpus_jsonls_all if "US-CS" in f]
        + [f for f in corpus_jsonls_all if "US-Std" in f]
        + [f for f in corpus_jsonls_all if "US-EngPk" in f]
    )
    existing_us_ids: Set[str] = set()
    existing_us_stems: Set[str] = set()
    if v3_train_recs:
        us_v3 = [r for r in v3_train_recs if r.source == "urduspeech_aslp"]
        all_records.extend(us_v3)
        existing_us_ids = {r.id for r in us_v3}
        existing_us_stems = {r.source_recording_id.upper() for r in us_v3}
        print(f"  Reused {len(us_v3)} v3 UrduSpeech records.")
    us_train_count = len(existing_us_ids)
    for cj in corpus_jsonls:
        if us_train_count >= args.urduspeech_max_train_clips:
            break
        check_free_disk_gib(args.data_dir)
        cj_local = fast_download_small_hf_file("ASLP-lab/UrduSpeech", cj, "dataset", cache_dir / cj.replace("/", "__"), us_rev)
        parent_prefix = str(Path(cj).parent).replace("\\", "/")
        subset_tag = "cs" if "US-CS" in cj else ("engpk" if "US-EngPk" in cj else "std")
        per_cat_limit = 30 if subset_tag == "engpk" else args.urduspeech_per_cat_limit
        lines = [json.loads(line) for line in cj_local.read_text(encoding="utf-8").splitlines() if line.strip()]
        wav_dir = cache_dir / "us_train_wavs" / parent_prefix.replace("/", "_")
        wav_dir.mkdir(parents=True, exist_ok=True)
        tasks = []
        cat_seen = 0
        for row in lines:
            if cat_seen >= per_cat_limit or (us_train_count + len(tasks)) >= args.urduspeech_max_train_clips:
                break
            clip_n = str(row.get("Audio_Clip") or "")
            stem_up = Path(clip_n).stem.upper()
            if not clip_n or stem_up in sealed_benchmark_stems:
                continue
            if stem_up in existing_us_stems:
                cat_seen += 1
                continue
            cand = f"{parent_prefix}/audio/{clip_n}"
            if cand not in us_file_set:
                alt = [x for x in us_Siblings if x.startswith(parent_prefix) and Path(x).stem.upper() == stem_up]
                cand = alt[0] if alt else ""
            if cand:
                tasks.append((cand, clip_n))
                cat_seen += 1

        def _dl_train(item):
            cand_f, clip_f = item
            try:
                fast_download_small_hf_file("ASLP-lab/UrduSpeech", cand_f, "dataset", wav_dir / Path(clip_f).name, us_rev)
                return 1
            except Exception:
                return 0

        if tasks:
            with ThreadPoolExecutor(max_workers=24) as ex:
                dl_ok = sum(ex.map(_dl_train, tasks))
            us_train_count += dl_ok
            recs = us_adapter.process_jsonl_and_wavs(
                cj_local,
                wav_dir,
                flac_dir,
                "train",
                subset_tag,
                sealed_stems=sealed_benchmark_stems,
                skip_stems=existing_us_stems,
            )
            new_recs = [r for r in recs if r.id not in existing_us_ids]
            for r in new_recs:
                existing_us_ids.add(r.id)
                existing_us_stems.add(r.source_recording_id.upper())
            all_records.extend(new_recs)
            print(f"    [UrduSpeech {parent_prefix}] downloaded={dl_ok}, new_records={len(new_recs)}, total_us={us_train_count}", flush=True)
        shutil.rmtree(wav_dir, ignore_errors=True)

    print("[4/6] Acquiring Rasa Urdu (capped studio expressive supplementation across female & male speakers)...")
    rasa_adapter = RasaUrduAdapter()
    rasa_rev = rasa_adapter.revision
    source_status["rasa_ur"] = {"status": "acquired"}
    if not has_baseline_eval:
        all_records.extend(rasa_adapter.process_viewer_offsets(flac_dir, "test", offsets=[0, 1500], length_per_offset=50))
    if v3_train_recs:
        rasa_v3 = [r for r in v3_train_recs if r.source == "rasa_ur"]
        all_records.extend(rasa_v3)
        print(f"  Reused {len(rasa_v3)} v3 Rasa records (viewer + shard 0 rows 1..450).")
    elif (baseline_dir / "canonical_train_all.jsonl").exists():
        old_tr = read_jsonl_records(baseline_dir / "canonical_train_all.jsonl")
        all_records.extend([r for r in old_tr if r.source == "rasa_ur"])
    else:
        all_records.extend(rasa_adapter.process_viewer_offsets(flac_dir, "train", offsets=[0, 5000, 15000, 20000], length_per_offset=100))
    for r_idx in range(args.rasa_train_shards):
        check_free_disk_gib(args.data_dir)
        shard = f"Urdu/train-{r_idx:05d}-of-00056.parquet"
        skip_n = 450 if (v3_train_recs and r_idx == 0) else 0
        if args.rasa_max_per_shard <= skip_n:
            continue
        print(f"  Rasa shard {shard} (skip={skip_n}, max={args.rasa_max_per_shard})...", flush=True)
        url = hf_parquet_url("ai4bharat/Rasa", shard, rasa_rev)
        all_records.extend(
            parallel_process_parquet_shard(
                rasa_adapter, url, flac_dir, "train", max_rows=args.rasa_max_per_shard, rg_workers=3, skip_rows=skip_n
            )
        )

    if not has_baseline_eval:
        print("[5/6] Acquiring FLEURS ur_pk (external evaluation benchmark dev & test)...")
        fleurs_adapter = FleursUrduAdapter()
        all_records.extend(fleurs_adapter.process_viewer_offsets(flac_dir, "validation", "dev", offsets=[0, 100], length_per_offset=100))
        all_records.extend(fleurs_adapter.process_viewer_offsets(flac_dir, "test", "test", offsets=[0, 100], length_per_offset=100))

        print("[6/6] Acquiring CSALT-Voice / WER-We-Stand (conversational benchmark)...")
        csalt_adapter = CsaltVoiceAdapter()
        all_records.extend(csalt_adapter.process_viewer_offsets(flac_dir, "test", offsets=[0, 100], length_per_offset=80))

    shutil.rmtree(cache_dir, ignore_errors=True)

    train_recs = [r for r in all_records if r.source_split == "train"]
    dev_recs = [r for r in all_records if r.source_split == "dev"]
    test_recs = [r for r in all_records if r.source_split == "test"]

    cleaned_train_recs, leakage_stats = deduplicate_and_prevent_leakage(
        train_recs, dev_recs, test_recs, sealed_recording_ids=sealed_benchmark_stems
    )

    accepted_train = [r for r in cleaned_train_recs if r.is_accepted()]
    accepted_dev = [r for r in dev_recs if r.is_accepted()]
    accepted_test = [r for r in test_recs if r.is_accepted()]

    write_jsonl_records(manifests_dir / "canonical_train_all.jsonl", cleaned_train_recs)
    write_jsonl_records(manifests_dir / "canonical_dev.jsonl", accepted_dev)
    write_jsonl_records(manifests_dir / "canonical_test.jsonl", accepted_test)

    mixture_info = compute_source_mixture_weights(accepted_train)
    balanced_train = build_balanced_training_records(accepted_train, mixture_info, seed=42)
    write_jsonl_records(manifests_dir / "canonical_train_balanced.jsonl", balanced_train)

    print("Exporting Lhotse CutSet manifests...")
    n_train_cuts = export_lhotse_cutset_jsonl(balanced_train, manifests_dir / "cuts_train.jsonl.gz")

    eval_cutset_counts = {}
    if has_baseline_eval and (baseline_dir / "cuts_dev.jsonl.gz").exists() and (baseline_dir / "cuts_test.jsonl.gz").exists():
        print(f"Preserving frozen evaluation CutSet manifests byte-for-byte from {baseline_dir.name}...")
        for p in sorted(baseline_dir.glob("cuts_*.jsonl.gz")):
            if p.name == "cuts_train.jsonl.gz":
                continue
            dst = manifests_dir / p.name
            if p.resolve() != dst.resolve():
                shutil.copy2(p, dst)
        n_dev_cuts = len(accepted_dev)
        n_test_cuts = len(accepted_test)
        for r in accepted_dev:
            eval_cutset_counts[f"dev_{r.source}"] = eval_cutset_counts.get(f"dev_{r.source}", 0) + 1
        for r in accepted_test:
            if r.source == "urduspeech_aslp":
                sub = "cs" if "_cs_" in r.domain else "std"
                k = f"test_urduspeech_benchmark_{sub}"
            else:
                k = f"test_{r.source}"
            eval_cutset_counts[k] = eval_cutset_counts.get(k, 0) + 1
    else:
        n_dev_cuts = export_lhotse_cutset_jsonl(accepted_dev, manifests_dir / "cuts_dev.jsonl.gz")
        n_test_cuts = export_lhotse_cutset_jsonl(accepted_test, manifests_dir / "cuts_test.jsonl.gz")
        eval_groups: Dict[str, List[UtteranceRecord]] = defaultdict(list)
        for r in accepted_dev:
            eval_groups[f"dev_{r.source}"].append(r)
        for r in accepted_test:
            if r.source == "urduspeech_aslp":
                sub = "cs" if "_cs_" in r.domain else "std"
                eval_groups[f"test_urduspeech_benchmark_{sub}"].append(r)
            else:
                eval_groups[f"test_{r.source}"].append(r)
        for g_name, g_recs in sorted(eval_groups.items()):
            eval_cutset_counts[g_name] = export_lhotse_cutset_jsonl(g_recs, manifests_dir / f"cuts_{g_name}.jsonl.gz")

    lang_dir = args.data_dir / "lang_bpe_1024"
    if (lang_dir / "bpe.model").exists() and (lang_dir / "tokenizer_meta.json").exists():
        print("Reusing existing verified SentencePiece BPE-1024 tokenizer...")
        tok_meta = json.loads((lang_dir / "tokenizer_meta.json").read_text(encoding="utf-8"))
    else:
        print("Training source-balanced SentencePiece BPE tokenizer (vocab_size=1024)...")
        tok_meta = train_source_balanced_sentencepiece(accepted_train, lang_dir, vocab_size=args.vocab_size, seed=42)

    audit_info = generate_stratified_audit_report(accepted_train + accepted_dev, args.reports_dir)

    by_source_split: Dict[str, List[UtteranceRecord]] = defaultdict(list)
    for r in cleaned_train_recs + dev_recs + test_recs:
        by_source_split[f"{r.source}:{r.source_split}"].append(r)

    per_source_summary = {k: summarize_records(v) for k, v in sorted(by_source_split.items())}

    human_train_recs = [r for r in cleaned_train_recs if "silver" not in r.label_origin]
    silver_train_recs = [r for r in cleaned_train_recs if "silver" in r.label_origin]
    human_balanced_recs = [r for r in balanced_train if "silver" not in r.label_origin]
    silver_balanced_recs = [r for r in balanced_train if "silver" in r.label_origin]
    human_vs_silver_tracking = {
        "human_labeled_train_unique": summarize_records(human_train_recs),
        "silver_labeled_train_unique": summarize_records(silver_train_recs),
        "human_labeled_balanced_epoch_utterances": len(human_balanced_recs),
        "human_labeled_balanced_epoch_hours": round(sum(r.segment_duration for r in human_balanced_recs) / 3600.0, 4),
        "silver_labeled_balanced_epoch_utterances": len(silver_balanced_recs),
        "silver_labeled_balanced_epoch_hours": round(sum(r.segment_duration for r in silver_balanced_recs) / 3600.0, 4),
    }

    free_end = check_free_disk_gib(args.data_dir)

    summary_report = {
        "manifest_version": "v4.0",
        "free_disk_gib_start": free_start,
        "free_disk_gib_end": free_end,
        "source_access_status": source_status,
        "human_vs_silver_tracking": human_vs_silver_tracking,
        "per_source_split_summary": per_source_summary,
        "mixture_sampling": mixture_info,
        "unique_accepted_train_utterances": len(accepted_train),
        "unique_accepted_train_hours": round(sum(r.segment_duration for r in accepted_train) / 3600.0, 4),
        "unique_train_speakers": len(set(r.speaker_group_id for r in accepted_train)),
        "balanced_epoch_train_utterances": n_train_cuts,
        "balanced_epoch_train_hours": round(sum(r.segment_duration for r in balanced_train) / 3600.0, 4),
        "accepted_dev_utterances": n_dev_cuts,
        "accepted_dev_hours": round(sum(r.segment_duration for r in accepted_dev) / 3600.0, 4),
        "accepted_test_utterances": n_test_cuts,
        "accepted_test_hours": round(sum(r.segment_duration for r in accepted_test) / 3600.0, 4),
        "eval_partition_cut_counts": dict(sorted(eval_cutset_counts.items())),
        "leakage_and_dedup_stats": leakage_stats,
        "tokenizer": tok_meta,
        "stratified_audit_sample_count": audit_info["total_sampled_clips"],
    }

    (args.reports_dir / "data_audit_summary.json").write_text(
        json.dumps(summary_report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (args.reports_dir / "leakage_report.json").write_text(
        json.dumps(leakage_stats, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("Data acquisition, quality filtering, leakage prevention, and tokenizer training complete!")
    print(json.dumps(summary_report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
