from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
import os
import tarfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pyarrow.parquet as pq
import soundfile as sf
import urllib3

from .normalizer import UrduNormalizer, classify_language_mix
from .quality import decode_and_standardize_audio, evaluate_utterance_quality
from .schema import UtteranceRecord

_HTTP_POOL = urllib3.PoolManager(
    num_pools=32,
    maxsize=32,
    retries=urllib3.Retry(total=8, backoff_factor=0.6),
    timeout=urllib3.Timeout(connect=15.0, read=60.0),
)


def get_hf_token() -> str:
    env_tok = os.environ.get("HF_TOKEN", "").strip()
    if env_tok:
        return env_tok
    tok_path = Path.home() / ".cache" / "huggingface" / "token"
    if tok_path.exists():
        return tok_path.read_text(encoding="utf-8").strip()
    return ""


def fetch_viewer_rows_with_audio(
    dataset: str,
    config: str,
    split: str,
    offsets: List[int],
    length_per_offset: int = 100,
    audio_col_name: str = "audio",
) -> List[Dict[str, Any]]:
    token = get_hf_token()
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    raw_rows: List[Dict[str, Any]] = []

    for off in offsets:
        cur_off = off
        rem = length_per_offset
        while rem > 0:
            step = min(100, rem)
            qs = urlencode({"dataset": dataset, "config": config, "split": split, "offset": cur_off, "length": step})
            url = f"https://datasets-server.huggingface.co/rows?{qs}"
            resp = _HTTP_POOL.request("GET", url, headers=headers)
            if resp.status >= 400:
                break
            payload = json.loads(resp.data.decode("utf-8"))
            batch = [item["row"] for item in payload.get("rows", []) if "row" in item]
            if not batch:
                break
            raw_rows.extend(batch)
            cur_off += len(batch)
            rem -= len(batch)
            if len(batch) < step:
                break

    def _download_row_audio(row: Dict[str, Any]) -> Dict[str, Any]:
        aval = row.get(audio_col_name)
        src_url = None
        orig_path = ""
        if isinstance(aval, list) and aval and isinstance(aval[0], dict):
            src_url = aval[0].get("src")
        elif isinstance(aval, dict):
            src_url = aval.get("src")
            orig_path = aval.get("path") or ""
        if src_url:
            try:
                r = _HTTP_POOL.request("GET", src_url)
                if r.status < 400 and r.data:
                    row[audio_col_name] = {"bytes": r.data, "path": orig_path or src_url.split("?")[0].split("/")[-1]}
            except Exception:
                pass
        return row

    with ThreadPoolExecutor(max_workers=24) as ex:
        enriched = list(ex.map(_download_row_audio, raw_rows))
    return enriched


class ParallelHttpRangeFile(io.RawIOBase):
    def __init__(self, url: str, token: str = "", workers: int = 16):
        self.url = url
        self.workers = workers
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        req = Request(url, headers={**headers, "Range": "bytes=0-0"})
        with urlopen(req, timeout=25) as r:
            self.final_url = r.geturl()
            cr = r.headers.get("Content-Range", "")
            if "/" in cr:
                self.size = int(cr.split("/")[-1])
            else:
                self.size = int(r.headers.get("Content-Length", "0"))
        self.pos = 0
        self._footer_cache_start = max(0, self.size - 512 * 1024)
        self._footer_bytes = self._fetch_range(self._footer_cache_start, self.size - 1)
        self._prefetch_start = -1
        self._prefetch_end = -1
        self._prefetch_bytes = b""
        self._small_start = -1
        self._small_end = -1
        self._small_bytes = b""

    def _fetch_range(self, start: int, end: int) -> bytes:
        if end < start:
            return b""
        chunk_size = 1024 * 1024
        spans = []
        cur = start
        while cur <= end:
            spans.append((cur, min(end, cur + chunk_size - 1)))
            cur += chunk_size

        def _get(span: Tuple[int, int]) -> bytes:
            s, e = span
            resp = _HTTP_POOL.request("GET", self.final_url, headers={"Range": f"bytes={s}-{e}"})
            return resp.data

        if len(spans) == 1:
            return _get(spans[0])
        with ThreadPoolExecutor(max_workers=min(self.workers, len(spans))) as ex:
            return b"".join(ex.map(_get, spans))

    def prefetch_row_group(self, pf: pq.ParquetFile, rg_idx: int, metadata_only: bool = False) -> None:
        if metadata_only:
            self._prefetch_start = -1
            self._prefetch_end = -1
            self._prefetch_bytes = b""
            return
        rg = pf.metadata.row_group(rg_idx)
        starts, ends = [], []
        for c_idx in range(rg.num_columns):
            col = rg.column(c_idx)
            if metadata_only and "audio" in col.path_in_schema.lower():
                continue
            offsets = [o for o in (col.dictionary_page_offset, col.data_page_offset, col.file_offset) if o and o > 0]
            s = min(offsets) if offsets else col.data_page_offset
            starts.append(s)
            ends.append(s + col.total_compressed_size + 16384)
        if not starts:
            return
        self._prefetch_start = 0 if (rg_idx == 0 and not metadata_only) else max(0, min(starts) - 8192)
        self._prefetch_end = min(self.size - 1, max(ends))
        self._prefetch_bytes = self._fetch_range(self._prefetch_start, self._prefetch_end)

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            self.pos = offset
        elif whence == io.SEEK_CUR:
            self.pos += offset
        elif whence == io.SEEK_END:
            self.pos = self.size + offset
        return self.pos

    def tell(self) -> int:
        return self.pos

    def readinto(self, b) -> int:
        n = len(b)
        if self.pos >= self.size or n == 0:
            return 0
        end = min(self.size - 1, self.pos + n - 1)
        if self.pos >= self._footer_cache_start:
            rel_s = self.pos - self._footer_cache_start
            rel_e = end - self._footer_cache_start + 1
            data = self._footer_bytes[rel_s:rel_e]
        elif self._prefetch_start <= self.pos and end <= self._prefetch_end:
            rel_s = self.pos - self._prefetch_start
            rel_e = end - self._prefetch_start + 1
            data = self._prefetch_bytes[rel_s:rel_e]
        else:
            data = self._fetch_range(self.pos, end)
        actual = len(data)
        b[:actual] = data
        self.pos += actual
        return actual


def parallel_download_hf_file(
    repo_id: str,
    filename: str,
    repo_type: str,
    out_path: Path,
    revision: str = "main",
    workers: int = 16,
) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and out_path.stat().st_size > 0:
        return out_path
    prefix = f"https://huggingface.co/datasets/{repo_id}" if repo_type == "dataset" else f"https://huggingface.co/{repo_id}"
    url = f"{prefix}/resolve/{revision}/{filename}"
    token = get_hf_token()
    headers = {"Authorization": f"Bearer {token}"} if token else {}

    req = Request(url, headers={**headers, "Range": "bytes=0-0"})
    with urlopen(req, timeout=25) as r:
        final_url = r.geturl()
        cr = r.headers.get("Content-Range", "")
        if "/" in cr:
            total_size = int(cr.split("/")[-1])
        else:
            resp_full = _HTTP_POOL.request("GET", url, headers=headers)
            out_path.write_bytes(resp_full.data)
            return out_path

    if total_size <= 2 * 1024 * 1024:
        resp_full = _HTTP_POOL.request("GET", final_url)
        out_path.write_bytes(resp_full.data)
        return out_path

    chunk_size = 2 * 1024 * 1024
    spans = []
    cur = 0
    while cur < total_size:
        spans.append((cur, min(total_size - 1, cur + chunk_size - 1)))
        cur += chunk_size

    def _get(span: Tuple[int, int]) -> bytes:
        s, e = span
        resp = _HTTP_POOL.request("GET", final_url, headers={"Range": f"bytes={s}-{e}"})
        return resp.data

    with ThreadPoolExecutor(max_workers=min(workers, len(spans))) as ex:
        parts = list(ex.map(_get, spans))
    out_path.write_bytes(b"".join(parts))
    return out_path


def fast_download_small_hf_file(
    repo_id: str,
    filename: str,
    repo_type: str,
    out_path: Path,
    revision: str = "main",
) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and out_path.stat().st_size > 0:
        return out_path
    prefix = f"https://huggingface.co/datasets/{repo_id}" if repo_type == "dataset" else f"https://huggingface.co/{repo_id}"
    url = f"{prefix}/resolve/{revision}/{filename}"
    token = get_hf_token()
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    resp = _HTTP_POOL.request("GET", url, headers=headers)
    if resp.status >= 400:
        raise RuntimeError(f"HTTP {resp.status} for {url}")
    out_path.write_bytes(resp.data)
    return out_path


def open_parquet_source(parquet_path_or_url: Path | str) -> Tuple[pq.ParquetFile, Optional[ParallelHttpRangeFile]]:
    if isinstance(parquet_path_or_url, str) and parquet_path_or_url.startswith("https://"):
        rf = ParallelHttpRangeFile(parquet_path_or_url, token=get_hf_token())
        return pq.ParquetFile(rf), rf
    return pq.ParquetFile(str(parquet_path_or_url)), None


def _extract_audio_bytes_and_name(audio_col: Any, fallback_name: str) -> Tuple[Optional[bytes], str]:
    if isinstance(audio_col, dict):
        b = audio_col.get("bytes")
        p = audio_col.get("path") or fallback_name
        return b, str(p)
    return None, fallback_name


def _save_flac_16k(waveform, sr: int, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not out_path.exists():
        tmp_path = out_path.with_name(f".{out_path.name}.{os.getpid()}.tmp")
        sf.write(str(tmp_path), waveform, sr, format="FLAC", subtype="PCM_16")
        os.replace(str(tmp_path), str(out_path))


def _speaker_split_is_test(speaker_id: str) -> bool:
    digest = hashlib.md5(speaker_id.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % 2 == 1


class CommonVoiceUrduAdapter:
    source_name = "common_voice_ur"
    MDC_DATASET_ID = "cmu5x1w3400cyo107st8cgsoq"
    EXPECTED_SHA256 = "12ad5f359fd94805db1bdaeb1441eb35720221f3c99229034069375fa246d990"

    def __init__(self, revision: str = EXPECTED_SHA256):
        self.revision = revision
        self.normalizer = UrduNormalizer()

    def process_extracted_dir(
        self,
        cv_root: Path,
        flac_out_dir: Path,
        split_name: str = "train",
        max_clips: Optional[int] = None,
    ) -> List[UtteranceRecord]:
        tsv_map = {"train": "train.tsv", "dev": "dev.tsv", "test": "test.tsv"}
        tsv_path = cv_root / tsv_map.get(split_name, f"{split_name}.tsv")
        clips_dir = cv_root / "clips"
        if not tsv_path.exists():
            return []
        records: List[UtteranceRecord] = []
        with tsv_path.open("r", encoding="utf-8") as f:
            header = f.readline().rstrip("\n").split("\t")
            for idx, line in enumerate(f):
                if max_clips is not None and idx >= max_clips:
                    break
                parts = line.rstrip("\n").split("\t")
                row = dict(zip(header, parts))
                clip_file = row.get("path", "")
                sentence = row.get("sentence", "")
                client_id = row.get("client_id", f"anon_{idx}")
                speaker_hash = hashlib.sha256(client_id.encode("utf-8")).hexdigest()[:16]
                audio_path = clips_dir / clip_file
                rec_id = Path(clip_file).stem
                utt_id = f"cv_ur_{split_name}_{rec_id}"
                wav, sr, dur, a_hash, d_flags, d_err = decode_and_standardize_audio(audio_path)
                t_train = self.normalizer.normalize(sentence)
                t_hash = self.normalizer.text_hash(sentence)
                lang_mix = classify_language_mix(sentence)
                if d_err is not None or wav is None:
                    continue
                q_flags, q_err = evaluate_utterance_quality(wav, dur, sentence, t_train, "human_validated")
                out_flac = flac_out_dir / self.source_name / split_name / f"{utt_id}.flac"
                if q_err is None:
                    _save_flac_16k(wav, sr, out_flac)
                records.append(
                    UtteranceRecord(
                        id=utt_id,
                        source=self.source_name,
                        source_revision=self.revision,
                        source_split=split_name,
                        source_recording_id=rec_id,
                        audio_path_or_reference=str(out_flac if q_err is None else audio_path),
                        segment_start=0.0,
                        segment_duration=round(dur, 4),
                        sample_rate=sr,
                        speaker_group_id=f"cv_{speaker_hash}",
                        session_group_id=f"cv_{speaker_hash}",
                        text_raw=sentence,
                        text_train=t_train,
                        label_origin="human_validated",
                        domain="read_prompt",
                        language_mix=lang_mix,
                        license_id="CC0-1.0-MDC-NoRehost-NoSpeakerID",
                        audio_hash=a_hash,
                        text_hash=t_hash,
                        quality_flags=d_flags + q_flags,
                        rejection_reason=q_err,
                    )
                )
        return records


class KathbathUrduAdapter:
    source_name = "kathbath_ur"

    def __init__(self, revision: str = "5b9e92849222026d9141acba4e8434fe816396bf"):
        self.revision = revision
        self.normalizer = UrduNormalizer()

    def process_rows(self, rows: List[Dict[str, Any]], flac_out_dir: Path, raw_split: str) -> List[UtteranceRecord]:
        records: List[UtteranceRecord] = []
        for idx, row in enumerate(rows, 1):
            fname = str(row.get("fname") or f"kb_{idx}")
            text_raw = str(row.get("text") or "")
            spk = str(row.get("speaker_id") or "unknown")
            if raw_split == "valid":
                eff_split = "test" if _speaker_split_is_test(f"kb_{spk}") else "dev"
            else:
                eff_split = "train"
            utt_id = f"kb_ur_{eff_split}_{Path(fname).stem}"
            out_flac = flac_out_dir / self.source_name / eff_split / f"{utt_id}.flac"
            t_train = self.normalizer.normalize(text_raw)
            t_hash = self.normalizer.text_hash(text_raw)
            lang_mix = classify_language_mix(text_raw)

            if out_flac.exists():
                wav, sr, dur, a_hash, d_flags, d_err = decode_and_standardize_audio(out_flac)
            else:
                audio_bytes, _ = _extract_audio_bytes_and_name(row.get("audio_filepath"), fname)
                if not audio_bytes:
                    continue
                wav, sr, dur, a_hash, d_flags, d_err = decode_and_standardize_audio(audio_bytes)

            if d_err is not None or wav is None:
                continue
            q_flags, q_err = evaluate_utterance_quality(wav, dur, text_raw, t_train, "human_gold")
            if q_err is None and not out_flac.exists():
                _save_flac_16k(wav, sr, out_flac)
            records.append(
                UtteranceRecord(
                    id=utt_id,
                    source=self.source_name,
                    source_revision=self.revision,
                    source_split=eff_split,
                    source_recording_id=Path(fname).stem,
                    audio_path_or_reference=str(out_flac if q_err is None else fname),
                    segment_start=0.0,
                    segment_duration=round(dur, 4),
                    sample_rate=sr,
                    speaker_group_id=f"kb_{spk}",
                    session_group_id=f"kb_{spk}",
                    text_raw=text_raw,
                    text_train=t_train,
                    label_origin="human_gold",
                    domain="read_speech",
                    language_mix=lang_mix,
                    license_id="CC-BY-4.0",
                    audio_hash=a_hash,
                    text_hash=t_hash,
                    quality_flags=d_flags + q_flags,
                    rejection_reason=q_err,
                )
            )
        return records

    def process_parquet(
        self,
        parquet_path_or_url: Path | str,
        flac_out_dir: Path,
        raw_split: str,
        max_rows: Optional[int] = None,
        metadata_only_if_cached: bool = True,
    ) -> List[UtteranceRecord]:
        pf, rf = open_parquet_source(parquet_path_or_url)
        records: List[UtteranceRecord] = []
        processed = 0
        meta_cols = [c for c in pf.schema.names if "audio" not in c.lower()]
        for rg_idx in range(pf.num_row_groups):
            if max_rows is not None and processed >= max_rows:
                break
            if rf is not None and metadata_only_if_cached:
                rf.prefetch_row_group(pf, rg_idx, metadata_only=True)
                meta_rows = pf.read_row_group(rg_idx, columns=meta_cols).to_pylist()
                cached_hits = 0
                for r in meta_rows:
                    fname = str(r.get("fname") or "")
                    spk = str(r.get("speaker_id") or "unknown")
                    eff = ("test" if _speaker_split_is_test(f"kb_{spk}") else "dev") if raw_split == "valid" else "train"
                    uid = f"kb_ur_{eff}_{Path(fname).stem}"
                    if (flac_out_dir / self.source_name / eff / f"{uid}.flac").exists():
                        cached_hits += 1
                if cached_hits >= int(0.75 * len(meta_rows)):
                    slice_rows = meta_rows[: (max_rows - processed)] if max_rows is not None else meta_rows
                    processed += len(slice_rows)
                    records.extend(self.process_rows(slice_rows, flac_out_dir, raw_split))
                    continue

            if rf is not None:
                rf.prefetch_row_group(pf, rg_idx, metadata_only=False)
            table = pf.read_row_group(rg_idx)
            rows = table.to_pylist()
            slice_rows = rows[: (max_rows - processed)] if max_rows is not None else rows
            processed += len(slice_rows)
            records.extend(self.process_rows(slice_rows, flac_out_dir, raw_split))
            if rg_idx % 5 == 0 or rg_idx == pf.num_row_groups - 1:
                print(f"    [Kathbath {raw_split}] rg={rg_idx+1}/{pf.num_row_groups} processed={processed} accepted={len(records)}", flush=True)
        return records


class IndicVoicesUrduAdapter:
    source_name = "indicvoices_ur"

    def __init__(self, revision: str = "c96f9088f138cf89d419da7e8e643e1f05c00a87"):
        self.revision = revision
        self.normalizer = UrduNormalizer()

    def process_rows(
        self,
        rows: List[Dict[str, Any]],
        flac_out_dir: Path,
        raw_split: str,
        shard_tag: str = "s0",
        start_idx: int = 0,
    ) -> List[UtteranceRecord]:
        records: List[UtteranceRecord] = []
        cached_by_suffix: Dict[Tuple[str, str], Path] = {}
        for sp_dir in ("dev", "test", "train"):
            d = flac_out_dir / self.source_name / sp_dir
            if d.exists():
                for p in d.glob(f"iv_ur_{sp_dir}_{shard_tag}_*.flac"):
                    idx_part = p.stem.rsplit("_", 1)[-1]
                    cached_by_suffix[(sp_dir, idx_part)] = p

        for offset, row in enumerate(rows, 1):
            processed = start_idx + offset
            text_raw = str(row.get("normalized") or row.get("text") or row.get("verbatim") or "")
            spk = str(row.get("speaker_id") or "unknown")
            scenario = str(row.get("scenario") or "spontaneous").lower()
            task_name = str(row.get("task_name") or "general").lower()
            domain = f"indicvoices_{scenario}"
            if raw_split == "valid":
                eff_split = "test" if _speaker_split_is_test(f"iv_{spk}") else "dev"
            else:
                eff_split = "train"

            cached_flac = cached_by_suffix.get((eff_split, str(processed)))
            if cached_flac is not None and cached_flac.exists():
                out_flac = cached_flac
                utt_id = out_flac.stem
                rec_stem = utt_id
                ref_name = out_flac.name
                wav, sr, dur, a_hash, d_flags, d_err = decode_and_standardize_audio(out_flac)
            else:
                audio_bytes, ref_name = _extract_audio_bytes_and_name(
                    row.get("audio_filepath"), f"{shard_tag}_{processed}"
                )
                rec_stem = Path(ref_name).stem if ref_name else f"{shard_tag}_{processed}"
                utt_id = f"iv_ur_{eff_split}_{shard_tag}_{rec_stem}_{processed}"
                out_flac = flac_out_dir / self.source_name / eff_split / f"{utt_id}.flac"
                if not audio_bytes:
                    continue
                wav, sr, dur, a_hash, d_flags, d_err = decode_and_standardize_audio(audio_bytes)

            t_train = self.normalizer.normalize(text_raw)
            t_hash = self.normalizer.text_hash(text_raw)
            lang_mix = classify_language_mix(text_raw)
            if d_err is not None or wav is None:
                continue
            q_flags, q_err = evaluate_utterance_quality(wav, dur, text_raw, t_train, "human_gold")
            if q_err is None and not out_flac.exists():
                _save_flac_16k(wav, sr, out_flac)
            records.append(
                UtteranceRecord(
                    id=utt_id,
                    source=self.source_name,
                    source_revision=self.revision,
                    source_split=eff_split,
                    source_recording_id=f"{shard_tag}_{rec_stem}_{processed}",
                    audio_path_or_reference=str(out_flac if q_err is None else ref_name),
                    segment_start=0.0,
                    segment_duration=round(dur, 4),
                    sample_rate=sr,
                    speaker_group_id=f"iv_{spk}",
                    session_group_id=f"iv_{spk}_{task_name}",
                    text_raw=text_raw,
                    text_train=t_train,
                    label_origin="human_gold",
                    domain=domain,
                    language_mix=lang_mix,
                    license_id="CC-BY-4.0",
                    audio_hash=a_hash,
                    text_hash=t_hash,
                    quality_flags=d_flags + q_flags,
                    rejection_reason=q_err,
                )
            )
        return records

    def process_viewer_offsets(
        self,
        flac_out_dir: Path,
        raw_split: str,
        offsets: List[int],
        length_per_offset: int = 100,
        shard_tag: str = "viewer",
    ) -> List[UtteranceRecord]:
        if raw_split == "valid":
            existing_val = len(list((flac_out_dir / self.source_name / "dev").glob("*.flac"))) + len(
                list((flac_out_dir / self.source_name / "test").glob("*.flac"))
            )
        else:
            existing_val = 0
        token = get_hf_token()
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        if existing_val >= 300 and raw_split == "valid":
            raw_rows: List[Dict[str, Any]] = []
            for off in offsets:
                qs = urlencode({"dataset": "ai4bharat/IndicVoices", "config": "urdu", "split": raw_split, "offset": off, "length": length_per_offset})
                resp = _HTTP_POOL.request("GET", f"https://datasets-server.huggingface.co/rows?{qs}", headers=headers)
                if resp.status < 400:
                    payload = json.loads(resp.data.decode("utf-8"))
                    for item in payload.get("rows", []):
                        r = item["row"]
                        aval = r.get("audio_filepath")
                        if isinstance(aval, list) and aval and isinstance(aval[0], dict):
                            src_u = aval[0].get("src", "")
                            r["audio_filepath"] = {"bytes": None, "path": src_u.split("?")[0].split("/")[-1]}
                        raw_rows.append(r)
            return self.process_rows(raw_rows, flac_out_dir, raw_split, shard_tag=shard_tag)

        rows = fetch_viewer_rows_with_audio(
            dataset="ai4bharat/IndicVoices",
            config="urdu",
            split=raw_split,
            offsets=offsets,
            length_per_offset=length_per_offset,
            audio_col_name="audio_filepath",
        )
        return self.process_rows(rows, flac_out_dir, raw_split, shard_tag=shard_tag)

    def process_parquet(
        self,
        parquet_path_or_url: Path | str,
        flac_out_dir: Path,
        raw_split: str,
        max_rows: Optional[int] = None,
        shard_tag: str = "s0",
        metadata_only_if_cached: bool = True,
    ) -> List[UtteranceRecord]:
        pf, rf = open_parquet_source(parquet_path_or_url)
        records: List[UtteranceRecord] = []
        processed = 0
        meta_cols = [c for c in pf.schema.names if "audio" not in c.lower()]
        existing_count = sum(
            len(list((flac_out_dir / self.source_name / sp).glob(f"iv_ur_{sp}_{shard_tag}_*.flac")))
            for sp in ("dev", "test", "train")
            if (flac_out_dir / self.source_name / sp).exists()
        )
        for rg_idx in range(pf.num_row_groups):
            if max_rows is not None and processed >= max_rows:
                break
            if rf is not None and metadata_only_if_cached:
                rf.prefetch_row_group(pf, rg_idx, metadata_only=True)
                meta_rows = pf.read_row_group(rg_idx, columns=meta_cols).to_pylist()
                cached_hits = 0
                for r_off, r in enumerate(meta_rows, 1):
                    spk = str(r.get("speaker_id") or "unknown")
                    eff = ("test" if _speaker_split_is_test(f"iv_{spk}") else "dev") if raw_split == "valid" else "train"
                    idx_s = str(processed + r_off)
                    if list((flac_out_dir / self.source_name / eff).glob(f"iv_ur_{eff}_{shard_tag}_*_{idx_s}.flac")):
                        cached_hits += 1
                if cached_hits >= int(0.75 * len(meta_rows)):
                    slice_rows = meta_rows[: (max_rows - processed)] if max_rows is not None else meta_rows
                    records.extend(self.process_rows(slice_rows, flac_out_dir, raw_split, shard_tag=shard_tag, start_idx=processed))
                    processed += len(slice_rows)
                    continue

            if rf is not None:
                rf.prefetch_row_group(pf, rg_idx, metadata_only=False)
            rows = pf.read_row_group(rg_idx).to_pylist()
            slice_rows = rows[: (max_rows - processed)] if max_rows is not None else rows
            records.extend(self.process_rows(slice_rows, flac_out_dir, raw_split, shard_tag=shard_tag, start_idx=processed))
            processed += len(slice_rows)
            print(f"    [IndicVoices {shard_tag}] rg={rg_idx+1}/{pf.num_row_groups} processed={processed} accepted={len(records)}", flush=True)
        return records


class RasaUrduAdapter:
    source_name = "rasa_ur"

    def __init__(self, revision: str = "632f55c7ac590219d41cd7adffce5b440e4604f5"):
        self.revision = revision
        self.normalizer = UrduNormalizer()

    def process_rows(self, rows: List[Dict[str, Any]], flac_out_dir: Path, raw_split: str) -> List[UtteranceRecord]:
        records: List[UtteranceRecord] = []
        eff_split = "test" if raw_split == "test" else "train"
        for idx, row in enumerate(rows, 1):
            fname = str(row.get("filename") or f"rasa_{idx}")
            text_raw = str(row.get("text") or "")
            gender = str(row.get("gender") or "unk").lower()
            style = str(row.get("style") or "neutral").lower()
            audio_bytes, ref_name = _extract_audio_bytes_and_name(row.get("audio"), fname)
            utt_id = f"rasa_ur_{eff_split}_{Path(fname).stem}"
            out_flac = flac_out_dir / self.source_name / eff_split / f"{utt_id}.flac"
            t_train = self.normalizer.normalize(text_raw)
            t_hash = self.normalizer.text_hash(text_raw)
            lang_mix = classify_language_mix(text_raw)
            if out_flac.exists():
                wav, sr, dur, a_hash, d_flags, d_err = decode_and_standardize_audio(out_flac)
            else:
                if not audio_bytes:
                    continue
                wav, sr, dur, a_hash, d_flags, d_err = decode_and_standardize_audio(audio_bytes)
            if d_err is not None or wav is None:
                continue
            q_flags, q_err = evaluate_utterance_quality(wav, dur, text_raw, t_train, "human_studio_expressive")
            if q_err is None and not out_flac.exists():
                _save_flac_16k(wav, sr, out_flac)
            records.append(
                UtteranceRecord(
                    id=utt_id,
                    source=self.source_name,
                    source_revision=self.revision,
                    source_split=eff_split,
                    source_recording_id=Path(fname).stem,
                    audio_path_or_reference=str(out_flac if q_err is None else ref_name),
                    segment_start=0.0,
                    segment_duration=round(dur, 4),
                    sample_rate=sr,
                    speaker_group_id=f"rasa_{gender}",
                    session_group_id=f"rasa_{gender}_{style}",
                    text_raw=text_raw,
                    text_train=t_train,
                    label_origin="human_studio_expressive",
                    domain=f"studio_expressive_{style}",
                    language_mix=lang_mix,
                    license_id="CC-BY-4.0",
                    audio_hash=a_hash,
                    text_hash=t_hash,
                    quality_flags=d_flags + q_flags,
                    rejection_reason=q_err,
                )
            )
        return records

    def process_viewer_offsets(
        self,
        flac_out_dir: Path,
        raw_split: str,
        offsets: List[int],
        length_per_offset: int = 80,
    ) -> List[UtteranceRecord]:
        rows = fetch_viewer_rows_with_audio(
            dataset="ai4bharat/Rasa",
            config="Urdu",
            split=raw_split,
            offsets=offsets,
            length_per_offset=length_per_offset,
            audio_col_name="audio",
        )
        return self.process_rows(rows, flac_out_dir, raw_split)

    def process_parquet(
        self,
        parquet_path_or_url: Path | str,
        flac_out_dir: Path,
        raw_split: str,
        max_rows: Optional[int] = None,
    ) -> List[UtteranceRecord]:
        pf, rf = open_parquet_source(parquet_path_or_url)
        records: List[UtteranceRecord] = []
        processed = 0
        for rg_idx in range(pf.num_row_groups):
            if max_rows is not None and processed >= max_rows:
                break
            if rf is not None:
                rf.prefetch_row_group(pf, rg_idx)
            table = pf.read_row_group(rg_idx)
            rows = table.to_pylist()
            slice_rows = rows[: (max_rows - processed)] if max_rows is not None else rows
            processed += len(slice_rows)
            records.extend(self.process_rows(slice_rows, flac_out_dir, raw_split))
        return records


class UrduSpeechASLPAdapter:
    source_name = "urduspeech_aslp"

    def __init__(self, revision: str = "16dd380cfd9049a3db7f06a98e878086916bf833"):
        self.revision = revision
        self.normalizer = UrduNormalizer()

    def process_jsonl_and_wavs(
        self,
        jsonl_path: Path,
        audio_dir: Path,
        flac_out_dir: Path,
        split_name: str,
        subset_tag: str,
        sealed_stems: Optional[Set[str]] = None,
        max_clips: Optional[int] = None,
        skip_stems: Optional[Set[str]] = None,
    ) -> List[UtteranceRecord]:
        records: List[UtteranceRecord] = []
        is_benchmark = split_name.startswith("test") or "benchmark" in str(jsonl_path).lower()
        label_origin = "human_gold_benchmark" if is_benchmark else "silver_model_validated"
        if not jsonl_path.exists():
            return records
        with jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                if max_clips is not None and len(records) >= max_clips:
                    break
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                clip_name = str(row.get("Audio_Clip") or "")
                text_raw = str(row.get("Transcription") or "")
                spk = str(row.get("Speaker_id") or "unk")
                category = str(row.get("Audio_category") or "general").lower()
                stem = Path(clip_name).stem
                if skip_stems and stem.upper() in skip_stems:
                    continue
                utt_id = f"us_{subset_tag}_{split_name}_{stem}"
                out_flac = flac_out_dir / self.source_name / split_name / f"{utt_id}.flac"
                wav_path = audio_dir / clip_name
                if out_flac.exists():
                    audio_src_for_decode = out_flac
                elif wav_path.exists():
                    audio_src_for_decode = wav_path
                else:
                    alt_candidates = list(audio_dir.glob(f"{stem}.*")) if audio_dir.exists() else []
                    if alt_candidates:
                        audio_src_for_decode = alt_candidates[0]
                    else:
                        continue
                t_train = self.normalizer.normalize(text_raw)
                t_hash = self.normalizer.text_hash(text_raw)
                lang_mix = classify_language_mix(text_raw)
                if not is_benchmark and sealed_stems and stem.upper() in sealed_stems:
                    records.append(
                        UtteranceRecord(
                            id=utt_id,
                            source=self.source_name,
                            source_revision=self.revision,
                            source_split=split_name,
                            source_recording_id=stem,
                            audio_path_or_reference=str(audio_src_for_decode),
                            segment_start=0.0,
                            segment_duration=0.0,
                            sample_rate=0,
                            speaker_group_id=f"us_{spk}_{category}",
                            session_group_id=f"us_{category}_{stem.rsplit('_', 1)[0]}",
                            text_raw=text_raw,
                            text_train=t_train,
                            label_origin=label_origin,
                            domain=f"urduspeech_{subset_tag}_{category}",
                            language_mix=lang_mix,
                            license_id="CC-BY-4.0",
                            audio_hash="",
                            text_hash=t_hash,
                            quality_flags=["blocked_benchmark_stem_overlap"],
                            rejection_reason="blocked_benchmark_stem_overlap",
                        )
                    )
                    continue
                wav, sr, dur, a_hash, d_flags, d_err = decode_and_standardize_audio(audio_src_for_decode)
                if d_err is not None or wav is None:
                    continue
                q_flags, q_err = evaluate_utterance_quality(wav, dur, text_raw, t_train, label_origin)
                if q_err is None and not out_flac.exists():
                    _save_flac_16k(wav, sr, out_flac)
                records.append(
                    UtteranceRecord(
                        id=utt_id,
                        source=self.source_name,
                        source_revision=self.revision,
                        source_split=split_name,
                        source_recording_id=stem,
                        audio_path_or_reference=str(out_flac if q_err is None else wav_path),
                        segment_start=0.0,
                        segment_duration=round(dur, 4),
                        sample_rate=sr,
                        speaker_group_id=f"us_{spk}_{category}",
                        session_group_id=f"us_{category}_{stem.rsplit('_', 1)[0]}",
                        text_raw=text_raw,
                        text_train=t_train,
                        label_origin=label_origin,
                        domain=f"urduspeech_{subset_tag}_{category}",
                        language_mix=lang_mix,
                        license_id="CC-BY-4.0",
                        audio_hash=a_hash,
                        text_hash=t_hash,
                        quality_flags=d_flags + q_flags,
                        rejection_reason=q_err,
                    )
                )
        return records


class FleursUrduAdapter:
    source_name = "fleurs_ur_pk"

    def __init__(self, revision: str = "70bb2e84b976b7e960aa89f1c648e09c59f894dd"):
        self.revision = revision
        self.normalizer = UrduNormalizer()

    def process_viewer_offsets(
        self,
        flac_out_dir: Path,
        viewer_split: str,
        eff_split: str,
        offsets: List[int],
        length_per_offset: int = 120,
    ) -> List[UtteranceRecord]:
        rows = fetch_viewer_rows_with_audio(
            dataset="google/fleurs",
            config="ur_pk",
            split=viewer_split,
            offsets=offsets,
            length_per_offset=length_per_offset,
            audio_col_name="audio",
        )
        records: List[UtteranceRecord] = []
        for idx, row in enumerate(rows, 1):
            prompt_id = str(row.get("id") or idx)
            text_raw = str(row.get("raw_transcription") or row.get("transcription") or "")
            audio_bytes, ref_name = _extract_audio_bytes_and_name(row.get("audio"), f"fleurs_{eff_split}_{idx}.wav")
            if not audio_bytes:
                continue
            utt_id = f"fleurs_ur_{eff_split}_{idx:04d}"
            out_flac = flac_out_dir / self.source_name / eff_split / f"{utt_id}.flac"
            t_train = self.normalizer.normalize(text_raw)
            t_hash = self.normalizer.text_hash(text_raw)
            lang_mix = classify_language_mix(text_raw)
            wav, sr, dur, a_hash, d_flags, d_err = decode_and_standardize_audio(audio_bytes)
            if d_err is not None or wav is None:
                continue
            q_flags, q_err = evaluate_utterance_quality(
                wav, dur, text_raw, t_train, "human_gold", min_duration=0.4, max_duration=35.0
            )
            if q_err is None:
                _save_flac_16k(wav, sr, out_flac)
            records.append(
                UtteranceRecord(
                    id=utt_id,
                    source=self.source_name,
                    source_revision=self.revision,
                    source_split=eff_split,
                    source_recording_id=f"fleurs_{prompt_id}_{idx}",
                    audio_path_or_reference=str(out_flac if q_err is None else ref_name),
                    segment_start=0.0,
                    segment_duration=round(dur, 4),
                    sample_rate=sr,
                    speaker_group_id=f"fleurs_spk_{idx}",
                    session_group_id=f"fleurs_sess_{idx}",
                    text_raw=text_raw,
                    text_train=t_train,
                    label_origin="human_gold",
                    domain="external_read_benchmark",
                    language_mix=lang_mix,
                    license_id="CC-BY-4.0",
                    audio_hash=a_hash,
                    text_hash=t_hash,
                    quality_flags=d_flags + q_flags,
                    rejection_reason=q_err,
                )
            )
        return records

    def process_tsv_and_tar(
        self,
        tsv_path: Path,
        tar_path: Path,
        flac_out_dir: Path,
        split_name: str,
    ) -> List[UtteranceRecord]:
        records: List[UtteranceRecord] = []
        audio_lookup: Dict[str, bytes] = {}
        with tarfile.open(str(tar_path), "r:gz") as tar:
            for member in tar.getmembers():
                if member.isfile() and member.name.endswith(".wav"):
                    fobj = tar.extractfile(member)
                    if fobj is not None:
                        audio_lookup[Path(member.name).name] = fobj.read()

        with tsv_path.open("r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 5:
                    continue
                prompt_id = parts[0]
                fname = parts[1]
                text_raw = parts[2]
                spk = parts[-2] if len(parts) >= 6 else f"spk_{idx}"
                audio_bytes = audio_lookup.get(fname)
                if not audio_bytes:
                    continue
                utt_id = f"fleurs_ur_{split_name}_{Path(fname).stem}"
                t_train = self.normalizer.normalize(text_raw)
                t_hash = self.normalizer.text_hash(text_raw)
                lang_mix = classify_language_mix(text_raw)
                wav, sr, dur, a_hash, d_flags, d_err = decode_and_standardize_audio(audio_bytes)
                if d_err is not None or wav is None:
                    continue
                q_flags, q_err = evaluate_utterance_quality(
                    wav, dur, text_raw, t_train, "human_gold", min_duration=0.4, max_duration=35.0
                )
                out_flac = flac_out_dir / self.source_name / split_name / f"{utt_id}.flac"
                if q_err is None:
                    _save_flac_16k(wav, sr, out_flac)
                records.append(
                    UtteranceRecord(
                        id=utt_id,
                        source=self.source_name,
                        source_revision=self.revision,
                        source_split=split_name,
                        source_recording_id=f"fleurs_{prompt_id}_{Path(fname).stem}",
                        audio_path_or_reference=str(out_flac if q_err is None else fname),
                        segment_start=0.0,
                        segment_duration=round(dur, 4),
                        sample_rate=sr,
                        speaker_group_id=f"fleurs_{spk}",
                        session_group_id=f"fleurs_{spk}",
                        text_raw=text_raw,
                        text_train=t_train,
                        label_origin="human_gold",
                        domain="external_read_benchmark",
                        language_mix=lang_mix,
                        license_id="CC-BY-4.0",
                        audio_hash=a_hash,
                        text_hash=t_hash,
                        quality_flags=d_flags + q_flags,
                        rejection_reason=q_err,
                    )
                )
        return records


class CsaltVoiceAdapter:
    source_name = "csalt_voice"

    def __init__(self, revision: str = "6f0c4fdeea94c77b4b64c4b88d5887e087566a5f"):
        self.revision = revision
        self.normalizer = UrduNormalizer()

    def process_viewer_offsets(
        self,
        flac_out_dir: Path,
        split_name: str = "test",
        offsets: List[int] = [0],
        length_per_offset: int = 120,
    ) -> List[UtteranceRecord]:
        rows = fetch_viewer_rows_with_audio(
            dataset="urdu-asr/csalt-voice",
            config="default",
            split="validation",
            offsets=offsets,
            length_per_offset=length_per_offset,
            audio_col_name="audio",
        )
        records: List[UtteranceRecord] = []
        for idx, row in enumerate(rows, 1):
            text_raw = str(row.get("transcription") or row.get("raw_transcription") or "")
            audio_bytes, ref_name = _extract_audio_bytes_and_name(row.get("audio"), f"csalt_{idx}.wav")
            if not audio_bytes:
                continue
            utt_id = f"csalt_ur_{split_name}_{idx:04d}"
            out_flac = flac_out_dir / self.source_name / split_name / f"{utt_id}.flac"
            t_train = self.normalizer.normalize(text_raw)
            t_hash = self.normalizer.text_hash(text_raw)
            lang_mix = classify_language_mix(text_raw)
            wav, sr, dur, a_hash, d_flags, d_err = decode_and_standardize_audio(audio_bytes)
            if d_err is not None or wav is None:
                continue
            q_flags, q_err = evaluate_utterance_quality(
                wav, dur, text_raw, t_train, "human_gold", min_duration=0.4, max_duration=35.0
            )
            if q_err is None:
                _save_flac_16k(wav, sr, out_flac)
            records.append(
                UtteranceRecord(
                    id=utt_id,
                    source=self.source_name,
                    source_revision=self.revision,
                    source_split=split_name,
                    source_recording_id=Path(ref_name).stem,
                    audio_path_or_reference=str(out_flac if q_err is None else ref_name),
                    segment_start=0.0,
                    segment_duration=round(dur, 4),
                    sample_rate=sr,
                    speaker_group_id=f"csalt_spk_{idx}",
                    session_group_id=f"csalt_sess_{idx}",
                    text_raw=text_raw,
                    text_train=t_train,
                    label_origin="human_gold",
                    domain="conversational_benchmark",
                    language_mix=lang_mix,
                    license_id="COLING-2025-Benchmark",
                    audio_hash=a_hash,
                    text_hash=t_hash,
                    quality_flags=d_flags + q_flags,
                    rejection_reason=q_err,
                )
            )
        return records
