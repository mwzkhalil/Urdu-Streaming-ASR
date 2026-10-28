from collections import defaultdict
import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

import sentencepiece as spm

from .schema import UtteranceRecord


def compute_source_mixture_weights(
    accepted_train_records: List[UtteranceRecord],
    temperature: float = 0.7,
    max_single_read_speech_share: float = 0.40,
    max_rasa_share: float = 0.10,
    max_silver_share: float = 0.20,
    max_english_only_share: float = 0.05,
) -> Dict[str, Any]:
    hours_by_source: Dict[str, float] = defaultdict(float)
    for rec in accepted_train_records:
        if rec.is_accepted():
            hours_by_source[rec.source] += rec.segment_duration / 3600.0

    if not hours_by_source:
        return {"hours_by_source": {}, "weights": {}, "multipliers": {}}

    raw_weights = {src: (hrs ** temperature) for src, hrs in hours_by_source.items() if hrs > 0}
    total_raw = sum(raw_weights.values())
    weights = {src: w / total_raw for src, w in raw_weights.items()}

    num_sources = len(weights)
    if num_sources >= 2:
        for _ in range(10):
            excess = 0.0
            unclamped = []
            for src, w in list(weights.items()):
                cap = 1.0
                if src in ("kathbath_ur", "common_voice_ur") and num_sources >= 3:
                    cap = min(cap, max_single_read_speech_share)
                if src == "rasa_ur":
                    cap = min(cap, max_rasa_share)
                if src == "urduspeech_aslp":
                    cap = min(cap, max_silver_share)
                if w > cap:
                    excess += w - cap
                    weights[src] = cap
                else:
                    unclamped.append(src)
            if excess <= 1e-7 or not unclamped:
                break
            unclamped_sum = sum(weights[s] for s in unclamped)
            if unclamped_sum > 0:
                for s in unclamped:
                    weights[s] += excess * (weights[s] / unclamped_sum)

    total_w = sum(weights.values())
    if total_w > 0:
        weights = {s: w / total_w for s, w in weights.items()}

    total_hours = sum(hours_by_source.values())
    natural_shares = {s: h / total_hours for s, h in hours_by_source.items()}
    multipliers = {
        s: round(weights[s] / natural_shares[s], 4) if natural_shares[s] > 0 else 1.0
        for s in weights
    }

    return {
        "hours_by_source": {s: round(h, 4) for s, h in hours_by_source.items()},
        "natural_shares": {s: round(p, 4) for s, p in natural_shares.items()},
        "target_weights": {s: round(w, 4) for s, w in weights.items()},
        "sampling_multipliers": multipliers,
        "max_english_only_share": max_english_only_share,
    }


def build_balanced_training_records(
    accepted_train_records: List[UtteranceRecord],
    mixture_info: Dict[str, Any],
    seed: int = 42,
    max_single_speaker_share: float = 0.005,
) -> List[UtteranceRecord]:
    rng = random.Random(seed)
    by_source: Dict[str, List[UtteranceRecord]] = defaultdict(list)
    en_only_records: List[UtteranceRecord] = []

    for rec in accepted_train_records:
        if not rec.is_accepted():
            continue
        if rec.language_mix == "en_only":
            en_only_records.append(rec)
        else:
            by_source[rec.source].append(rec)

    target_weights = mixture_info.get("target_weights", {})
    hours_by_source = mixture_info.get("hours_by_source", {})
    total_base_hours = sum(hours_by_source.values())
    if total_base_hours <= 0:
        return []

    max_spk_sec = max(120.0, total_base_hours * 3600.0 * max_single_speaker_share)
    spk_used_sec: Dict[str, float] = defaultdict(float)

    sampled: List[UtteranceRecord] = []
    for src, recs in by_source.items():
        if not recs:
            continue
        # Prioritize code-switching, conversational, and 1-2 word short replies when sampling fractional copies
        rng.shuffle(recs)
        recs.sort(
            key=lambda r: (
                0 if (r.language_mix == "ur_en_codeswitch" or "_cs_" in r.domain or len(r.text_train.split()) <= 2) else 1
            )
        )
        target_w = target_weights.get(src, 1.0 / max(1, len(by_source)))
        target_hrs = total_base_hours * target_w
        src_hrs = sum(r.segment_duration for r in recs) / 3600.0
        if src_hrs <= 0:
            continue
        ratio = target_hrs / src_hrs
        full_copies = int(math.floor(ratio))
        frac = ratio - full_copies
        candidates = []
        for _ in range(max(0, full_copies)):
            candidates.extend(recs)
        if frac > 0:
            k = max(1, int(round(frac * len(recs))))
            candidates.extend(recs[:k])

        for r in candidates:
            spk = r.speaker_group_id or "unk"
            if spk != "unk" and spk_used_sec[spk] + r.segment_duration > max_spk_sec:
                continue
            spk_used_sec[spk] += r.segment_duration
            sampled.append(r)

    if en_only_records:
        rng.shuffle(en_only_records)
        current_hours = sum(r.segment_duration for r in sampled) / 3600.0
        max_en_hours = current_hours * mixture_info.get("max_english_only_share", 0.05)
        acc_en = 0.0
        for rec in en_only_records:
            if acc_en + (rec.segment_duration / 3600.0) <= max_en_hours:
                sampled.append(rec)
                acc_en += rec.segment_duration / 3600.0

    rng.shuffle(sampled)
    return sampled


def summarize_records_distribution(records: List[UtteranceRecord], is_sampled_stage: bool = False) -> Dict[str, Any]:
    total_sec = sum(r.segment_duration for r in records)
    total_hrs = total_sec / 3600.0
    unique_ids = {}
    for r in records:
        if r.id not in unique_ids:
            unique_ids[r.id] = r
    unique_hrs = sum(r.segment_duration for r in unique_ids.values()) / 3600.0
    repeated_hrs = max(0.0, total_hrs - unique_hrs)

    by_source_hrs: Dict[str, float] = defaultdict(float)
    by_origin_hrs: Dict[str, float] = defaultdict(float)
    by_lang_hrs: Dict[str, float] = defaultdict(float)
    by_domain_hrs: Dict[str, float] = defaultdict(float)
    by_spk_hrs: Dict[str, float] = defaultdict(float)
    short_reply_hrs = 0.0
    short_reply_utts = 0
    pseudo_labeled_sampled_hrs = 0.0

    for r in records:
        h = r.segment_duration / 3600.0
        by_source_hrs[r.source] += h
        by_origin_hrs[r.label_origin] += h
        by_lang_hrs[r.language_mix] += h
        by_domain_hrs[r.domain] += h
        by_spk_hrs[r.speaker_group_id or "unk"] += h
        if len(r.text_train.split()) <= 3:
            short_reply_hrs += h
            short_reply_utts += 1
        if "pseudo_labeled" in r.label_origin:
            pseudo_labeled_sampled_hrs += h

    pseudo_unique_hrs = sum(
        r.segment_duration / 3600.0 for r in unique_ids.values() if "pseudo_labeled" in r.label_origin
    )
    denom = max(1e-6, total_hrs)
    max_spk_h = max(by_spk_hrs.values()) if by_spk_hrs else 0.0

    return {
        "total_utterances": len(records),
        "unique_utterances": len(unique_ids),
        "total_hours": round(total_hrs, 4),
        "unique_accepted_hours": round(unique_hrs, 4),
        "repeated_sampling_hours": round(repeated_hrs, 4),
        "pseudo_labeled_unique_hours": round(pseudo_unique_hrs, 4),
        "pseudo_labeled_sampled_hours": round(pseudo_labeled_sampled_hrs, 4),
        "online_augmentation_new_corpus_hours_counted": 0.0,
        "short_replies_le_3_words_utterances": short_reply_utts,
        "short_replies_le_3_words_hours": round(short_reply_hrs, 4),
        "short_replies_share_pct": round(100.0 * short_reply_hrs / denom, 2),
        "distinct_speaker_groups": len(by_spk_hrs),
        "max_single_speaker_share_pct": round(100.0 * max_spk_h / denom, 3),
        "hours_by_source": {k: round(v, 4) for k, v in sorted(by_source_hrs.items())},
        "share_pct_by_source": {k: round(100.0 * v / denom, 2) for k, v in sorted(by_source_hrs.items())},
        "hours_by_label_origin": {k: round(v, 4) for k, v in sorted(by_origin_hrs.items())},
        "share_pct_by_label_origin": {k: round(100.0 * v / denom, 2) for k, v in sorted(by_origin_hrs.items())},
        "hours_by_language_mix": {k: round(v, 4) for k, v in sorted(by_lang_hrs.items())},
        "share_pct_by_language_mix": {k: round(100.0 * v / denom, 2) for k, v in sorted(by_lang_hrs.items())},
        "hours_by_domain_top20": {
            k: round(v, 4) for k, v in sorted(by_domain_hrs.items(), key=lambda x: -x[1])[:20]
        },
    }


def train_source_balanced_sentencepiece(
    train_records: List[UtteranceRecord],
    output_dir: Path,
    vocab_size: int = 1024,
    seed: int = 42,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    by_source: Dict[str, List[str]] = defaultdict(list)
    for rec in train_records:
        if rec.is_accepted() and rec.source_split == "train" and rec.text_train:
            by_source[rec.source].append(rec.text_train)

    if not by_source:
        raise ValueError("No accepted training transcripts available for SentencePiece training.")

    max_count = max(len(v) for v in by_source.values())
    balanced_lines: List[str] = []
    for src, lines in sorted(by_source.items()):
        shuffled = list(lines)
        rng.shuffle(shuffled)
        target_n = min(max_count, max(len(shuffled), int(round(max_count * 0.65))))
        reps = (target_n // len(shuffled)) + 1
        balanced_lines.extend((shuffled * reps)[:target_n])

    rng.shuffle(balanced_lines)
    corpus_file = output_dir / "spm_train_corpus.txt"
    corpus_text = "\n".join(balanced_lines) + "\n"
    corpus_file.write_text(corpus_text, encoding="utf-8")
    corpus_hash = hashlib.sha256(corpus_text.encode("utf-8")).hexdigest()

    essential_urdu_chars = list("ابپتٹثجچحخدڈذرڑزژسشصضطظعغفقکگلمنںوہھءیئےؤآ")
    essential_latin_digits = list("abcdefghijklmnopqrstuvwxyz0123456789")
    corpus_char_set = set(corpus_text)
    required_chars = "".join(
        ch for ch in dict.fromkeys(essential_urdu_chars + essential_latin_digits) if ch in corpus_char_set
    )

    model_prefix = str(output_dir / "bpe")
    spm.SentencePieceTrainer.train(
        input=str(corpus_file),
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type="bpe",
        character_coverage=1.0,
        required_chars=required_chars,
        bos_id=1,
        eos_id=2,
        unk_id=3,
        pad_id=-1,
        bos_piece="<sos/eos>",
        eos_piece="</s>",
        unk_piece="<unk>",
        user_defined_symbols=["<blk>"],
        train_extremely_large_corpus=False,
    )

    sp = spm.SentencePieceProcessor()
    sp.load(f"{model_prefix}.model")

    tokens_path = output_dir / "tokens.txt"
    with tokens_path.open("w", encoding="utf-8") as f:
        for idx in range(sp.get_piece_size()):
            piece = sp.id_to_piece(idx)
            f.write(f"{piece} {idx}\n")

    bpe_sha256 = hashlib.sha256((output_dir / "bpe.model").read_bytes()).hexdigest()
    tokens_sha256 = hashlib.sha256(tokens_path.read_bytes()).hexdigest()

    meta = {
        "vocab_size": sp.get_piece_size(),
        "blank_id": sp.piece_to_id("<blk>"),
        "sos_eos_id": sp.piece_to_id("<sos/eos>"),
        "unk_id": sp.unk_id(),
        "training_corpus_lines": len(balanced_lines),
        "training_corpus_sha256": corpus_hash,
        "bpe_model_sha256": bpe_sha256,
        "tokens_txt_sha256": tokens_sha256,
        "sources_included": {k: len(v) for k, v in by_source.items()},
    }
    (output_dir / "tokenizer_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return meta


def verify_flac_integrity(audio_path: Path) -> Tuple[bool, int, int]:
    """Verify FLAC header AND decode first/last 4096 frames to catch truncated or corrupted streams."""
    import soundfile as sf

    try:
        with sf.SoundFile(str(audio_path)) as f:
            num_samples = int(f.frames)
            sr = int(f.samplerate)
            if num_samples <= 0 or sr <= 0:
                return False, 0, 0
            head_n = min(2048, num_samples)
            head = f.read(head_n, dtype="float32")
            if len(head) != head_n:
                return False, 0, 0
            if num_samples > 2048:
                seek_pos = max(0, num_samples - 4096)
                f.seek(seek_pos)
                tail_n = num_samples - seek_pos
                tail = f.read(tail_n, dtype="float32")
                if len(tail) != tail_n:
                    return False, 0, 0
            return True, num_samples, sr
    except Exception:
        return False, 0, 0


def export_lhotse_cutset_jsonl(records: List[UtteranceRecord], out_cuts_jsonl_gz: Path) -> int:
    from lhotse import MonoCut, Recording, SupervisionSegment, CutSet
    from lhotse.audio import AudioSource

    project_root = out_cuts_jsonl_gz.resolve().parents[2]
    verified_cache: Dict[str, Tuple[bool, int, int]] = {}
    cuts = []
    for idx, rec in enumerate(records):
        if not rec.is_accepted():
            continue
        audio_path = Path(rec.audio_path_or_reference)
        if not audio_path.is_absolute():
            audio_path = (project_root / audio_path).resolve()
        ap_str = str(audio_path)
        if ap_str not in verified_cache:
            if not audio_path.exists():
                verified_cache[ap_str] = (False, 0, 0)
            else:
                verified_cache[ap_str] = verify_flac_integrity(audio_path)
        ok, num_samples, sr = verified_cache[ap_str]
        if not ok:
            continue
        dur = round(num_samples / float(sr), 4)
        recording = Recording(
            id=f"{rec.id}_{idx}",
            sources=[AudioSource(type="file", channels=[0], source=str(audio_path))],
            sampling_rate=sr,
            num_samples=num_samples,
            duration=dur,
        )
        sup = SupervisionSegment(
            id=f"{rec.id}_{idx}",
            recording_id=recording.id,
            start=0.0,
            duration=dur,
            channel=0,
            text=rec.text_train,
            language=rec.language_mix,
            speaker=rec.speaker_group_id,
            custom={
                "source": rec.source,
                "domain": rec.domain,
                "text_raw": rec.text_raw,
                "label_origin": rec.label_origin,
                "audio_hash": rec.audio_hash,
            },
        )
        cut = MonoCut(
            id=f"{rec.id}_{idx}",
            start=0.0,
            duration=dur,
            channel=0,
            recording=recording,
            supervisions=[sup],
        )
        cuts.append(cut)

    out_cuts_jsonl_gz.parent.mkdir(parents=True, exist_ok=True)
    cut_set = CutSet.from_cuts(cuts)
    cut_set.to_file(str(out_cuts_jsonl_gz))
    return len(cuts)
