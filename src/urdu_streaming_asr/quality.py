import hashlib
import io
import math
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import soundfile as sf

from .normalizer import classify_language_mix, normalize_urdu_text
from .schema import UtteranceRecord

TARGET_SAMPLE_RATE = 16000


def resample_linear_1d(waveform: np.ndarray, orig_sr: int, target_sr: int = TARGET_SAMPLE_RATE) -> np.ndarray:
    if orig_sr == target_sr:
        return waveform.astype(np.float32, copy=False)
    duration = len(waveform) / float(orig_sr)
    target_len = max(1, int(round(duration * target_sr)))
    x_old = np.linspace(0.0, 1.0, num=len(waveform), endpoint=False, dtype=np.float64)
    x_new = np.linspace(0.0, 1.0, num=target_len, endpoint=False, dtype=np.float64)
    return np.interp(x_new, x_old, waveform).astype(np.float32)


def decode_and_standardize_audio(
    audio_bytes_or_path: bytes | str | Path,
    segment_start: float = 0.0,
    segment_duration: Optional[float] = None,
) -> Tuple[Optional[np.ndarray], int, float, str, List[str], Optional[str]]:
    flags: List[str] = []
    try:
        if isinstance(audio_bytes_or_path, (bytes, bytearray)):
            data, sr = sf.read(io.BytesIO(audio_bytes_or_path), dtype="float32", always_2d=True)
        else:
            data, sr = sf.read(str(audio_bytes_or_path), dtype="float32", always_2d=True)
    except Exception as exc:
        return None, 0, 0.0, "", ["decode_error"], f"decode_error:{type(exc).__name__}"

    if data.size == 0 or sr <= 0:
        return None, 0, 0.0, "", ["empty_audio"], "empty_audio"

    if not np.isfinite(data).all():
        return None, sr, 0.0, "", ["non_finite_samples"], "non_finite_samples"

    if data.shape[1] > 1:
        flags.append("downmixed_stereo")
        mono = data.mean(axis=1, dtype=np.float32)
    else:
        mono = data[:, 0]

    total_samples = len(mono)
    if segment_start < 0.0:
        return None, sr, 0.0, "", ["negative_offset"], "negative_offset"

    if segment_start > 0.0 or segment_duration is not None:
        start_idx = int(round(segment_start * sr))
        if start_idx >= total_samples:
            return None, sr, 0.0, "", ["offset_out_of_bounds"], "offset_out_of_bounds"
        if segment_duration is not None:
            if segment_duration <= 0:
                return None, sr, 0.0, "", ["non_positive_duration"], "non_positive_duration"
            end_idx = min(total_samples, start_idx + int(round(segment_duration * sr)))
        else:
            end_idx = total_samples
        mono = mono[start_idx:end_idx]

    if len(mono) == 0:
        return None, sr, 0.0, "", ["empty_segment"], "empty_segment"

    if sr != TARGET_SAMPLE_RATE:
        flags.append(f"resampled_from_{sr}")
        mono = resample_linear_1d(mono, sr, TARGET_SAMPLE_RATE)
        sr = TARGET_SAMPLE_RATE

    duration = len(mono) / float(sr)
    pcm16 = np.clip(np.round(mono * 32767.0), -32768, 32767).astype(np.int16)
    audio_hash = hashlib.sha256(pcm16.tobytes()).hexdigest()
    return mono, sr, duration, audio_hash, flags, None


def evaluate_utterance_quality(
    waveform: np.ndarray,
    duration: float,
    text_raw: str,
    text_train: str,
    label_origin: str,
    min_duration: float = 0.6,
    max_duration: float = 25.0,
    min_rms: float = 0.001,
    max_clipping_ratio: float = 0.05,
    min_chars_per_sec: float = 1.5,
    max_chars_per_sec: float = 32.0,
    min_duration_short_reply: float = 0.35,
) -> Tuple[List[str], Optional[str]]:
    flags: List[str] = []
    if not text_raw or not text_raw.strip():
        return ["empty_raw_text"], "empty_raw_text"
    if not text_train or not text_train.strip():
        return ["empty_normalized_text"], "empty_normalized_text"

    rms = float(np.sqrt(np.mean(np.square(waveform, dtype=np.float64))))
    if not math.isfinite(duration) or duration < min_duration_short_reply:
        return ["duration_too_short"], "duration_too_short"
    if duration < min_duration:
        words = text_train.split()
        if 1 <= len(words) <= 4 and rms >= max(min_rms, 0.002):
            flags.append("short_reply_0.35_to_0.60s_retained")
        else:
            return ["duration_too_short"], "duration_too_short"
    if duration > max_duration:
        return ["duration_too_long"], "duration_too_long"

    if rms < min_rms:
        return ["near_silent_audio"], "near_silent_audio"

    clip_ratio = float(np.mean(np.abs(waveform) >= 0.995))
    if clip_ratio > max_clipping_ratio:
        return ["severe_clipping"], "severe_clipping"
    elif clip_ratio > 0.01:
        flags.append("minor_clipping")

    frame_len = 320
    if len(waveform) >= frame_len:
        n_frames = len(waveform) // frame_len
        frames = waveform[: n_frames * frame_len].reshape(n_frames, frame_len)
        frame_rms = np.sqrt(np.mean(np.square(frames), axis=1))
        silent_ratio = float(np.mean(frame_rms < 0.002))
        if silent_ratio > 0.85:
            return ["excessive_silence"], "excessive_silence"
        elif silent_ratio > 0.65:
            flags.append("high_silence_ratio")

    non_space_chars = len(text_train.replace(" ", ""))
    cps = non_space_chars / duration
    if cps < min_chars_per_sec:
        return ["cps_too_low"], "cps_too_low"
    if cps > max_chars_per_sec:
        return ["cps_too_high"], "cps_too_high"

    if "silver" in label_origin:
        words = text_train.split()
        if len(words) >= 6:
            unique_ratio = len(set(words)) / float(len(words))
            if unique_ratio < 0.35:
                return ["silver_repetitive_hallucination"], "silver_repetitive_hallucination"
        if cps < 2.2 or cps > 26.0:
            return ["silver_implausible_cps"], "silver_implausible_cps"

    return flags, None


def deduplicate_and_prevent_leakage(
    train_records: List[UtteranceRecord],
    dev_records: List[UtteranceRecord],
    test_records: List[UtteranceRecord],
    sealed_recording_ids: Optional[Set[str]] = None,
) -> Tuple[List[UtteranceRecord], Dict[str, int]]:
    stats = {
        "exact_audio_hash_duplicates_removed": 0,
        "sealed_eval_audio_hash_collisions_removed": 0,
        "sealed_eval_recording_id_collisions_removed": 0,
        "sealed_eval_long_text_collisions_removed": 0,
        "identical_short_prompts_retained": 0,
    }

    eval_audio_hashes: Set[str] = set()
    eval_rec_ids: Set[str] = set(sealed_recording_ids or set())
    eval_long_text_hashes: Set[str] = set()
    eval_short_text_hashes: Set[str] = set()

    for rec in dev_records + test_records:
        if not rec.is_accepted():
            continue
        eval_audio_hashes.add(rec.audio_hash)
        eval_rec_ids.add(rec.source_recording_id)
        word_count = len(rec.text_train.split())
        if word_count >= 5:
            eval_long_text_hashes.add(rec.text_hash)
        else:
            eval_short_text_hashes.add(rec.text_hash)

    seen_train_audio_hashes: Set[str] = set()
    cleaned_train: List[UtteranceRecord] = []

    for rec in train_records:
        if not rec.is_accepted():
            cleaned_train.append(rec)
            continue
        if rec.audio_hash in eval_audio_hashes:
            rec.rejection_reason = "leakage_eval_audio_hash"
            rec.quality_flags.append("leakage_eval_audio_hash")
            stats["sealed_eval_audio_hash_collisions_removed"] += 1
            cleaned_train.append(rec)
            continue
        if rec.source_recording_id in eval_rec_ids:
            rec.rejection_reason = "leakage_eval_recording_id"
            rec.quality_flags.append("leakage_eval_recording_id")
            stats["sealed_eval_recording_id_collisions_removed"] += 1
            cleaned_train.append(rec)
            continue
        if rec.audio_hash in seen_train_audio_hashes:
            rec.rejection_reason = "duplicate_train_audio_hash"
            rec.quality_flags.append("duplicate_train_audio_hash")
            stats["exact_audio_hash_duplicates_removed"] += 1
            cleaned_train.append(rec)
            continue
        if rec.text_hash in eval_long_text_hashes:
            rec.rejection_reason = "leakage_eval_long_transcript"
            rec.quality_flags.append("leakage_eval_long_transcript")
            stats["sealed_eval_long_text_collisions_removed"] += 1
            cleaned_train.append(rec)
            continue
        if rec.text_hash in eval_short_text_hashes:
            rec.quality_flags.append("shared_short_prompt_with_eval")
            stats["identical_short_prompts_retained"] += 1

        seen_train_audio_hashes.add(rec.audio_hash)
        cleaned_train.append(rec)

    return cleaned_train, stats
