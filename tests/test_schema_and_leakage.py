import io
import numpy as np
import soundfile as sf

from urdu_streaming_asr.quality import (
    decode_and_standardize_audio,
    deduplicate_and_prevent_leakage,
    evaluate_utterance_quality,
)
from urdu_streaming_asr.schema import UtteranceRecord


def _make_wav_bytes(sr: int = 16000, dur: float = 2.0, freq: float = 220.0, channels: int = 1) -> bytes:
    t = np.linspace(0, dur, int(sr * dur), endpoint=False, dtype=np.float32)
    sig = 0.25 * np.sin(2 * np.pi * freq * t)
    if channels == 2:
        sig = np.stack([sig, sig * 0.8], axis=1)
    buf = io.BytesIO()
    sf.write(buf, sig, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def test_decode_and_quality_checks():
    raw_bytes = _make_wav_bytes(sr=24000, dur=2.0, channels=2)
    wav, sr, dur, a_hash, flags, err = decode_and_standardize_audio(raw_bytes)
    assert err is None
    assert wav is not None
    assert sr == 16000
    assert abs(dur - 2.0) < 0.02
    assert "downmixed_stereo" in flags
    assert "resampled_from_24000" in flags
    q_flags, q_err = evaluate_utterance_quality(wav, dur, "یہ ایک اچھا جملہ ہے", "یہ ایک اچھا جملہ ہے", "human_gold")
    assert q_err is None


def test_deduplication_and_leakage_prevention():
    def rec(uid, split, rec_id, a_hash, text_train):
        return UtteranceRecord(
            id=uid,
            source="kathbath_ur",
            source_revision="rev1",
            source_split=split,
            source_recording_id=rec_id,
            audio_path_or_reference=f"{uid}.flac",
            segment_start=0.0,
            segment_duration=3.0,
            sample_rate=16000,
            speaker_group_id="spk1",
            session_group_id="sess1",
            text_raw=text_train,
            text_train=text_train,
            label_origin="human_gold",
            domain="read_speech",
            language_mix="ur_only",
            license_id="CC-BY-4.0",
            audio_hash=a_hash,
            text_hash=f"th_{text_train}",
        )

    dev = [rec("d1", "dev", "rec_dev_1", "hash_dev_1", "یہ ایک لمبا جملہ ہے جو ٹیسٹ میں موجود ہے")]
    test = [rec("t1", "test", "rec_test_1", "hash_test_1", "شکریہ")]
    train = [
        rec("tr_ok", "train", "rec_tr_1", "hash_tr_1", "یہ بالکل نیا ٹریننگ جملہ ہے"),
        rec("tr_dup_audio", "train", "rec_tr_2", "hash_tr_1", "دوسرا جملہ لیکن آواز وہی"),
        rec("tr_leak_audio", "train", "rec_tr_3", "hash_dev_1", "کوئی اور متن"),
        rec("tr_leak_text", "train", "rec_tr_4", "hash_tr_4", "یہ ایک لمبا جملہ ہے جو ٹیسٹ میں موجود ہے"),
        rec("tr_short_prompt", "train", "rec_tr_5", "hash_tr_5", "شکریہ"),
    ]

    cleaned, stats = deduplicate_and_prevent_leakage(train, dev, test)
    accepted_ids = [r.id for r in cleaned if r.is_accepted()]
    assert accepted_ids == ["tr_ok", "tr_short_prompt"]
    assert stats["exact_audio_hash_duplicates_removed"] == 1
    assert stats["sealed_eval_audio_hash_collisions_removed"] == 1
    assert stats["sealed_eval_long_text_collisions_removed"] == 1
    assert stats["identical_short_prompts_retained"] == 1
