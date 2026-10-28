from urdu_streaming_asr.normalizer import UrduNormalizer, classify_language_mix, normalize_urdu_text


def test_urdu_letter_preservation():
    norm = UrduNormalizer()
    sample = "گھر میں ٹھنڈا پانی، لڑکا، ڈبہ، الفاظ، آئینہ، نئی، مؤثر اور بیٹے ہیں"
    out = norm.normalize(sample)
    assert "گھر" in out
    assert "ٹھنڈا" in out
    assert "لڑکا" in out
    assert "ڈبہ" in out
    assert "آئینہ" in out
    assert "نئی" in out
    assert "مؤثر" in out
    assert "بیٹے" in out
    assert "ہیں" in out


def test_arabic_to_urdu_canonical_and_digits():
    raw = "كہانی يہ ہے كہ ٢٠٢٦ اور ۱۴ اگست کو 5 بجے!"
    out = normalize_urdu_text(raw)
    assert out == "کہانی یہ ہے کہ 2026 اور 14 اگست کو 5 بجے"


def test_code_switching_preserved():
    raw = "میرا account بازیاب کریں اور online meeting link بھیجیں۔"
    out = normalize_urdu_text(raw)
    assert out == "میرا account بازیاب کریں اور online meeting link بھیجیں"
    assert classify_language_mix(out) == "ur_en_codeswitch"


def test_bidi_and_determinism():
    raw = "\u200fیہ\u200c ایک\ufeff ٹیسٹ\u0640 ہے؟"
    norm = UrduNormalizer()
    assert norm.normalize(raw) == "یہ ایک ٹیسٹ ہے"
    assert norm.text_hash(raw) == norm.text_hash("یہ ایک ٹیسٹ ہے")
