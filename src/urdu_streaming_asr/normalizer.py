import hashlib
import re
import unicodedata
from typing import Dict, Tuple

NORMALIZER_VERSION = "urdu_norm_v1.0"

BIDI_AND_ZERO_WIDTH = {
    "\u200b",
    "\u200c",
    "\u200d",
    "\u200e",
    "\u200f",
    "\u202a",
    "\u202b",
    "\u202c",
    "\u202d",
    "\u202e",
    "\u2060",
    "\u2066",
    "\u2067",
    "\u2068",
    "\u2069",
    "\ufeff",
    "\u0640",
}

DIACRITICS = {
    "\u0610", "\u0611", "\u0612", "\u0613", "\u0614", "\u0615",
    "\u0616", "\u0617", "\u0618", "\u0619", "\u061a",
    "\u064b", "\u064c", "\u064d", "\u064e", "\u064f", "\u0650",
    "\u0651", "\u0652", "\u0653", "\u0654", "\u0655", "\u0656",
    "\u0657", "\u0658", "\u0659", "\u065a", "\u065b", "\u065c",
    "\u065d", "\u065e", "\u065f", "\u0670",
}

ARABIC_TO_URDU_CANONICAL: Dict[str, str] = {
    "\u0643": "\u06a9",
    "\u06aa": "\u06a9",
    "\u06ac": "\u06a9",
    "\u064a": "\u06cc",
    "\u0649": "\u06cc",
    "\u06d0": "\u06cc",
    "\u0647": "\u06c1",
    "\u06d5": "\u06c1",
    "\u06c0": "\u06c2",
    "\u0629": "\u06c1",
    "\u06ba": "\u06ba",
    "\u06be": "\u06be",
    "\u0679": "\u0679",
    "\u0688": "\u0688",
    "\u0691": "\u0691",
    "\u06d2": "\u06d2",
    "\u0622": "\u0622",
    "\u0623": "\u0627",
    "\u0625": "\u0627",
    "\u0671": "\u0627",
}

DIGIT_MAP: Dict[str, str] = {
    "٠": "0", "١": "1", "٢": "2", "٣": "3", "٤": "4",
    "٥": "5", "٦": "6", "٧": "7", "٨": "8", "٩": "9",
    "۰": "0", "۱": "1", "۲": "2", "۳": "3", "۴": "4",
    "۵": "5", "۶": "6", "۷": "7", "۸": "8", "۹": "9",
}

COMPOSITE_SEQUENCES: Tuple[Tuple[str, str], ...] = (
    ("\u0627\u0653", "\u0622"),
    ("\u06cc\u0654", "\u0626"),
    ("\u0648\u0654", "\u0624"),
    ("\u06c1\u0654", "\u06c2"),
    ("\u06d2\u0654", "\u06d3"),
)

PUNCT_PATTERN = re.compile(
    r"[\u060c\u061b\u061f\u06d4\u066a\u066b\u066c!\"#$%&()*+,\-./:;<=>?@\[\\\]^_`{|}~«»“”‘’—–…،؛؟۔]+"
)
WHITESPACE_PATTERN = re.compile(r"\s+")
URDU_CHAR_PATTERN = re.compile(r"[\u0600-\u06ff\u0750-\u077f]")
LATIN_CHAR_PATTERN = re.compile(r"[a-zA-Z]")
ALLOWED_TOKEN_CHARS = re.compile(r"^[\u0600-\u06ff\u0750-\u077fa-z0-9' ]+$")


class UrduNormalizer:
    version: str = NORMALIZER_VERSION

    def normalize(self, text: str) -> str:
        if not text:
            return ""
        s = unicodedata.normalize("NFKC", text)
        for src, dst in COMPOSITE_SEQUENCES:
            s = s.replace(src, dst)
        out_chars = []
        for ch in s:
            if ch in BIDI_AND_ZERO_WIDTH:
                continue
            if ch in DIACRITICS:
                continue
            if ch in DIGIT_MAP:
                out_chars.append(DIGIT_MAP[ch])
                continue
            if ch in ARABIC_TO_URDU_CANONICAL:
                out_chars.append(ARABIC_TO_URDU_CANONICAL[ch])
                continue
            out_chars.append(ch)
        s = "".join(out_chars)
        s = s.replace("’", "'").replace("‘", "'")
        s = re.sub(r"(?<![a-zA-Z])'|'(?![a-zA-Z])", " ", s)
        s = PUNCT_PATTERN.sub(" ", s)
        s = s.lower()
        s = WHITESPACE_PATTERN.sub(" ", s).strip()
        return s

    def text_hash(self, text: str) -> str:
        norm = self.normalize(text)
        return hashlib.sha256(norm.encode("utf-8")).hexdigest()


_DEFAULT_NORMALIZER = UrduNormalizer()


def normalize_urdu_text(text: str) -> str:
    return _DEFAULT_NORMALIZER.normalize(text)


def classify_language_mix(text: str) -> str:
    has_urdu = bool(URDU_CHAR_PATTERN.search(text))
    has_latin = bool(LATIN_CHAR_PATTERN.search(text))
    if has_urdu and has_latin:
        return "ur_en_codeswitch"
    if has_urdu:
        return "ur_only"
    if has_latin:
        return "en_only"
    return "other"
