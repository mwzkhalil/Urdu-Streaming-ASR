from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

CANONICAL_FIELDS: List[str] = [
    "id",
    "source",
    "source_revision",
    "source_split",
    "source_recording_id",
    "audio_path_or_reference",
    "segment_start",
    "segment_duration",
    "sample_rate",
    "speaker_group_id",
    "session_group_id",
    "text_raw",
    "text_train",
    "label_origin",
    "domain",
    "language_mix",
    "license_id",
    "audio_hash",
    "text_hash",
    "quality_flags",
    "rejection_reason",
]

OPTIONAL_PROVENANCE_FIELDS: List[str] = [
    "teacher_proposal_text",
    "confidence_evidence",
    "acceptance_decision",
]


@dataclass
class UtteranceRecord:
    id: str
    source: str
    source_revision: str
    source_split: str
    source_recording_id: str
    audio_path_or_reference: str
    segment_start: float
    segment_duration: float
    sample_rate: int
    speaker_group_id: str
    session_group_id: str
    text_raw: str
    text_train: str
    label_origin: str
    domain: str
    language_mix: str
    license_id: str
    audio_hash: str
    text_hash: str
    quality_flags: List[str] = field(default_factory=list)
    rejection_reason: Optional[str] = None
    teacher_proposal_text: Optional[str] = None
    confidence_evidence: Optional[Dict[str, Any]] = None
    acceptance_decision: str = "accepted"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UtteranceRecord":
        missing = [k for k in CANONICAL_FIELDS if k not in data]
        if missing:
            raise ValueError(f"Missing canonical fields: {missing}")
        kwargs = {k: data[k] for k in CANONICAL_FIELDS}
        for opt_k in OPTIONAL_PROVENANCE_FIELDS:
            if opt_k in data:
                kwargs[opt_k] = data[opt_k]
        if "acceptance_decision" not in kwargs:
            rej = kwargs.get("rejection_reason")
            kwargs["acceptance_decision"] = f"rejected:{rej}" if rej else "accepted"
        return cls(**kwargs)

    def is_accepted(self) -> bool:
        return self.rejection_reason is None or self.rejection_reason == ""

    def validate(self) -> None:
        if not self.id or not self.source or not self.source_revision:
            raise ValueError(f"Invalid provenance on record {self.id}")
        if self.is_accepted():
            if self.segment_duration <= 0 or self.sample_rate <= 0:
                raise ValueError(f"Invalid audio duration/rate on accepted record {self.id}")
            if not self.text_train or not self.audio_hash or not self.text_hash:
                raise ValueError(f"Missing normalized text or hashes on accepted record {self.id}")


def write_jsonl_records(path: Path, records: Iterable[UtteranceRecord]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            rec.validate()
            f.write(json.dumps(rec.to_dict(), ensure_ascii=False) + "\n")
            count += 1
    return count


def read_jsonl_records(path: Path) -> List[UtteranceRecord]:
    records: List[UtteranceRecord] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(UtteranceRecord.from_dict(json.loads(line)))
    return records
