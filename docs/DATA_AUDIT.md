# Data Quality & Leakage Audit Report

## 1.Canonical Schema & Deterministic Screening
All utterances across all sources were processed through source-specific adapters into the 21-field canonical schema (`src/urdu_streaming_asr/schema.py`) and converted to 16 kHz mono PCM-16 FLAC (`src/urdu_streaming_asr/quality.py`).

### Quality Screening Rules Applied:
- **Duration Window**: `0.5s – 25.0s` for training utterances (`0.4s – 35.0s` for evaluation sets).
- **Signal Integrity**: Rejected empty, non-finite, or near-silent clips (`RMS < 0.001`), severe clipping (`> 5%` samples at full scale), and excessive silence (`> 85%` silent frames).
- **Character Rate Plausibility**: Rejected transcripts outside `1.5 – 32.0` non-space characters/sec (`2.2 – 26.0` chars/sec for silver labels, plus repetitive-token hallucination screening).
- **Urdu Normalization (`urdu_norm_v1.0`)**: Canonicalized Arabic/Persian presentation variants (`ك -> ک`, `ي/ى -> ی`, `ه -> ہ`), converted Eastern Arabic/Urdu digits to ASCII `0-9`, removed zero-width/bidi control characters, preserved Urdu letters (`ھ`, `ہ`, `ں`, `ٹ`, `ڈ`, `ڑ`, `ء`, `ئ`, `ؤ`, `ے`), and preserved English words in code-switched transcripts.

## 2. Accepted & Rejected Utterances per Source/Split
Full machine-readable breakdown is stored in `reports/data_audit_summary.json`.

| Source & Split | Discovered | Accepted | Rejected | Accepted Hours | Unique Speakers | Label Origin | Primary Domain(s) |
|---|---:|---:|---:|---:|---:|---|---|
| `kathbath_ur:train` | 900 | 900 | 0 | 1.3246 h | 14 | `human_gold` | `read_speech` |
| `indicvoices_ur:train` | 882 | 875 | 7 (`duration_too_short`) | 1.6125 h | 146 | `human_gold` | `extempore` (518), `conversation` (293), `read` (64) |
| `urduspeech_aslp:train` | 280 | 279 | 1 (`silver_implausible_cps`) | 0.3725 h | 85 | `silver_model_validated` | `comedy` (80), `film` (80), `drama` (79), `food` (40) |
| `rasa_ur:train` | 200 | 199 | 1 (`duration_too_short`) | 0.3559 h | 2 | `human_studio_expressive` | 15 expressive/studio styles |
| **Total Unique Train** | **2,262** | **2,253** | **9** | **3.6656 h** | **247** | Mixed Human + Screened Silver | Multi-domain Urdu + Code-Switched |
| `kathbath_ur:dev` | 418 | 418 | 0 | 0.6130 h | 8 | `human_gold` | `read_speech` |
| `indicvoices_ur:dev` | 151 | 151 | 0 | 0.3369 h | 42 | `human_gold` | `extempore`, `conversation`, `read` |
| `fleurs_ur_pk:dev` | 200 | 200 | 0 | 0.5685 h | 200 | `human_gold` | `external_read_benchmark` |
| `kathbath_ur:test` | 182 | 182 | 0 | 0.2432 h | 4 | `human_gold` | `read_speech` |
| `indicvoices_ur:test` | 245 | 245 | 0 | 0.4433 h | 41 | `human_gold` | `extempore`, `conversation`, `read` |
| `urduspeech_aslp:test` | 52 | 52 | 0 | 0.1957 h | 18 | `human_gold_benchmark` | `US-benchmark-CS` (`comedyshow`, `drama`, `film`) |
| `rasa_ur:test` | 100 | 100 | 0 | 0.1799 h | 2 | `human_studio_expressive` | Studio expressive styles |
| `fleurs_ur_pk:test` | 200 | 200 | 0 | 0.5195 h | 200 | `human_gold` | `external_read_benchmark` |
| `csalt_voice:test` | 159 | 159 | 0 | 0.4065 h | 159 | `human_gold` | `conversational_benchmark` |

## 3. Leakage Prevention & Deduplication Results (`reports/leakage_report.json`)
- **Exact Audio SHA256 Collisions with Dev/Test**: `0`
- **Source Recording ID Collisions with Sealed Eval (`US-benchmark`, etc.)**: `0`
- **Long Transcript (`>= 5` words) Collisions with Dev/Test**: `0`
- **Short Common Prompt Overlaps (`< 5` words, e.g. single words/greetings) Retained with Flag**: `41`
- **Speaker Disjointness**: Official validation splits of `Kathbath` and `IndicVoices` were deterministically partitioned by `md5(speaker_id)` into speaker-disjoint `dev` and `test` sets.

## 4. Stratified Sample Audit (`reports/stratified_sample_audit.json`)
- A deterministic stratified sample of **60 clips** across admitted training and development sources was audited for 16 kHz mono FLAC decoding, duration accuracy, character-rate alignment, and script normalization, with an interactive local HTML review page generated at `reports/review_page/index.html`.
