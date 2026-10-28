# Data Sources, Provenance, Licensing, and Access Inventory

## 1. Summary Inventory of Investigated Sources

| Source ID | Canonical Owner & URL | Pinned Revision / SHA | Urdu Config / Files | Label Origin | License / Terms | Access Status | Role in Pipeline |
|---|---|---|---|---|---|---|---|
| `common_voice_ur` | Mozilla Data Collective (`https://mozilladatacollective.com/datasets/cmu5x1w3400cyo107st8cgsoq`) | Archive SHA256: `12ad5f359fd94805db1bdaeb1441eb35720221f3c99229034069375fa246d990` | `common-voice-scripted-speech-27-0-urdu-19469962.tar.gz` (6.22 GB) | Human-validated read speech | CC0-1.0 with MDC terms prohibiting dataset re-hosting and speaker identification | **Blocked (HTTP 401)** on `/api/datasets/.../download` without `MDC_API_KEY`; HF `mozilla-foundation/common_voice_17_0` removed data files | Adapter implemented; pending `MDC_API_KEY` |
| `kathbath_ur` | AI4Bharat (`https://huggingface.co/datasets/ai4bharat/Kathbath`, `https://github.com/AI4Bharat/IndicSUPERB`) | `5b9e92849222026d9141acba4e8434fe816396bf` | `urdu/train-00000-of-00017.parquet`..`00016` (49,201 rows), `urdu/valid-00000-of-00002.parquet`..`00001` (3,229 rows) | Human-labeled read speech | Hub metadata: `CC-BY-4.0`; README notes CC0 packaging with source text attribution caveats | **Accessible (HTTP 200)** via authorized HF session | Core human-labeled training + held-out dev/test |
| `indicvoices_ur` | AI4Bharat (`https://huggingface.co/datasets/ai4bharat/IndicVoices`, `https://github.com/AI4Bharat/IndicVoices`) | `c96f9088f138cf89d419da7e8e643e1f05c00a87` | `urdu/train-00000-of-00066.parquet`..`00065` (255,626 rows), `urdu/valid-00000-of-00001.parquet` (3,383 rows) | Human-labeled read, extempore, and conversational speech | `CC-BY-4.0` (with attribution to AI4Bharat IndicVoices) | **Accessible (HTTP 200)** via authorized HF session | Core human-labeled spontaneous/conversational training + held-out dev/test |
| `urduspeech_aslp` | ASLP-lab (`https://huggingface.co/datasets/ASLP-lab/UrduSpeech`, `https://arxiv.org/abs/2605.17846`) | `16dd380cfd9049a3db7f06a98e878086916bf833` | `corpus/US-Std/*`, `corpus/US-CS/*`, `corpus/US-EngPk/*` (train candidates); `benchmark/US-benchmark-*` (eval) | `corpus/`: Silver (Gemini 2.5 Pro pipeline); `benchmark/`: Gold manually corrected | `CC-BY-4.0` | **Accessible (HTTP 200)** | Screened silver Pakistani/code-switched training (`corpus/`) + sealed benchmark (`benchmark/`) |
| `rasa_ur` | AI4Bharat (`https://huggingface.co/datasets/ai4bharat/Rasa`) | `632f55c7ac590219d41cd7adffce5b440e4604f5` | `Urdu/train-00000-of-00056.parquet`..`00055` (26,102 rows), `Urdu/test-00000-of-00007.parquet`..`00006` (2,901 rows) | Human studio expressive speech (TTS recording corpus, 2 speakers) | `CC-BY-4.0` | **Accessible (HTTP 200)** via authorized HF session | Capped supplemental clean expressive training (<=10% mixture weight) |
| `fleurs_ur_pk` | Google (`https://huggingface.co/datasets/google/fleurs`) | `70bb2e84b976b7e960aa89f1c648e09c59f894dd` | `data/ur_pk/audio/{dev,test}.tar.gz`, `data/ur_pk/{dev,test}.tsv` | Human-labeled read encyclopedic speech | `CC-BY-4.0` | **Accessible (HTTP 200)** | Sealed external domain evaluation (`dev` and `test` only; `train` unused) |
| `csalt_voice` | CSALT / `sameearif/WER-We-Stand` (`https://huggingface.co/datasets/urdu-asr/csalt-voice`, `https://aclanthology.org/2025.coling-main.397/`) | `6f0c4fdeea94c77b4b64c4b88d5887e087566a5f` | `data/validation-00000-of-00002.parquet`..`00001` (471 rows) | Human-labeled conversational speech | Public benchmark release accompanying COLING 2025 paper | **Accessible (HTTP 200)** | Sealed conversational evaluation benchmark |
| `cle_phonetic` | CLE Pakistan (`https://cle.org.pk/software/ling_resources/phoneticallyrichurduspeechcorpus.htm`) | Archive: `UrduPhoneticSpeechCorpus.rar` | 708 sentences (~70 mins) | Human read phonetic sentences | Dual linked notices on page: ITU 2017 CC-BY (`CreativeCommonsnew-1.html`) AND CRULP Non-Commercial (`CreativeCommons.html`) | Excluded from primary general-purpose release due to conflicting NC license link on landing page | Documented in excluded/optional research table |

## 2. Local & Remote Cache Inspection (Section 6.F Audit)
- Local `D:\listening-ur`: Contains 10 `.wav` files (`ur_01_label.wav`..`ur_10_paragraph.wav`) and `urdu_fasih_tts_evaluation.csv`. This is synthetic TTS evaluation material -> **Excluded**.
- Local `D:\pak-cleaned-multilang\data`: Contains 3 parquet shards with columns `(text: string, lang: string)` only (no audio) -> **Excluded**.
- Local `D:\Training_data` and `D:\Testing_data`: Contains document OCR images/PDFs unrelated to speech -> **Excluded**.
- Remote `/home/proxima/.cache/huggingface/datasets`: Empty prior to this project. Other project directories under `~/PROXIMA/MAHWIZ` (`K-Electric-Voice-Bot`, `Vowren-*`, `grafon-*`, `privacy-filter`, `llama.cpp`) were left untouched per mutation scope rules.

## 3. Source Schemas & Split Policy
1. **Kathbath (`ai4bharat/Kathbath`, config `urdu`)**:
   - Parquet columns: `fname` (str), `text` (str), `audio_filepath` (struct `{bytes, path}`), `lang` (str), `duration` (float64), `gender` (str), `speaker_id` (int64).
   - Official splits in Hub parquet release: `train` (17 shards) and `valid` (2 shards). To preserve a strictly sealed test split while supporting development selection, `valid` is deterministically partitioned by `speaker_id` into `dev` (50% of validation speakers) and `test` (50% of validation speakers), with zero speaker overlap between `train`, `dev`, and `test`.
2. **IndicVoices (`ai4bharat/IndicVoices`, config `urdu`)**:
   - Parquet columns: `audio_filepath` (struct `{bytes, path}`), `text` (str), `duration` (float64), `lang` (str), `samples` (int64), `verbatim` (str), `normalized` (str), `speaker_id` (str), `scenario` (str: `read`, `extempore`, `conversational`), `task_name` (str), `gender` (str), `age_group` (str), `district` (str), `state` (str).
   - Uses ordinary collected speech (`ai4bharat/IndicVoices`), NOT restored `IndicVoices-R`.
   - Official `valid` split is deterministically partitioned by `speaker_id` into `dev` and `test` so speaker-disjoint dev and test sets are both protected.
3. **ASLP UrduSpeech (`ASLP-lab/UrduSpeech`)**:
   - JSONL metadata fields: `Speaker_id`, `Audio_category`, `Audio_Clip`, `audio_id`, `Duration_seconds`, `Word_count`, `Character_count`, `Transcription`.
   - `benchmark/US-benchmark-{Std,CS,EngPk}`: Reserved 100% for evaluation (`test_us_benchmark`). Any recording/clip prefix appearing in `benchmark/` is blocked from entering training.
   - `corpus/US-{Std,CS,EngPk}/short/*`: Screened silver training candidates. Filtered by decoded audio duration, char-rate plausibility, non-empty normalized transcript, and audio hash deduplication.
4. **Rasa (`ai4bharat/Rasa`, config `Urdu`)**:
   - Parquet columns: `filename`, `text`, `language`, `gender`, `style`, `duration`, `wav_path`, `audio` (`{bytes, path}`).
   - Studio expressive recordings from 2 speakers (male/female). Used only as a capped (<=10%) supplemental source.
5. **FLEURS (`google/fleurs`, config `ur_pk`)**:
   - `dev.tsv` + `dev.tar.gz` -> `dev_fleurs`; `test.tsv` + `test.tar.gz` -> `test_fleurs`. `train` split is not used so FLEURS remains a pure external domain benchmark.
6. **CSALT-Voice (`urdu-asr/csalt-voice`)**:
   - Parquet columns: `audio` (`{bytes, path}`), `raw_transcription`, `transcription`.
   - Used strictly as an external conversational evaluation benchmark (`test_csalt`).

## 4. Incremental Disk-Safe Acquisition Budget
To respect the 149 GiB free space on `/` and maintain >= 40 GiB free at all times:
- We download only specific Urdu parquet shards / subsets incrementally, extract 16 kHz mono FLAC directly into `data/audio_flac/<source>/<split>/`, record exact SHA256 hashes and canonical manifest entries, and remove redundant raw parquet/archive cache shards once verified in the manifest.
