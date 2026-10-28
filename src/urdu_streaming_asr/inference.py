from dataclasses import dataclass
from pathlib import Path
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf

from .quality import TARGET_SAMPLE_RATE, resample_linear_1d


@dataclass
class StreamingTranscriptEvent:
    chunk_index: int
    audio_time_sec: float
    wall_time_sec: float
    text: str
    is_final: bool


class UrduStreamingRecognizer:
    def __init__(
        self,
        encoder_path: str | Path,
        decoder_path: str | Path,
        joiner_path: str | Path,
        tokens_path: str | Path,
        num_threads: int = 1,
        provider: str = "cpu",
        decoding_method: str = "greedy_search",
        max_active_paths: int = 4,
        blank_penalty: float = 0.0,
        enable_endpoint: bool = True,
        rule1_min_trailing_silence: float = 2.4,
        rule2_min_trailing_silence: float = 1.2,
        rule3_min_utterance_length: float = 20.0,
    ):
        import sherpa_onnx

        self.sample_rate = TARGET_SAMPLE_RATE
        self.num_threads = num_threads
        self.provider = provider
        self.recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
            tokens=str(tokens_path),
            encoder=str(encoder_path),
            decoder=str(decoder_path),
            joiner=str(joiner_path),
            num_threads=num_threads,
            sample_rate=self.sample_rate,
            feature_dim=80,
            decoding_method=decoding_method,
            max_active_paths=max_active_paths,
            blank_penalty=blank_penalty,
            enable_endpoint_detection=enable_endpoint,
            rule1_min_trailing_silence=rule1_min_trailing_silence,
            rule2_min_trailing_silence=rule2_min_trailing_silence,
            rule3_min_utterance_length=rule3_min_utterance_length,
            provider=provider,
        )
        self.stream = self.recognizer.create_stream()

    def reset(self) -> None:
        self.stream = self.recognizer.create_stream()

    def accept_chunk(self, samples: np.ndarray, sample_rate: int = TARGET_SAMPLE_RATE) -> str:
        if samples.ndim == 2:
            samples = samples.mean(axis=1, dtype=np.float32)
        else:
            samples = samples.astype(np.float32, copy=False)
        if sample_rate != self.sample_rate:
            samples = resample_linear_1d(samples, sample_rate, self.sample_rate)
        self.stream.accept_waveform(self.sample_rate, samples)
        while self.recognizer.is_ready(self.stream):
            self.recognizer.decode_stream(self.stream)
        return self.recognizer.get_result(self.stream).strip()

    def is_endpoint(self) -> bool:
        return bool(self.recognizer.is_endpoint(self.stream))

    def finalize(self) -> str:
        tail_pad = np.zeros(int(0.66 * self.sample_rate), dtype=np.float32)
        self.stream.accept_waveform(self.sample_rate, tail_pad)
        self.stream.input_finished()
        while self.recognizer.is_ready(self.stream):
            self.recognizer.decode_stream(self.stream)
        text = self.recognizer.get_result(self.stream).strip()
        return text

    def transcribe_file_streaming(
        self,
        audio_path: str | Path,
        chunk_ms: int = 320,
    ) -> Tuple[str, List[StreamingTranscriptEvent], Dict[str, float]]:
        self.reset()
        data, sr = sf.read(str(audio_path), dtype="float32", always_2d=True)
        mono = data.mean(axis=1, dtype=np.float32) if data.shape[1] > 1 else data[:, 0]
        if sr != self.sample_rate:
            mono = resample_linear_1d(mono, sr, self.sample_rate)
            sr = self.sample_rate

        chunk_samples = max(1, int(round((chunk_ms / 1000.0) * sr)))
        events: List[StreamingTranscriptEvent] = []
        t0 = time.perf_counter()
        first_partial_sec: Optional[float] = None
        last_text = ""

        num_chunks = (len(mono) + chunk_samples - 1) // chunk_samples
        for idx in range(num_chunks):
            chunk = mono[idx * chunk_samples : (idx + 1) * chunk_samples]
            current = self.accept_chunk(chunk, sr)
            elapsed = time.perf_counter() - t0
            audio_t = min(len(mono), (idx + 1) * chunk_samples) / float(sr)
            if current and first_partial_sec is None:
                first_partial_sec = elapsed
            if current != last_text:
                events.append(
                    StreamingTranscriptEvent(
                        chunk_index=idx,
                        audio_time_sec=round(audio_t, 4),
                        wall_time_sec=round(elapsed, 4),
                        text=current,
                        is_final=False,
                    )
                )
                last_text = current

        t_before_flush = time.perf_counter()
        final_text = self.finalize()
        t_end = time.perf_counter()
        total_wall = t_end - t0
        finalize_delay = t_end - t_before_flush
        audio_dur = len(mono) / float(sr)
        events.append(
            StreamingTranscriptEvent(
                chunk_index=num_chunks,
                audio_time_sec=round(audio_dur, 4),
                wall_time_sec=round(total_wall, 4),
                text=final_text,
                is_final=True,
            )
        )
        metrics = {
            "audio_duration_sec": round(audio_dur, 4),
            "wall_time_sec": round(total_wall, 4),
            "rtf": round(total_wall / max(1e-6, audio_dur), 5),
            "time_to_first_partial_sec": round(first_partial_sec if first_partial_sec is not None else total_wall, 4),
            "finalization_delay_sec": round(finalize_delay, 4),
            "algorithmic_chunk_ms": 320.0,
            "algorithmic_lookahead_ms": 40.0,
        }
        return final_text, events, metrics
