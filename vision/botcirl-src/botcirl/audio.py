"""Microphone capture and voice activity detection.

Runs on its own thread and hands finished speech segments to the video loop
through a queue. Audio cannot share the video loop's cadence: frames arrive at
~11 fps and blocking the microphone callback for 90 ms drops samples on the
floor, which VAD then reads as a gap in speech.

Timestamps are the contract between the two. Every segment carries wall-clock
start and end, so it can be lined up against `person_appeared` /
`identity_assigned` / `hand_raised` events after the fact.

On this MacBook the built-in mic is a single processed channel - macOS does its
own beamforming below Core Audio and does not expose the raw capsules - so
there is no direction information to extract here. That arrives with a USB mic
array; see `docs` in the README.
"""

from __future__ import annotations

import queue
import threading
import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from .config import AudioConfig


@dataclass
class SpeechSegment:
    """One continuous stretch of speech, with the audio that made it."""

    start: float  # wall-clock epoch seconds
    end: float
    audio: np.ndarray  # float32 mono at AudioConfig.samplerate
    samplerate: int = 16000

    @property
    def duration(self) -> float:
        return self.end - self.start

    def rms(self) -> float:
        return float(np.sqrt(np.mean(self.audio**2))) if self.audio.size else 0.0

    def __repr__(self) -> str:
        return (
            f"SpeechSegment({self.duration:.2f}s, rms={self.rms():.4f}, "
            f"{self.audio.size} samples)"
        )


class AudioListener:
    """Captures the mic, runs VAD, emits SpeechSegments.

    Usage:
        listener = AudioListener(cfg.audio)
        listener.start()
        ...
        for seg in listener.poll():   # non-blocking, returns what is ready
            ...
        listener.stop()
    """

    def __init__(self, cfg: AudioConfig | None = None):
        self.cfg = cfg or AudioConfig()
        self._segments: queue.Queue[SpeechSegment] = queue.Queue()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._error: Exception | None = None
        self.started_at: float | None = None
        # Rolling window of recent audio, so a segment can be sliced out after
        # VAD confirms where it ended. Sized generously: a long sentence plus
        # the padding either side.
        self._buffer: deque[np.ndarray] = deque()
        self._buffer_start_sample = 0
        self._total_samples = 0
        self._lock = threading.Lock()

    # ---------- lifecycle ----------

    def start(self) -> AudioListener:
        if self._thread is not None:
            return self
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="audio", daemon=True)
        self._thread.start()
        # Surface a bad device immediately rather than silently never hearing.
        deadline = time.time() + 3.0
        while self.started_at is None and self._error is None and time.time() < deadline:
            time.sleep(0.02)
        if self._error is not None:
            raise self._error
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def __enter__(self) -> AudioListener:
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()

    # ---------- consumption ----------

    def poll(self) -> list[SpeechSegment]:
        """Everything finished since the last call. Never blocks."""
        out = []
        while True:
            try:
                out.append(self._segments.get_nowait())
            except queue.Empty:
                return out

    # ---------- internals ----------

    def _sample_to_time(self, sample: int) -> float:
        return (self.started_at or 0.0) + sample / self.cfg.samplerate

    def _slice(self, first: int, last: int) -> np.ndarray:
        """Pull samples [first, last) out of the rolling buffer."""
        with self._lock:
            if not self._buffer:
                return np.zeros(0, np.float32)
            chunks, cursor = [], self._buffer_start_sample
            for block in self._buffer:
                block_end = cursor + len(block)
                if block_end > first and cursor < last:
                    lo = max(0, first - cursor)
                    hi = min(len(block), last - cursor)
                    chunks.append(block[lo:hi])
                cursor = block_end
            return np.concatenate(chunks) if chunks else np.zeros(0, np.float32)

    def _trim_buffer(self) -> None:
        keep = int(self.cfg.buffer_s * self.cfg.samplerate)
        with self._lock:
            while self._buffer and (self._total_samples - self._buffer_start_sample) > keep:
                dropped = self._buffer.popleft()
                self._buffer_start_sample += len(dropped)

    def _run(self) -> None:
        try:
            import sounddevice as sd
            import torch
            from silero_vad import VADIterator, load_silero_vad
        except Exception as exc:  # noqa: BLE001
            self._error = exc
            return

        cfg = self.cfg
        try:
            model = load_silero_vad()
            vad = VADIterator(
                model,
                threshold=cfg.vad_threshold,
                sampling_rate=cfg.samplerate,
                min_silence_duration_ms=cfg.min_silence_ms,
                speech_pad_ms=cfg.speech_pad_ms,
            )
            blocks: queue.Queue[np.ndarray] = queue.Queue()

            def on_audio(indata, frames, time_info, status):
                # Keep this callback trivial: it runs on a realtime audio thread
                # and anything slow here shows up as dropped samples.
                blocks.put(indata[:, 0].copy())

            stream = sd.InputStream(
                samplerate=cfg.samplerate,
                blocksize=cfg.block_size,
                channels=1,
                dtype="float32",
                device=cfg.device,
                callback=on_audio,
            )
            with stream:
                self.started_at = time.time()
                speech_start: int | None = None
                while not self._stop.is_set():
                    try:
                        block = blocks.get(timeout=0.2)
                    except queue.Empty:
                        continue

                    with self._lock:
                        self._buffer.append(block)
                        self._total_samples += len(block)
                    self._trim_buffer()

                    verdict = vad(torch.from_numpy(block))
                    if verdict is None:
                        # Cut a segment that has run on too long, so one
                        # continuous talker still produces usable chunks.
                        if speech_start is not None:
                            held = (self._total_samples - speech_start) / cfg.samplerate
                            if held >= cfg.max_segment_s:
                                self._emit(speech_start, self._total_samples)
                                speech_start = self._total_samples
                        continue

                    if "start" in verdict:
                        speech_start = int(verdict["start"])
                    elif "end" in verdict and speech_start is not None:
                        self._emit(speech_start, int(verdict["end"]))
                        speech_start = None
        except Exception as exc:  # noqa: BLE001
            self._error = exc

    def _emit(self, first: int, last: int) -> None:
        cfg = self.cfg
        if (last - first) / cfg.samplerate < cfg.min_speech_s:
            return  # too short to be a useful utterance
        audio = self._slice(first, last)
        if audio.size == 0:
            return
        self._segments.put(
            SpeechSegment(
                start=self._sample_to_time(first),
                end=self._sample_to_time(last),
                audio=audio,
                samplerate=cfg.samplerate,
            )
        )
