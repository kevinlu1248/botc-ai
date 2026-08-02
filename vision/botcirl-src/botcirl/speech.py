"""Ties the microphone to the people in the room.

Owns the voice gallery, the voice-to-face binder, and the bookkeeping that turns
a finished speech segment into "Sam said something, for 2.3 seconds, and here is
how sure we are it was Sam".

Runs off the video loop's thread but never blocks it: audio capture and VAD live
on their own thread, and the embedding of a finished segment costs ~20 ms, which
is cheap enough to do inline once per utterance.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from .audio import AudioListener, SpeechSegment
from .binding import Attribution, SpeakerBinder
from .config import DATA_DIR, VOICE_GALLERY_DIR, Config
from .identity import IdentityGallery
from .voices import build_voice_backend, voiceprint


@dataclass
class SpeakingNow:
    """Short-lived UI state: who the last utterance was attributed to."""

    pid: str | None
    label: str
    until: float
    confidence: float
    basis: str


class SpeechTracker:
    def __init__(self, cfg: Config, pipeline):
        self.cfg = cfg
        self.pipeline = pipeline
        self.backend = build_voice_backend(cfg.voice)
        self.voices = IdentityGallery(
            VOICE_GALLERY_DIR,
            max_per_identity=cfg.voice.embeddings_per_identity,
            prefix="v",  # keeps voice ids out of the face gallery's namespace
            label_word="Speaker",
        )
        self.voices.assert_compatible(self.backend.embedding_dim)
        if cfg.voice.match_threshold is None:
            cfg.voice.match_threshold = self.backend.default_match_threshold
        if cfg.voice.enroll_threshold is None:
            cfg.voice.enroll_threshold = self.backend.default_enroll_threshold

        self.binder = SpeakerBinder(DATA_DIR / "voice_face_links.json")
        self.listener = AudioListener(cfg.audio)
        self.events: list[dict] = []
        self.speaking: SpeakingNow | None = None
        self.history: list[Attribution] = []

    # ---------- lifecycle ----------

    def start(self) -> SpeechTracker:
        self.listener.start()
        print(f"[speech] listening - voice gallery has {len(self.voices.identities)} speakers")
        for voice, face, strength in self.binder.linked_pairs():
            print(f"[speech] known link: {voice} -> {self.pipeline.gallery.label_of(face)} "
                  f"({strength:.2f})")
        return self

    def stop(self) -> None:
        self.listener.stop()
        self.voices.save()
        self.binder.save()

    # ---------- per-frame ----------

    def poll(self) -> list[Attribution]:
        """Handle any utterances that finished since the last call."""
        out = []
        for segment in self.listener.poll():
            att = self._handle(segment)
            if att is not None:
                out.append(att)
        # Expire the on-screen "speaking" flag.
        if self.speaking and time.time() > self.speaking.until:
            self.speaking = None
        return out

    def _handle(self, segment: SpeechSegment) -> Attribution | None:
        vcfg = self.cfg.voice
        print_ = voiceprint(self.backend, segment)
        if print_ is None:
            return None

        voice_pid, sim = self.voices.match(print_.embedding)
        quality = print_.quality()

        if voice_pid is not None and sim >= vcfg.match_threshold:
            self.voices.reinforce(voice_pid, print_.embedding, quality)
        elif (
            sim < vcfg.enroll_threshold
            and segment.duration >= vcfg.min_enroll_s
            and quality >= vcfg.min_enroll_quality
        ):
            # A new voice. Same reasoning as faces: only mint one when there is
            # enough signal, because an identity created from a cough is
            # permanent and pollutes every later match.
            idt = self.voices.enroll(print_.embedding, label=None)
            voice_pid = idt.pid
            self.voices.save()
            self._emit("voice_created", pid=voice_pid, duration_s=round(segment.duration, 2))
        else:
            voice_pid = None  # heard, but not confidently anyone

        present = self.pipeline.present_during(segment.start, segment.end)
        known_faces = set(self.pipeline.gallery.identities)
        att = self.binder.attribute(segment, voice_pid, present, known_faces)
        self.history.append(att)

        label = (
            self.pipeline.gallery.label_of(att.speaker_pid)
            if att.speaker_pid
            else (voice_pid or "unknown voice")
        )
        self.speaking = SpeakingNow(
            pid=att.speaker_pid,
            label=label,
            until=time.time() + 1.5,
            confidence=att.confidence,
            basis=att.basis,
        )
        self._emit(
            "speech",
            start=round(segment.start, 3),
            end=round(segment.end, 3),
            duration_s=round(segment.duration, 2),
            voice_pid=voice_pid,
            voice_similarity=round(sim, 3),
            speaker_pid=att.speaker_pid,
            label=label,
            present=sorted(present),
            confidence=round(att.confidence, 2),
            basis=att.basis,
        )
        return att

    def _emit(self, kind: str, **payload) -> None:
        self.events.append({"t": time.time(), "event": kind, **payload})

    def drain_events(self) -> list[dict]:
        out, self.events = self.events, []
        return out

    # ---------- reporting ----------

    def summary(self) -> str:
        lines = [f"voices: {len(self.voices.identities)}"]
        for pid, idt in self.voices.identities.items():
            face, strength = self.binder.best_link(pid)
            who = (
                f" -> {self.pipeline.gallery.label_of(face)} ({strength:.2f})"
                if face else " -> unlinked"
            )
            lines.append(f"  {pid}: {len(idt.embeddings)} samples, {idt.sightings} utterances{who}")
        attributed = sum(1 for a in self.history if a.attributed)
        lines.append(f"utterances: {len(self.history)} ({attributed} attributed to a person)")
        return "\n".join(lines)
