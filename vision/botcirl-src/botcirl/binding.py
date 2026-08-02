"""Attributing speech to a person.

Three sources of evidence, in increasing order of how much they are worth:

  1. presence     - only people visible while the words were said can be the
                    speaker. When exactly one person is in the room this alone
                    settles it, and that is the case worth exploiting, because
                    it is how the system bootstraps everything else.
  2. learned link - having watched enough one-person moments, "voice 3 belongs
                    to face 2" becomes a fact that survives a crowded room and
                    a turned back.
  3. direction    - not available on this laptop. macOS hands over a single
                    beamformed channel, so there is no angle to compare against
                    where people are standing. Arrives with a USB mic array.

The learned link is the interesting one. Nobody has to label anything: the robot
sits in a room, notices that whenever this voice speaks that face is present and
that when the face is away the voice never appears, and concludes they are the
same person. Negative evidence is what makes it sound - co-occurrence alone
would link a voice to whoever simply happens to be in the room most.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from .audio import SpeechSegment


@dataclass
class Attribution:
    """Who said it, and how much to trust that."""

    segment: SpeechSegment
    voice_pid: str | None = None  # identity in the voice gallery
    speaker_pid: str | None = None  # identity in the *face* gallery
    candidates: tuple[str, ...] = ()  # face identities present while speaking
    confidence: float = 0.0
    basis: str = "unattributed"

    @property
    def attributed(self) -> bool:
        return self.speaker_pid is not None


@dataclass
class LinkStats:
    """Evidence that one voice and one face are the same person."""

    together: int = 0  # this voice spoke and this face was present
    apart: int = 0  # this voice spoke and this face was not

    @property
    def support(self) -> int:
        return self.together + self.apart

    def strength(self) -> float:
        """How strongly the evidence says these are the same person.

        A ratio alone would call one lucky co-occurrence a certainty, so it is
        discounted by how little evidence there is. Twenty observations at 95%
        should beat two observations at 100%, and this does that.
        """
        if self.support == 0:
            return 0.0
        ratio = self.together / self.support
        confidence = 1.0 - math.exp(-self.support / 6.0)
        return float(ratio * confidence)


class SpeakerBinder:
    """Learns which voice goes with which face, and attributes speech."""

    def __init__(self, path: Path | None = None, min_strength: float = 0.6,
                 min_support: int = 4):
        self.path = Path(path) if path is not None else None
        self.min_strength = min_strength
        self.min_support = min_support
        self.links: dict[str, dict[str, LinkStats]] = defaultdict(lambda: defaultdict(LinkStats))
        if self.path is not None:
            self.load()

    # ---------- learning ----------

    def observe(self, voice_pid: str, present: set[str], known_faces: set[str]) -> None:
        """Record one utterance: this voice spoke, these faces were present.

        `known_faces` is every face identity the system has ever seen, so that
        absence counts as evidence too - without it a voice would link equally
        well to everyone who happens to share the room.
        """
        if not voice_pid:
            return
        for face_pid in known_faces:
            stats = self.links[voice_pid][face_pid]
            if face_pid in present:
                stats.together += 1
            else:
                stats.apart += 1

    def best_link(self, voice_pid: str) -> tuple[str | None, float]:
        """The face this voice most likely belongs to, and how sure we are."""
        best_face, best_strength = None, 0.0
        for face_pid, stats in self.links.get(voice_pid, {}).items():
            if stats.support < self.min_support:
                continue
            s = stats.strength()
            if s > best_strength:
                best_face, best_strength = face_pid, s
        return best_face, best_strength

    def linked_pairs(self) -> list[tuple[str, str, float]]:
        """Every voice->face link that currently clears the bar."""
        out = []
        for voice_pid in self.links:
            face, strength = self.best_link(voice_pid)
            if face is not None and strength >= self.min_strength:
                out.append((voice_pid, face, strength))
        return sorted(out, key=lambda p: p[2], reverse=True)

    # ---------- attribution ----------

    def attribute(
        self,
        segment: SpeechSegment,
        voice_pid: str | None,
        present: set[str],
        known_faces: set[str] | None = None,
    ) -> Attribution:
        att = Attribution(
            segment=segment, voice_pid=voice_pid, candidates=tuple(sorted(present))
        )

        # A learned link beats presence: it still works in a crowded room, and
        # it is the only thing that can name a speaker whose face is turned away.
        if voice_pid:
            face, strength = self.best_link(voice_pid)
            if face is not None and strength >= self.min_strength:
                att.speaker_pid = face
                att.confidence = strength
                att.basis = "learned voice-face link"
                # Only reinforce when the visual evidence agrees, so a wrong
                # link cannot keep confirming itself forever.
                if known_faces and face in present:
                    self.observe(voice_pid, present, known_faces)
                return att

        if len(present) == 1:
            only = next(iter(present))
            att.speaker_pid = only
            att.confidence = 0.9
            att.basis = "only person visible"
            if voice_pid and known_faces:
                self.observe(voice_pid, present, known_faces)
            return att

        if not present:
            att.basis = "nobody visible"
            return att

        att.basis = f"{len(present)} people visible, cannot tell them apart"
        return att

    # ---------- persistence ----------

    def save(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        blob = {
            voice: {face: {"together": s.together, "apart": s.apart}
                    for face, s in faces.items()}
            for voice, faces in self.links.items()
        }
        self.path.write_text(json.dumps(blob, indent=2))

    def load(self) -> None:
        if self.path is None or not self.path.exists():
            return
        blob = json.loads(self.path.read_text())
        for voice, faces in blob.items():
            for face, s in faces.items():
                self.links[voice][face] = LinkStats(
                    together=s.get("together", 0), apart=s.get("apart", 0)
                )
