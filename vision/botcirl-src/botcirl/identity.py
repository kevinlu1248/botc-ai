"""The face gallery: who we have seen before, and what they look like.

An identity is a bag of L2-normalised embeddings plus a label. Matching is
cosine similarity against the *best* embedding in each bag, which is what makes
"same person, different angle" work: one enrolled profile shot is enough to
carry a head turn that the frontal shot would miss.

The gallery persists to disk, so restarting the process does not forget anyone.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import config


@dataclass
class Identity:
    pid: str
    label: str
    embeddings: list[np.ndarray] = field(default_factory=list)
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    sightings: int = 0

    def matrix(self) -> np.ndarray:
        return np.stack(self.embeddings) if self.embeddings else np.zeros((0, 1), np.float32)


class IdentityGallery:
    """A set of identities matched by cosine similarity on unit embeddings.

    Used for both faces and voices. `prefix` keeps their ids in separate
    namespaces - without it both galleries mint "p1", and a voice-to-face link
    of "p1 -> p1" is two unrelated people that happen to share a string.
    """

    def __init__(self, path: Path | None = None, max_per_identity: int = 12,
                 prefix: str = "p", label_word: str = "Person"):
        # Resolved here, not as a default argument: a default binds the module
        # global once at import, so tests that redirect GALLERY_DIR to a tmp dir
        # would silently keep writing to the real one.
        self.path = Path(path) if path is not None else Path(config.GALLERY_DIR)
        self.path.mkdir(parents=True, exist_ok=True)
        self.max_per_identity = max_per_identity
        self.prefix = prefix
        self.label_word = label_word
        self.identities: dict[str, Identity] = {}
        self._next_num = 1
        self._dim: int | None = None
        self.load()

    # ---------- persistence ----------

    @property
    def _meta_file(self) -> Path:
        return self.path / "gallery.json"

    @property
    def _emb_file(self) -> Path:
        return self.path / "embeddings.npz"

    def load(self) -> None:
        if not self._meta_file.exists():
            return
        meta = json.loads(self._meta_file.read_text())
        embs = np.load(self._emb_file) if self._emb_file.exists() else {}
        for pid, rec in meta.get("identities", {}).items():
            arr = embs[pid] if pid in embs else np.zeros((0, 0), np.float32)
            self.identities[pid] = Identity(
                pid=pid,
                label=rec.get("label", pid),
                embeddings=[np.asarray(v, np.float32) for v in arr],
                first_seen=rec.get("first_seen", time.time()),
                last_seen=rec.get("last_seen", time.time()),
                sightings=rec.get("sightings", 0),
            )
            if arr.size:
                self._dim = arr.shape[1]
        self._next_num = meta.get("next_num", len(self.identities) + 1)
        print(f"[gallery] loaded {len(self.identities)} identities from {self.path}")

    def save(self) -> None:
        meta = {
            "next_num": self._next_num,
            "dim": self._dim,
            "identities": {
                pid: {
                    "label": idt.label,
                    "first_seen": idt.first_seen,
                    "last_seen": idt.last_seen,
                    "sightings": idt.sightings,
                }
                for pid, idt in self.identities.items()
            },
        }
        self._meta_file.write_text(json.dumps(meta, indent=2))
        np.savez_compressed(
            self._emb_file,
            **{pid: idt.matrix() for pid, idt in self.identities.items() if idt.embeddings},
        )

    @property
    def dim(self) -> int | None:
        """Embedding width of what is stored. None if the gallery is empty."""
        return self._dim

    def assert_compatible(self, dim: int) -> None:
        """ArcFace (512) and SFace (128) embeddings are not comparable.

        Silently mixing them would make every match fail in a way that looks
        like "the recogniser got worse", so fail loudly instead.
        """
        if self._dim is not None and self._dim != dim:
            raise SystemExit(
                f"Gallery at {self.path} holds {self._dim}-d embeddings but the "
                f"current face backend produces {dim}-d. Embeddings from different "
                "models cannot be compared. Either switch back to the original "
                f"backend, or move {self.path} aside and re-enroll."
            )

    # ---------- matching ----------

    def match(self, embedding: np.ndarray) -> tuple[str | None, float]:
        """Return (best identity id, cosine similarity). (None, -1) if empty."""
        best_pid, best_sim = None, -1.0
        for pid, idt in self.identities.items():
            if not idt.embeddings:
                continue
            sim = float(np.max(idt.matrix() @ embedding))
            if sim > best_sim:
                best_pid, best_sim = pid, sim
        return best_pid, best_sim

    def enroll(self, embedding: np.ndarray, label: str | None = None) -> Identity:
        pid = f"{self.prefix}{self._next_num}"
        num = self._next_num
        self._next_num += 1
        idt = Identity(
            pid=pid, label=label or f"{self.label_word} {num}", embeddings=[embedding]
        )
        self.identities[pid] = idt
        self._dim = int(embedding.shape[0])
        return idt

    def reinforce(self, pid: str, embedding: np.ndarray, quality: float) -> None:
        """Add a new view of a known face, keeping the bag diverse.

        We only keep an embedding if it is *not* already well covered by what we
        have. Otherwise the bag fills up with twelve copies of the same frontal
        pose and stops helping.
        """
        idt = self.identities.get(pid)
        if idt is None:
            return
        idt.last_seen = time.time()
        idt.sightings += 1
        if quality < 0.5:
            return
        if idt.embeddings:
            redundancy = float(np.max(idt.matrix() @ embedding))
            if redundancy > 0.82:
                return
        idt.embeddings.append(embedding)
        if len(idt.embeddings) > self.max_per_identity:
            idt.embeddings.pop(0)

    def rename(self, pid: str, label: str) -> None:
        if pid in self.identities:
            self.identities[pid].label = label
            self.save()

    def label_of(self, pid: str | None) -> str:
        if pid is None:
            return "unknown"
        idt = self.identities.get(pid)
        return idt.label if idt else pid
