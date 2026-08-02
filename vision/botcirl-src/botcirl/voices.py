"""Speaker embeddings - the voice half of identity.

Deliberately the same shape as `faces.py`: a backend turns a signal into an
L2-normalised vector, and `IdentityGallery` does the rest. Cosine similarity,
enrol-when-new, reinforce-when-seen, persist to disk - all of it is reused, so a
voice gallery and a face gallery behave identically and can be reasoned about
together.

The two galleries stay separate on disk. Linking "voice 3 is face 2" is a
different problem with different evidence, and fusing them prematurely would
throw away the ability to say "I heard someone I know but cannot see them".

Duration is the thing to watch. A speaker embedding from half a second of audio
is close to noise; from three seconds it is solid. Everything here gates on it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .audio import SpeechSegment
from .config import VoiceConfig


@dataclass
class VoicePrint:
    embedding: np.ndarray  # L2-normalised
    duration: float
    rms: float

    def quality(self) -> float:
        """0..1 confidence that this embedding is worth trusting.

        Almost entirely about how much voice there was. Level matters only at
        the extremes - a whisper across the room carries little speaker
        information, and clipping destroys it.
        """
        dur = float(np.clip((self.duration - 0.5) / 2.0, 0.0, 1.0))
        level = float(np.clip(self.rms / 0.02, 0.0, 1.0))
        if self.rms > 0.35:  # clipping or blowing into the mic
            level *= 0.5
        return float(dur * (0.4 + 0.6 * level))


class SpeechBrainECAPA:
    """ECAPA-TDNN trained on VoxCeleb. The standard speaker-recognition model."""

    name = "speechbrain-ecapa"
    embedding_dim = 192
    # Cosine thresholds for ECAPA on short utterances. Verified against real
    # recordings before being trusted - see tests and the README.
    default_match_threshold = 0.45
    default_enroll_threshold = 0.30

    def __init__(self, cfg: VoiceConfig):
        import torch
        from speechbrain.inference.speaker import EncoderClassifier

        self.cfg = cfg
        self.torch = torch
        self.model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=str(cfg.model_dir),
            run_opts={"device": "cpu"},
        )

    def embed(self, audio: np.ndarray, samplerate: int = 16000) -> np.ndarray | None:
        if audio.size < int(self.cfg.min_embed_s * samplerate):
            return None
        wav = self.torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32))[None, :]
        with self.torch.no_grad():
            emb = self.model.encode_batch(wav).squeeze().cpu().numpy().astype(np.float32)
        norm = float(np.linalg.norm(emb))
        if norm < 1e-6:
            return None
        return emb / norm


def build_voice_backend(cfg: VoiceConfig):
    backend = SpeechBrainECAPA(cfg)
    print(f"[voices] backend: {backend.name} (ECAPA-TDNN / VoxCeleb)")
    return backend


def voiceprint(backend, segment: SpeechSegment) -> VoicePrint | None:
    """Embed a speech segment, or None if there is not enough voice in it."""
    emb = backend.embed(segment.audio, segment.samplerate)
    if emb is None:
        return None
    return VoicePrint(embedding=emb, duration=segment.duration, rms=segment.rms())
