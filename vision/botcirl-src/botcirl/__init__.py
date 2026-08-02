"""botcirl - room perception prototype.

Stage 1 (this code): who is in the room, where they are, and who is who.
Stage 2 (not built yet): who is speaking, and who they are speaking to.
"""

from .config import Config
from .pipeline import Pipeline, PersonTrack, Scene

__all__ = ["Config", "Pipeline", "PersonTrack", "Scene"]
