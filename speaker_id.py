"""
speaker_id.py — Speaker identification and diarization for voice-inject.

Provides:
  SpeakerIdentifier — per-segment speaker identification via pyannote/embedding
  SpeakerDiarizer   — speaker turn detection via pyannote/segmentation
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, TYPE_CHECKING

import numpy as np
import torch

# Conditional imports for typing
if TYPE_CHECKING:
    from .speaker_db import SpeakerDB

logger = logging.getLogger(__name__)

class SpeakerIdentifier:
    """Identifies speakers by comparing segment embeddings to an enrollment database."""

    def __init__(
        self,
        speaker_db: SpeakerDB,
        model_name: str = "pyannote/embedding",
        use_auth_token: Optional[str | bool] = True,
        device: Optional[torch.device] = None,
        threshold: float = 0.75,
    ) -> None:
        from pyannote.audio import Model
        from pyannote.audio.pipelines import SpeakerVerification

        self.db = speaker_db
        self.similarity_threshold = threshold
        
        # We handle device internally in the worker thread for MLX apps, 
        # but pyannote-audio still uses torch.
        if device is None:
            if torch.backends.mps.is_available():
                device = torch.device("mps")
            elif torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")
        self.device = device

        logger.info("Loading SpeakerIdentifier model '%s' on %s...", model_name, self.device)
        self._model = Model.from_pretrained(model_name, use_auth_token=use_auth_token).to(self.device)
        self._verify = SpeakerVerification(model=self._model, threshold=self.similarity_threshold)

    def get_embedding(self, audio_data: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        """Extract a pyannote embedding vector (512-dim) from raw mono audio."""
        # Convert to torch tensor: (channels, samples)
        tensor = torch.from_numpy(audio_data.astype(np.float32)).unsqueeze(0).to(self.device)
        # Wrap in pyannote-style dict
        with torch.no_grad():
            embedding = self._model(tensor)
        # embedding is (1, 512)
        return embedding.cpu().numpy()[0]

    def identify(self, audio_data: np.ndarray, sample_rate: int = 16000) -> str:
        """Identify the speaker in *audio_data*. Returns name or 'Unknown'."""
        emb = self.get_embedding(audio_data, sample_rate)
        name, similarity = self.db.find_closest(emb)
        
        if name and similarity is not None and similarity >= self.similarity_threshold:
            logger.debug("Identified speaker '%s' (similarity=%.3f).", name, similarity)
            return name
            
        logger.debug("Speaker unknown (best similarity: %.3f < %.3f).", 
                     similarity if similarity is not None else 0.0,
                     self.similarity_threshold)
        return "Unknown"

class SpeakerDiarizer:
    """Simple VAD-based diarization / turn detection using pyannote."""

    def __init__(
        self,
        model_name: str = "pyannote/segmentation-3.0",
        use_auth_token: Optional[str | bool] = True,
        device: Optional[torch.device] = None,
    ) -> None:
        from pyannote.audio import Model
        from pyannote.audio.pipelines import VoiceActivityDetection
        
        if device is None:
            if torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")
        self.device = device

        logger.info("Loading SpeakerDiarizer model '%s' on %s...", model_name, self.device)
        self._model = Model.from_pretrained(model_name, use_auth_token=use_auth_token).to(self.device)
        self._vad = VoiceActivityDetection(segmentation=self._model)
        
        # Standard hypers for VAD
        self._vad.instantiate({
            "onset": 0.5,
            "offset": 0.5,
            "min_duration_on": 0.3,
            "min_duration_off": 0.3,
        })

    def get_speech_segments(self, audio_data: np.ndarray, sample_rate: int = 16000):
        """Detect speech segments in a buffer. Returns pyannote Annotation."""
        # Convert to torch tensor: (channels, samples)
        tensor = torch.from_numpy(audio_data.astype(np.float32)).unsqueeze(0).to(self.device)
        # Wrap for pyannote
        audio_dict = {"waveform": tensor, "sample_rate": sample_rate}
        return self._vad(audio_dict)
